"""
PP=2 straggler injection experiment using Megatron-Core.

Parallelism: TP=2, PP=2, DP=2  →  world_size = 8
  stage 0 (ranks 0-3): embedding + layers 0-1
  stage 1 (ranks 4-7): layers 2-3 + lm_head / loss

Synthetic data (torch.randint) — avoids DataLoader/MockGPTDataset PP hangs.

Usage:
    TORCH_NCCL_TRACE_BUFFER_SIZE=10000 TORCH_NCCL_ENABLE_TIMING=1 \\
    torchrun --nproc_per_node=8 run_simple_pp2_train_loop.py \\
        --inject-type host --inject-rank 3 --inject-delay-ms 50 \\
        --output-dir ./traces/run1
"""

import argparse
import json
import os
import pickle
import time

import torch
import torch.distributed
import torch.multiprocessing as mp

from megatron.core import parallel_state
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
TP = 2
PP = 2
DP = 2
WORLD_SIZE = TP * PP * DP  # 8

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
NUM_LAYERS = 4       # divisible by PP → 2 layers per stage
HIDDEN_SIZE = 256
NUM_HEADS = 4
VOCAB_SIZE = 1024
SEQ_LEN = 64

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MICRO_BATCH = 2
NUM_ITERATIONS = 30
LR = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject-type", choices=["none", "host", "kernel"],
                        default="none")
    parser.add_argument("--inject-rank", type=int, default=3)
    parser.add_argument("--inject-delay-ms", type=float, default=50.0)
    parser.add_argument("--output-dir", type=str, default="./traces")
    parser.add_argument("--num-iterations", type=int, default=NUM_ITERATIONS)
    return parser.parse_args()


def estimate_gpu_cycles(delay_ms: float) -> int:
    """H200 boost clock ~2.5 GHz."""
    return int(delay_ms * 1e-3 * 2.5e9)


def make_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        pipeline_dtype=torch.float32,
    )


def build_model(config: TransformerConfig) -> GPTModel:
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )


def get_batch():
    tokens = torch.randint(0, VOCAB_SIZE, (MICRO_BATCH, SEQ_LEN), device="cuda")
    labels = torch.randint(0, VOCAB_SIZE, (MICRO_BATCH, SEQ_LEN), device="cuda")
    position_ids = (
        torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(MICRO_BATCH, -1)
    )
    return tokens, labels, position_ids


def forward_step(data_iterator, model):
    tokens, labels, position_ids = next(data_iterator)
    output = model(tokens, position_ids, attention_mask=None, labels=labels)

    def loss_func(output_tensor):
        loss = output_tensor.mean()
        return loss, {"loss": loss.detach()}

    return output, loss_func


def dump_fr_traces(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rank = torch.distributed.get_rank()

    trace_bytes = torch._C._distributed_c10d._dump_nccl_trace(
        includeCollectives=True,
        includeStackTraces=False,
        onlyActive=False,
    )
    trace_dict = pickle.loads(trace_bytes)

    path = os.path.join(output_dir, f"_dump_{rank}.json")
    with open(path, "w") as f:
        json.dump(trace_dict, f, indent=4)
        os.fsync(f.fileno())

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    if rank == 0:
        print(f"FR traces dumped to {output_dir}/ ({torch.distributed.get_world_size()} ranks)",
              flush=True)


# ---------------------------------------------------------------------------
# Per-rank training function
# ---------------------------------------------------------------------------

def train(rank: int, world_size: int, args) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
    )
    model_parallel_cuda_manual_seed(42)

    config = make_config()
    model = build_model(config).cuda()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    model = DistributedDataParallel(config=config, ddp_config=ddp_config, module=model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    fwd_bwd = get_forward_backward_func()

    is_inject_rank = (rank == args.inject_rank)
    gpu_cycles = estimate_gpu_cycles(args.inject_delay_ms)
    delay_s = args.inject_delay_ms / 1000.0

    if rank == 0:
        print(f"inject_type={args.inject_type} inject_rank={args.inject_rank} "
              f"delay={args.inject_delay_ms}ms iterations={args.num_iterations}",
              flush=True)

    for iteration in range(args.num_iterations):
        # --- straggler injection ---
        if args.inject_type == "host" and is_inject_rank:
            time.sleep(delay_s)
        elif args.inject_type == "kernel" and is_inject_rank:
            torch.cuda._sleep(gpu_cycles)

        data_iter = iter([get_batch()])
        optimizer.zero_grad()

        losses = fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=data_iter,
            model=[model],
            num_microbatches=1,
            seq_length=SEQ_LEN,
            micro_batch_size=MICRO_BATCH,
            forward_only=False,
        )

        finalize_model_grads([model])
        optimizer.step()

        if losses:
            print(
                f"[rank {rank}] iter {iteration:3d} | loss = {losses[0]['loss'].item():.4f}",
                flush=True,
            )

    # --- FR trace dump ---
    dump_fr_traces(args.output_dir)

    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if "RANK" in os.environ:
        train(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args)
    else:
        mp.spawn(train, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
