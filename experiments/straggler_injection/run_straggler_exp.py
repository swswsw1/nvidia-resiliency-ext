"""
Straggler injection experiment for FR-based detection POC.

Based on Megatron-LM's examples/run_simple_mcore_train_loop.py.
Runs GPT training with TP=2, PP=1, DP=4 on 8 GPUs.
Injects host-side or kernel-side stragglers on a configurable rank.
Dumps FR traces with timing info at the end.

Usage:
    TORCH_NCCL_ENABLE_TIMING=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
    torchrun --nproc_per_node=8 run_straggler_exp.py \
        --inject-type host --inject-rank 3 --inject-delay-ms 50 \
        --output-dir ./traces_host
"""

import argparse
import json
import os
import pickle
import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, Tuple

import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.data import DataLoader

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.transformer.transformer_config import TransformerConfig


_SEQUENCE_LENGTH = 64
_NUM_LAYERS = 4       # divisible by PP=2 → 2 layers per stage
_HIDDEN_SIZE = 256    # large enough for meaningful collectives
_NUM_ATTN_HEADS = 8
_VOCAB_SIZE = 1024
_BATCH_SIZE = 8
_NUM_MICROBATCHES = 1
_NUM_ITERATIONS = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Straggler injection experiment")
    parser.add_argument(
        "--inject-type",
        type=str,
        choices=["none", "host", "kernel"],
        default="none",
        help="Type of straggler to inject",
    )
    parser.add_argument(
        "--inject-rank",
        type=int,
        default=3,
        help="Global rank to inject straggler on",
    )
    parser.add_argument(
        "--inject-delay-ms",
        type=float,
        default=50.0,
        help="Delay magnitude in milliseconds",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./traces",
        help="Directory to dump FR traces",
    )
    return parser.parse_args()


def initialize_distributed(
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 1,
) -> None:
    parallel_state.destroy_model_parallel()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def model_provider() -> GPTModel:
    transformer_config = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_ATTN_HEADS,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_SEQUENCE_LENGTH,
    )
    return gpt_model


def get_train_data_iterator() -> Iterator:
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            compile_helpers()
        dist.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=MegatronTokenizer.from_pretrained(
            metadata_path={"library": "null-text"},
            vocab_size=_VOCAB_SIZE,
        ),
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=_BATCH_SIZE, shuffle=True)
    return iter(train_dataloader)


def forward_step_func(
    data_iterator: Iterator, model: torch.nn.Module
) -> Tuple[torch.Tensor, Callable]:
    def loss_func(
        loss_mask: torch.Tensor, output_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm loss": loss}

    data = next(data_iterator)
    device = torch.device("cuda")
    tokens = data["tokens"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    loss_mask = data["loss_mask"].to(device)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    return output_tensor, partial(loss_func, loss_mask)


def dump_fr_traces(output_dir: str):
    """Dump FR traces from all ranks to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    rank = dist.get_rank()

    trace_bytes = torch._C._distributed_c10d._dump_nccl_trace(
        includeCollectives=True,
        includeStackTraces=False,
        onlyActive=False,
    )
    trace_dict = pickle.loads(trace_bytes)

    output_path = os.path.join(output_dir, f"_dump_{rank}.json")
    with open(output_path, "w") as f:
        json.dump(trace_dict, f, indent=4)
        os.fsync(f.fileno())

    dist.barrier()
    if rank == 0:
        print(f"FR traces dumped to {output_dir}/ ({dist.get_world_size()} ranks)")


def estimate_gpu_cycles(delay_ms: float) -> int:
    """Estimate GPU clock cycles for a given delay in ms.

    B200 boost clock is ~2.1 GHz. We estimate conservatively and
    the actual delay can be calibrated from experiment results.
    """
    clock_ghz = 2.1  # approximate B200 boost clock
    cycles = int(delay_ms * 1e-3 * clock_ghz * 1e9)
    return cycles


def main():
    args = parse_args()

    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    rank = dist.get_rank()
    device = torch.device("cuda")

    if rank == 0:
        print(f"Config: inject_type={args.inject_type}, inject_rank={args.inject_rank}, "
              f"inject_delay_ms={args.inject_delay_ms}, output_dir={args.output_dir}")
        print(f"World size: {dist.get_world_size()}, TP=2, PP=1, DP=4")

    gpt_model = model_provider()
    gpt_model.to(device)

    config = gpt_model.config
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    gpt_model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=gpt_model,
    )

    optim = Adam(gpt_model.parameters())
    train_iterator = get_train_data_iterator()
    forward_backward_func = get_forward_backward_func()

    # Precompute injection parameters
    delay_s = args.inject_delay_ms / 1000.0
    gpu_delay_cycles = estimate_gpu_cycles(args.inject_delay_ms)
    is_inject_rank = (rank == args.inject_rank)

    if is_inject_rank and args.inject_type != "none":
        print(f"[Rank {rank}] Will inject {args.inject_type} straggler: "
              f"delay={args.inject_delay_ms}ms"
              f"{f', gpu_cycles={gpu_delay_cycles}' if args.inject_type == 'kernel' else ''}")

    for iteration in range(_NUM_ITERATIONS):
        iter_start = time.time()

        # === HOST-SIDE INJECTION ===
        # time.sleep() delays the CPU → late time_created_ns for all
        # subsequent collectives this iteration
        if args.inject_type == "host" and is_inject_rank:
            time.sleep(delay_s)

        # === KERNEL-SIDE INJECTION ===
        # torch.cuda._sleep() blocks the GPU stream but CPU returns immediately
        # → normal time_created_ns, late time_discovered_started_ns
        if args.inject_type == "kernel" and is_inject_rank:
            torch.cuda._sleep(gpu_delay_cycles)

        optim.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=_NUM_MICROBATCHES,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=_BATCH_SIZE,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )

        finalize_model_grads([gpt_model])
        optim.step()

        iter_time = time.time() - iter_start
        if rank == 0:
            print(f"Iteration {iteration}: loss={losses_reduced}, time={iter_time:.3f}s")

    # Dump FR traces
    dump_fr_traces(args.output_dir)

    # Cleanup
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
