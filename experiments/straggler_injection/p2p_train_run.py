"""
PP=2 straggler injection experiment using Megatron-Core.

Parallelism: TP=2, PP=2, DP=2  →  world_size = 8
  stage 0 (ranks 0-3): embedding + layers 0-1
  stage 1 (ranks 4-7): layers 2-3 + lm_head / loss

Synthetic data (torch.randint) — avoids DataLoader/MockGPTDataset PP hangs.

Usage:
    TORCH_NCCL_TRACE_BUFFER_SIZE=10000 TORCH_NCCL_ENABLE_TIMING=1 \\
    torchrun --nproc_per_node=8 p2p_train_run.py \\
        --inject-type host --inject-rank 3 --inject-delay-ms 50 \\
        --output-dir ./traces/run1
"""

import argparse
import json
import os
import pickle
import sys
import time

# Cheap-stats trigger: install BEFORE importing torch.distributed-using
# framework code (Megatron / DDP) so dist.{all_reduce, barrier, broadcast}
# wrappers are seen. Env-gated so the existing pipeline runs unchanged
# unless explicitly enabled. Mirrors run_straggler_exp.py.
_TRIGGER = None
if os.environ.get("FR_CHEAP_STATS_TRIGGER", "0") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fr_cheap_stats_trigger import StragglerTrigger as _StragglerTrigger  # noqa: E402

    _TRIGGER = _StragglerTrigger(
        window_size=int(os.environ.get("FR_TRIG_WINDOW", "10")),
        check_freq=int(os.environ.get("FR_TRIG_CHECK_FREQ", "3")),
        persistence=int(os.environ.get("FR_TRIG_PERSISTENCE", "2")),
        device_id=int(os.environ.get("LOCAL_RANK", "0")),
        log_path=os.environ.get("FR_TRIG_LOG"),
        offset_ms=float(os.environ.get("FR_TRIG_OFFSET_MS", "1.0")),
    )
    _TRIGGER.patch_distributed()

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
# Parallelism (env-overridable for partial-cluster runs)
# ---------------------------------------------------------------------------
TP = int(os.environ.get("TP", "2"))
PP = int(os.environ.get("PP", "2"))
DP = int(os.environ.get("DP", "2"))
WORLD_SIZE = TP * PP * DP

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
    parser.add_argument("--inject-rank", type=str, default="3",
                        help="comma-separated rank list, e.g. '3' or '2,3'")
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
    world = torch.distributed.get_world_size()

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
    if rank != 0:
        return

    # Post-hoc end-of-run slicing — only when the cheap-stats trigger is OFF.
    # When the trigger is ON, per-block dumps are produced live during training
    # via `dump_one_block` and the trigger-driven step-format files already
    # populate `output_dir`; running the slicer here would collide with them
    # and produce conflicting verdicts. With the trigger OFF we still want
    # per-block analysis available, so we slice the full FR ring buffer into
    # per-iteration blocks using fr_block_slicer, then write each (block, rank)
    # into the step-format `_dump_{rank}_step{N:06d}.json` that
    # fr_analyze_dumps.py consumes. The original `_dump_{rank}.json` files are
    # kept under `full_dumps/` for backup/inspection so the analyzer's glob
    # doesn't accidentally pick them up alongside the per-block files.
    if _TRIGGER is not None:
        print(
            f"FR traces dumped to {output_dir}/ ({world} ranks); "
            f"trigger ON → per-block files came from trigger, skipping post-hoc slice.",
            flush=True,
        )
        return

    from fr_block_slicer import slice_into_blocks

    per_rank_merged: dict = {}
    for r in range(world):
        with open(os.path.join(output_dir, f"_dump_{r}.json")) as f:
            per_rank_merged[r] = json.load(f)

    blocks = slice_into_blocks(per_rank_merged)

    full_dir = os.path.join(output_dir, "full_dumps")
    os.makedirs(full_dir, exist_ok=True)
    for r in range(world):
        src = os.path.join(output_dir, f"_dump_{r}.json")
        dst = os.path.join(full_dir, f"_dump_{r}.json")
        os.rename(src, dst)

    for block in blocks:
        bid = block["block_id"]
        for r, rank_data in block["by_rank"].items():
            out_path = os.path.join(output_dir, f"_dump_{r}_step{bid:06d}.json")
            with open(out_path, "w") as f:
                json.dump(rank_data, f)
                os.fsync(f.fileno())

    print(
        f"FR traces dumped + sliced to {output_dir}/ "
        f"({world} ranks, {len(blocks)} blocks; full dumps in full_dumps/)",
        flush=True,
    )


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

    inject_ranks = {int(r) for r in args.inject_rank.split(",") if r.strip()}
    is_inject_rank = (rank in inject_ranks)
    gpu_cycles = estimate_gpu_cycles(args.inject_delay_ms)
    delay_s = args.inject_delay_ms / 1000.0

    if rank == 0:
        print(f"inject_type={args.inject_type} inject_ranks={sorted(inject_ranks)} "
              f"delay={args.inject_delay_ms}ms iterations={args.num_iterations}",
              flush=True)

    for iteration in range(args.num_iterations):
        # --- straggler injection ---
        if args.inject_type == "host" and is_inject_rank:
            time.sleep(delay_s)
        elif args.inject_type == "kernel" and is_inject_rank:
            torch.cuda._sleep(gpu_cycles)

        data_iter = iter(get_batch, None)   # infinite: callable never returns None
        optimizer.zero_grad()

        losses = fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=data_iter,
            model=[model],
            num_microbatches=2,
            seq_length=SEQ_LEN,
            micro_batch_size=MICRO_BATCH,
            forward_only=False,
        )

        finalize_model_grads([model])
        optimizer.step()

        # Per-iter world-group barrier — produces the default_pg event that
        # bounds each iteration into one block for the analyzer (and that
        # the cheap-stats trigger times). See run_straggler_exp.py for the
        # same pattern in TP-only runs.
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

        # Cheap-stats trigger: dump exactly one default_pg-bracketed block
        # when persistent straggler signal detected. Idempotent within an iter.
        if _TRIGGER is not None and _TRIGGER.consume_dump_request():
            from fr_cheap_stats_trigger import dump_one_block
            out = dump_one_block(rank, args.output_dir, iteration)
            if rank == 0:
                print(f"[trigger] iter={iteration} dumped: {out}", flush=True)

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