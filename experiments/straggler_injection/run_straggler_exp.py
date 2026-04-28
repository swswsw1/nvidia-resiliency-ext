"""
Straggler injection experiment for FR-based detection POC.

Based on Megatron-LM's examples/run_simple_mcore_train_loop.py.
Runs GPT training with TP=2 PP=1 (DP = world_size / 2). Injects host-side
or kernel-side stragglers on a configurable rank. Trigger-driven: enable
FR_CHEAP_STATS_TRIGGER=1 and the trainer will dump exactly one
default_pg-bracketed block per persistent-straggler detection.

Per-iter `dist.barrier()` is intentional — it's the only world-group call
in the loop, so it's what the cheap-stats trigger times. Without it the
trigger has nothing to observe.

Usage:
    FR_CHEAP_STATS_TRIGGER=1 TORCH_NCCL_ENABLE_TIMING=1 \
    torchrun --nproc_per_node=4 run_straggler_exp.py \
        --inject-type host --inject-rank 3 --inject-delay-ms 50 \
        --output-dir ./traces_host
"""

import argparse
import os
import time
from functools import partial
from typing import Callable, Dict, Iterator, Tuple

# Cheap-stats trigger: install BEFORE importing torch.distributed-using
# framework code (Megatron / DDP) so dist.{all_reduce, barrier, broadcast}
# wrappers are seen. Env-gated so the existing pipeline runs unchanged
# unless explicitly enabled.
_TRIGGER = None
if os.environ.get("FR_CHEAP_STATS_TRIGGER", "0") == "1":
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fr_cheap_stats_trigger import StragglerTrigger as _StragglerTrigger  # noqa: E402

    _TRIGGER = _StragglerTrigger(
        window_size=int(os.environ.get("FR_TRIG_WINDOW", "10")),
        check_freq=int(os.environ.get("FR_TRIG_CHECK_FREQ", "3")),
        persistence=int(os.environ.get("FR_TRIG_PERSISTENCE", "2")),
        device_id=int(os.environ.get("LOCAL_RANK", "0")),
        log_path=os.environ.get("FR_TRIG_LOG"),
    )
    _TRIGGER.patch_distributed()

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

    world = dist.get_world_size()
    if rank == 0:
        print(f"Config: inject_type={args.inject_type}, inject_rank={args.inject_rank}, "
              f"inject_delay_ms={args.inject_delay_ms}, output_dir={args.output_dir}")
        print(f"World size: {world}, TP=2, PP=1, DP={world // 2}")

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
    os.makedirs(args.output_dir, exist_ok=True)

    # Precompute injection parameters
    delay_s = args.inject_delay_ms / 1000.0
    gpu_delay_cycles = estimate_gpu_cycles(args.inject_delay_ms)
    # Support comma-separated ranks via env var INJECT_RANKS for multi-injection tests.
    env_ranks = os.environ.get("INJECT_RANKS", "").strip()
    if env_ranks:
        inject_rank_set = {int(r) for r in env_ranks.split(",") if r}
    else:
        inject_rank_set = {args.inject_rank}
    is_inject_rank = rank in inject_rank_set

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

        # Per-iter world-group barrier. Doubles as the default_pg event the
        # cheap-stats trigger times — without it the trigger has no
        # world-group call to observe.
        dist.barrier()

        # Cheap-stats trigger: dump exactly one default_pg-bracketed block
        # when persistent straggler signal detected. Idempotent within an iter.
        if _TRIGGER is not None and _TRIGGER.consume_dump_request():
            from fr_cheap_stats_trigger import dump_one_block
            out = dump_one_block(rank, args.output_dir, iteration)
            if rank == 0:
                print(f"[trigger] iter={iteration} dumped: {out}")

        iter_time = time.time() - iter_start
        if rank == 0:
            print(f"Iteration {iteration}: loss={losses_reduced}, time={iter_time:.3f}s")

    dist.barrier()

    # Cleanup
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
