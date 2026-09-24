"""Single-process multi-GPU support (Kaggle's "GPU T4 x2").

`nn.DataParallel` prefixes every key in `state_dict()` with `module.`. If a checkpoint is
written from a wrapped model it will not load into an unwrapped one, and vice versa --
`init_from` would report hundreds of unexpected keys, or silently load almost nothing and
train from noise. That failure is quiet, so the rule here is absolute:

    ALWAYS save and load the UNWRAPPED state_dict.

Checkpoints are then wrapper-agnostic: every artefact already on HF (QACP v3/v4, any
Stage-1 checkpoint) keeps loading on one GPU or two, in either direction.

DataParallel rather than DistributedDataParallel because a Kaggle notebook is a single
process; DDP would need torchrun or a spawn harness for a modest gain at two devices.
DataParallel replicates the model per batch and splits along dim 0, so the per-device
batch is `batch_size // n_gpus` -- `parallel_batch_warning` exists because a batch that
does not divide cleanly silently wastes a device.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def n_visible_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def unwrap(model: nn.Module) -> nn.Module:
    """The underlying module, whether or not `model` is a DataParallel wrapper.

    Use at EVERY model state_dict save/load site.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_parallel(model: nn.Module, device: str = "cuda", enabled: bool = True) -> nn.Module:
    """Wrap in DataParallel when more than one GPU is visible, else return unchanged.

    Call AFTER `.to(device)`, AFTER `init_from`, and AFTER gradient checkpointing is
    enabled -- those all reach into the real module and are clearer before wrapping.
    """
    n = n_visible_gpus()
    if not enabled or device != "cuda" or n < 2:
        if n >= 2 and not enabled:
            print(f"[parallel] {n} GPUs visible but data_parallel=false — using cuda:0 only")
        return model
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
    print(f"[parallel] DataParallel across {n} GPUs: {names}")
    return nn.DataParallel(model)


def parallel_batch_warning(batch_size: int, model: nn.Module) -> None:
    """Warn when the batch does not split evenly across the devices.

    DataParallel scatters along dim 0. A batch of 4 over 2 GPUs is 2 each, which is
    small enough to hurt BatchNorm-free attention throughput; a batch smaller than the
    device count leaves GPUs with nothing to do, which is also what makes Kaggle reclaim
    a session for under-using its accelerators.
    """
    if not isinstance(model, nn.DataParallel):
        return
    n = len(model.device_ids)
    if batch_size < n:
        print(f"[parallel] WARNING batch_size={batch_size} < {n} GPUs — "
              f"{n - batch_size} device(s) will idle. Raise batch_size to at least {n}.")
    elif batch_size % n:
        print(f"[parallel] WARNING batch_size={batch_size} does not divide across {n} GPUs — "
              f"the last device gets a short batch. Use a multiple of {n}.")
    else:
        print(f"[parallel] per-device batch: {batch_size // n} ({batch_size} split over {n} GPUs)")
