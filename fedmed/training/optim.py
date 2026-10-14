"""Optimizer and learning rate scheduler factories for centralized and federated training.

Note on federated learning:
In federated rounds (Prompt 5), the scheduler is driven by the global round index,
not the local client epoch count.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from fedmed.utils.logging import get_logger

logger = get_logger(__name__)


class PolyDecayWithMinLR(LRScheduler):
    """Polynomial learning rate decay scheduler where final rate equals min_lr.

    Formula:
        lr = min_lr + (base_lr - min_lr) * ((1 - min(total_iters, epoch) / total_iters) ** power)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        min_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_iters = max(1, total_iters)
        self.min_lr = float(min_lr)
        self.power = float(power)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [
            self.min_lr
            + (base_lr - self.min_lr)
            * (
                (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters)
                ** self.power
            )
            for base_lr in self.base_lrs
        ]


def get_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Partition model parameters into weight-decay and no-decay groups.

    Biases and normalization weights (GroupNorm, InstanceNorm, LayerNorm, BatchNorm)
    as well as 1D tensors are excluded from weight decay.

    Args:
        model: Target PyTorch model.
        weight_decay: Weight decay penalty for kernel/weight parameters.

    Returns:
        List of parameter group dictionaries.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    norm_modules = (
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    )

    norm_param_ids: set[int] = set()
    for m in model.modules():
        if isinstance(m, norm_modules):
            for p in m.parameters(recurse=False):
                norm_param_ids.add(id(p))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            id(param) in norm_param_ids
            or name.endswith(".bias")
            or name == "bias"
            or param.ndim <= 1
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay)})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    """Build optimizer with bias and normalization weights excluded from weight decay.

    Supported optimizers: 'adamw' (default), 'adam', 'sgd'.

    Args:
        model: Model whose parameters will be optimized.
        cfg: Validated experiment configuration mapping.

    Returns:
        Configured PyTorch Optimizer instance.
    """
    train_cfg = cfg.get("training", {})
    opt_cfg = train_cfg.get("optimizer", {})

    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", train_cfg.get("lr", 2e-4)))
    weight_decay = float(opt_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-5)))

    param_groups = get_parameter_groups(model, weight_decay=weight_decay)

    if opt_name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return AdamW(param_groups, lr=lr, betas=betas)
    elif opt_name == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        return Adam(param_groups, lr=lr, betas=betas)
    elif opt_name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = bool(opt_cfg.get("nesterov", False))
        return SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        raise ValueError(
            f"Unsupported optimizer '{opt_name}'. Expected 'adamw', 'adam', or 'sgd'."
        )


def build_scheduler(
    optimizer: Optimizer,
    cfg: Mapping[str, Any],
    steps_per_epoch: int | None = None,
) -> LRScheduler | None:
    """Build per-epoch learning rate scheduler with linear warmup and cosine/poly decay.

    In federated training rounds, the scheduler is stepped once per global round,
    not per local client epoch (implemented in Prompt 5).

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        cfg: Configuration mapping with training and scheduler settings.
        steps_per_epoch: Optional step count per epoch (reserved for step-based schedulers).

    Returns:
        Configured LRScheduler or None if scheduler.name is 'none'.
    """
    train_cfg = cfg.get("training", {})
    sched_cfg = train_cfg.get("scheduler", {})
    sched_name = str(sched_cfg.get("name", "cosine")).lower()

    if sched_name == "none":
        return None

    total_epochs = int(train_cfg.get("epochs", 5))
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
    min_lr = float(sched_cfg.get("min_lr", 0.0))
    power = float(sched_cfg.get("poly_power", 1.0))

    base_lr = float(train_cfg.get("optimizer", {}).get("lr", train_cfg.get("lr", 2e-4)))

    if warmup_epochs > 0:
        start_factor = (
            (min_lr / base_lr) if (base_lr > 0 and 0 < min_lr < base_lr) else 0.01
        )
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        decay_iters = max(1, total_epochs - warmup_epochs)
        if sched_name == "cosine":
            decay = CosineAnnealingLR(optimizer, T_max=decay_iters, eta_min=min_lr)
        elif sched_name == "poly":
            decay = PolyDecayWithMinLR(
                optimizer, total_iters=decay_iters, min_lr=min_lr, power=power
            )
        else:
            raise ValueError(f"Unknown scheduler '{sched_name}'")

        return SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_epochs],
        )

    # No warmup phase
    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=min_lr)
    elif sched_name == "poly":
        return PolyDecayWithMinLR(
            optimizer, total_iters=max(1, total_epochs), min_lr=min_lr, power=power
        )
    else:
        raise ValueError(f"Unknown scheduler '{sched_name}'")


def get_lr(optimizer: Optimizer) -> float:
    """Return the current learning rate from the first parameter group of the optimizer.

    Args:
        optimizer: PyTorch optimizer instance.

    Returns:
        Current learning rate as a float.
    """
    return float(optimizer.param_groups[0]["lr"])
