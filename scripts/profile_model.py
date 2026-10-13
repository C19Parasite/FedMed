"""Model profiling CLI evaluating runtime, parameters, memory, and GPU headroom.

CLI Arguments:
  --config PATH: Path to YAML configuration file (e.g. configs/base.yaml)
  --device [cpu|cuda]: Execution device (defaults to cuda if available, else cpu)
  --amp: Enable Automatic Mixed Precision (torch.cuda.amp.autocast)
  --sweep: Sweep across batch sizes [1, 2, 4] and report memory/time for each
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from fedmed.models.factory import build_model
from fedmed.models.params import count_parameters, model_size_mb
from fedmed.training.losses import build_loss
from fedmed.utils.config import load_config
from fedmed.utils.logging import get_logger

logger = get_logger("profile_model")

# Standard GPU VRAM thresholds in MB
GPU_8GB_MB = 8192.0
GPU_12GB_MB = 12288.0


def profile_single_run(
    model: nn.Module,
    loss_fn: nn.Module,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    roi_size: list[int],
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Execute one warm-up and one timed forward+backward pass."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Synthetic input and target tensors
    x = torch.randn(batch_size, in_channels, *roi_size, device=device)
    y = (torch.rand(batch_size, out_channels, *roi_size, device=device) > 0.5).float()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = use_amp and (device.type == "cuda")

    # Warmup pass
    optimizer.zero_grad()
    with torch.autocast(device_type=device_type, enabled=amp_enabled):
        out_warmup = model(x)
        loss_warmup = loss_fn(out_warmup, y)
    loss_warmup.backward()
    optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Measured pass
    t0 = time.perf_counter()
    with torch.autocast(device_type=device_type, enabled=amp_enabled):
        out = model(x)
        loss = loss_fn(out, y)
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

    out_shape = list(out.shape) if isinstance(out, torch.Tensor) else [list(o.shape) for o in out]

    return {
        "batch_size": batch_size,
        "out_shape": out_shape,
        "elapsed_ms": elapsed_ms,
        "peak_mem_mb": peak_mem_mb,
    }


def format_gpu_recommendation(peak_mem_mb: float, is_cuda: bool) -> str:
    """Provide clear suitability recommendation for 8GB and 12GB GPUs."""
    if not is_cuda:
        return "Measured on CPU. GPU memory usage not recorded (CUDA unavailable)."

    # Allow 15% safety margin for OS, CUDA context overhead, and PyTorch runtime
    safety_margin = 1.15
    required_mb = peak_mem_mb * safety_margin

    fits_8gb = required_mb <= GPU_8GB_MB
    fits_12gb = required_mb <= GPU_12GB_MB

    rec_8 = "YES" if fits_8gb else "NO (exceeds budget with overhead)"
    rec_12 = "YES" if fits_12gb else "NO (exceeds budget with overhead)"

    return (
        f"GPU Headroom Recommendation:\n"
        f"  - Peak Allocated: {peak_mem_mb:.2f} MB (est. with overhead: {required_mb:.2f} MB)\n"
        f"  - Fits 8 GB GPU:  {rec_8}\n"
        f"  - Fits 12 GB GPU: {rec_12}"
    )


def run_profiler(
    config_path: str | Path,
    device_name: str | None = None,
    use_amp: bool = False,
    sweep: bool = False,
) -> None:
    """Run model profiling according to configuration specifications."""
    cfg = load_config(config_path)

    # Select device
    if device_name is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_name.lower()

    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device_str = "cpu"

    device = torch.device(device_str)

    model_cfg = cfg["model"]
    data_cfg = cfg.get("data", {})

    arch_name = model_cfg["name"]
    in_channels = int(model_cfg.get("in_channels", 4))
    out_channels = int(model_cfg.get("out_channels", 3))
    roi_size = list(data_cfg.get("roi_size", [32, 32, 32]))
    base_batch_size = int(data_cfg.get("batch_size", 1))

    # Build model and loss module
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)

    n_params = count_parameters(model, trainable_only=False)
    n_trainable = count_parameters(model, trainable_only=True)
    size_mb = model_size_mb(model)

    print("=" * 70)
    print("FedMed Model Profiler")
    print("=" * 70)
    print(f"Config:            {config_path}")
    print(f"Architecture:      {arch_name}")
    print(f"Device:            {device}")
    print(f"AMP:               {use_amp}")
    print(f"Parameters:        {n_params:,} total ({n_trainable:,} trainable)")
    print(f"Model Weights:     {size_mb:.2f} MB")
    print(f"ROI Size:          {roi_size}")
    print("-" * 70)

    batch_sizes = [1, 2, 4] if sweep else [base_batch_size]

    for bsz in batch_sizes:
        print(f"\nProfiling Batch Size: {bsz}")
        try:
            results = profile_single_run(
                model=model,
                loss_fn=loss_fn,
                batch_size=bsz,
                in_channels=in_channels,
                out_channels=out_channels,
                roi_size=roi_size,
                device=device,
                use_amp=use_amp,
            )
            print(f"  - Output Shape:          {results['out_shape']}")
            print(f"  - Forward+Backward Time: {results['elapsed_ms']:.2f} ms")
            if device.type == "cuda":
                print(f"  - Peak Memory:           {results['peak_mem_mb']:.2f} MB")
                print(f"  - {format_gpu_recommendation(results['peak_mem_mb'], is_cuda=True)}")
            else:
                print("  - Peak Memory:           N/A (CPU execution)")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"  - Batch size {bsz}: OUT OF MEMORY (OOM) caught cleanly.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise e

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile FedMed segmentation models.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Execution device.")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision.")
    parser.add_argument("--sweep", action="store_true", help="Sweep batch sizes 1, 2, 4.")

    args = parser.parse_args()
    run_profiler(
        config_path=args.config,
        device_name=args.device,
        use_amp=args.amp,
        sweep=args.sweep,
    )


if __name__ == "__main__":
    main()
