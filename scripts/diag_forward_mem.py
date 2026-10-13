#!/usr/bin/env python3
"""Diagnostic: compare torch-reported GPU memory vs nvidia-smi for one forward.

Usage: diag_forward_mem.py MODEL.pt BATCH
"""
import subprocess
import sys

import torch


def smi():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().replace("\n", " | ")
    except Exception as exc:  # noqa: BLE001
        return f"(nvidia-smi failed: {exc})"


def main() -> None:
    model_path = sys.argv[1]
    batch = int(sys.argv[2])
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        print("CUDA_UNAVAILABLE")
        return
    print(f"=== {model_path} BATCH {batch} torch {torch.__version__}", flush=True)
    print("NVIDIA_BEFORE", smi(), flush=True)
    model = torch.jit.load(model_path, map_location=device).eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch, 5, 512, device=device)
    with torch.no_grad():
        try:
            model(x)
        except Exception:  # noqa: BLE001
            model([x])
    torch.cuda.synchronize()
    gib = 1024 ** 3
    print("TORCH_ALLOCATED_GIB", round(torch.cuda.memory_allocated(device) / gib, 3), flush=True)
    print("TORCH_RESERVED_GIB", round(torch.cuda.memory_reserved(device) / gib, 3), flush=True)
    print("TORCH_MAX_ALLOCATED_GIB", round(torch.cuda.max_memory_allocated(device) / gib, 3), flush=True)
    print("TORCH_MAX_RESERVED_GIB", round(torch.cuda.max_memory_reserved(device) / gib, 3), flush=True)
    print("NVIDIA_AFTER", smi(), flush=True)


if __name__ == "__main__":
    main()
