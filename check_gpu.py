"""
Run this BEFORE train.py to confirm your GPU is actually usable by PyTorch.
Catches the most common failure: torch installed but CPU-only build.

Usage:
    python check_gpu.py
"""

import sys


def main():
    print("=" * 60)
    print("GPU READINESS CHECK")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed at all.")
        print("Run: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)

    print(f"torch version: {torch.__version__}")

    cuda_ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_ok}")

    if not cuda_ok:
        print()
        print("FAIL: torch cannot see a GPU. This almost always means you have")
        print("the CPU-only build of torch installed, even if your GPU itself")
        print("is fine (check with `nvidia-smi` in a separate terminal).")
        print()
        print("Fix:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("Then run this script again.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"GPU: {device_name}")
    print(f"Compute capability: {capability}")
    print(f"Total VRAM: {total_mem_gb:.1f} GB")

    # Actually run a small op on the GPU, not just check availability —
    # is_available() can be True while a real op still fails on some broken setups.
    print()
    print("Running a real tensor op on the GPU to confirm end-to-end...")
    try:
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"OK: matmul result shape {tuple(c.shape)}, sum={c.sum().item():.2f}")
    except Exception as e:
        print(f"FAIL: GPU op raised an error: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — safe to run train.py, it will use the GPU.")
    print("=" * 60)


if __name__ == "__main__":
    main()
