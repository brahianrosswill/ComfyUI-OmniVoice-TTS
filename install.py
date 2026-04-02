"""Installation script for OmniVoice TTS custom node.

This runs BEFORE __init__.py when the node is first installed via ComfyUI-Manager.

Strategy:
1. Install omnivoice with --no-deps (won't touch existing packages)
2. Install only the NEW packages that omnivoice needs but aren't already in ComfyUI
3. Never touch torch/numpy/transformers/etc - ComfyUI already has them
"""

import subprocess
import sys


def pip_install(spec):
    """Install a package. Returns True on success."""
    embedded = "python_embeded" in sys.executable
    base = [sys.executable] + (["-s"] if embedded else [])

    # Try pip first, then uv
    cmd = base + ["-m", "pip", "install"] + spec.split()

    # Check if uv is available as fallback
    try:
        subprocess.check_output(
            base + ["-m", "uv", "--version"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        cmd = base + ["-m", "uv", "pip", "install"] + spec.split()
    except Exception:
        pass

    print(f"[OmniVoice Install] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print(f"[OmniVoice Install] Successfully installed: {spec}")
        return True
    print(f"[OmniVoice Install] Failed to install {spec}:\n{result.stderr}")
    return False


def check_package(import_name):
    """Check if a package is already installed."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def main():
    print("[OmniVoice Install] Starting installation...")

    # 1. Install omnivoice with --no-deps so it can't override anything
    if not check_package("omnivoice"):
        print("[OmniVoice Install] Installing omnivoice (with --no-deps to protect existing packages)...")
        if not pip_install("omnivoice --no-deps"):
            print("[OmniVoice Install] ERROR: Failed to install omnivoice")
            return
    else:
        print("[OmniVoice Install] omnivoice already installed")

    # 2. Install only the packages that omnivoice needs that ComfyUI might NOT have
    # ComfyUI already has: torch, numpy, transformers, einops, safetensors, huggingface_hub, accelerate
    # omnivoice may need these additional packages:
    extra_deps = [
        ("soundfile", "soundfile"),        # Audio file I/O
        ("librosa", "librosa>=0.10.1"),    # Audio processing
        ("alpha_clip", "alpha-clip"),      # May be needed by omnivoice
        ("sentencepiece", "sentencepiece"), # Tokenization
        ("jieba", "jieba"),                 # Chinese text segmentation
    ]

    for import_name, pip_spec in extra_deps:
        if not check_package(import_name):
            print(f"[OmniVoice Install] Installing {pip_spec}...")
            pip_install(pip_spec)
        else:
            print(f"[OmniVoice Install] {import_name} already installed")

    # 3. Verify torch is still CUDA build (sanity check)
    try:
        import torch
        version = torch.__version__
        if "+cu" in version:
            print(f"[OmniVoice Install] torch {version} (CUDA) - OK")
        else:
            print(f"[OmniVoice Install] WARNING: torch {version} is not CUDA build!")
            print("[OmniVoice Install] ComfyUI may not work correctly.")
    except ImportError:
        print("[OmniVoice Install] WARNING: torch not found!")

    print("[OmniVoice Install] Installation complete!")


if __name__ == "__main__":
    main()
