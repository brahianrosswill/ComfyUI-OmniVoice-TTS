"""Model loading and utilities for OmniVoice TTS.

Handles:
  - Model folder registration with ComfyUI
  - Auto-download from HuggingFace
  - Device and precision resolution
  - Model loading with OmniVoice.from_pretrained
  - Audio format conversion for ComfyUI
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("OmniVoice")

# Model folder name in ComfyUI/models/
MODELS_FOLDER_NAME = "omnivoice"


def _get_models_base() -> Path:
    """Get or create the models folder path."""
    try:
        import folder_paths
        base = Path(folder_paths.models_dir) / MODELS_FOLDER_NAME
    except ImportError:
        base = Path(__file__).resolve().parent.parent / "checkpoints" / MODELS_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def _register_folder() -> None:
    """Register models folder with ComfyUI's folder_paths."""
    try:
        import folder_paths
        base = str(_get_models_base())
        folder_paths.add_model_folder_path(MODELS_FOLDER_NAME, base)
        logger.info(f"Models folder registered: {base}")
    except ImportError:
        pass


# HuggingFace model configuration
# Keys are display names (shown in dropdown), repo_id is the actual HF repo
HF_MODELS = {
    "OmniVoice": {
        "repo_id": "k2-fsa/OmniVoice",
        "url": "https://huggingface.co/k2-fsa/OmniVoice",
        "description": "Full OmniVoice model - 600+ languages (fp32, ~4GB)",
    },
    "OmniVoice-bf16": {
        "repo_id": "drbaph/OmniVoice-bf16",
        "url": "https://huggingface.co/drbaph/OmniVoice-bf16",
        "description": "Bfloat16 quantized OmniVoice - smaller VRAM (~2GB)",
    },
}
HF_DEFAULT_MODEL = "k2-fsa/OmniVoice"
_AUTO_DOWNLOAD_SUFFIX = " (auto download)"


def _auto_download_model(model_name: str = HF_DEFAULT_MODEL) -> bool:
    """Download model from HuggingFace if not already present."""
    if model_name not in HF_MODELS:
        logger.error(f"Unknown model: {model_name}")
        return False

    cfg = HF_MODELS[model_name]
    repo_id = cfg["repo_id"]
    dest = _get_models_base() / model_name.replace("/", "_")

    if dest.is_dir() and any(dest.iterdir()):
        return True

    logger.info(f"Downloading '{model_name}' ({cfg['description']}) from HuggingFace...")
    logger.info(f"Repo: {repo_id}")
    logger.info(f"Destination: {dest}")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "*.h5"],
        )
        logger.info(f"Model downloaded to: {dest}")
        return True
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return False


def _is_model_downloaded(model_name: str) -> bool:
    """Check if a model is already downloaded."""
    base = _get_models_base()
    safe_name = model_name.replace("/", "_")
    model_path = base / safe_name
    if not model_path.is_dir():
        return False
    # Check for config.json or any weight files
    has_config = (model_path / "config.json").is_file()
    has_weights = any(
        f.suffix in {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf"}
        for f in model_path.iterdir()
        if f.is_file()
    )
    return has_config or has_weights


def get_model_names() -> list[str]:
    """Get list of available models (downloaded + auto-download options)."""
    base = _get_models_base()
    names = []

    # Add HF models with auto-download suffix if not present
    for model_name in HF_MODELS.keys():
        if _is_model_downloaded(model_name):
            names.append(model_name)
        else:
            names.append(f"{model_name}{_AUTO_DOWNLOAD_SUFFIX}")

    # Add any local models in the folder
    try:
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            safe_name = entry.name
            # Convert back to HF format if it matches
            hf_name = safe_name.replace("_", "/")
            if hf_name in HF_MODELS:
                continue
            # Check if it has model files
            has_config = (entry / "config.json").is_file()
            has_weights = any(
                f.suffix in {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf"}
                for f in entry.iterdir()
                if f.is_file()
            )
            if has_config or has_weights:
                names.append(safe_name)
    except OSError:
        pass

    return names


def _strip_auto_download_suffix(name: str) -> str:
    """Remove the auto-download suffix from a model name."""
    if name.endswith(_AUTO_DOWNLOAD_SUFFIX):
        return name[: -len(_AUTO_DOWNLOAD_SUFFIX)]
    return name


def _supports_bfloat16() -> bool:
    """Check if the GPU supports bfloat16."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def resolve_device(device_choice: str) -> Tuple[str, Optional[torch.dtype]]:
    """Resolve device choice to actual device string."""
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda", None
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", None
        logger.warning("No CUDA or MPS GPU detected — falling back to CPU.")
        return "cpu", None
    return device_choice, None


def resolve_precision(precision_choice: str, device: str) -> torch.dtype:
    """Resolve precision choice to torch dtype."""
    if precision_choice == "auto":
        if device == "cuda":
            return torch.bfloat16 if _supports_bfloat16() else torch.float16
        elif device == "mps":
            return torch.float16
        return torch.float32
    if precision_choice == "bf16":
        if device == "cuda" and not _supports_bfloat16():
            logger.warning(
                "bfloat16 requested but GPU does not support it (compute capability < 8.0). "
                "Consider using 'fp16' instead."
            )
        return torch.bfloat16
    if precision_choice == "fp16":
        return torch.float16
    return torch.float32


def numpy_audio_to_comfy(audio_np: np.ndarray, sample_rate: int) -> dict:
    """Convert numpy audio array to ComfyUI AUDIO format.

    Args:
        audio_np: Audio samples as numpy array (samples,) or (channels, samples)
        sample_rate: Sample rate in Hz

    Returns:
        dict with 'waveform' tensor (1, channels, samples) and 'sample_rate'
    """
    import torch

    # Ensure float32 for ComfyUI
    audio_np = audio_np.astype(np.float32)

    # Handle different input shapes
    if audio_np.ndim == 1:
        # (samples,) -> (1, 1, samples)
        audio_np = audio_np[np.newaxis, np.newaxis, :]
    elif audio_np.ndim == 2:
        # (channels, samples) -> (1, channels, samples)
        audio_np = audio_np[np.newaxis, :, :]
    else:
        # Already (batch, channels, samples) - squeeze to expected shape
        audio_np = audio_np[np.newaxis, :, :]

    waveform = torch.from_numpy(audio_np).contiguous()
    return {"waveform": waveform, "sample_rate": sample_rate}


def comfy_audio_to_numpy(audio_dict: dict, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Convert ComfyUI AUDIO format to numpy array.

    Args:
        audio_dict: ComfyUI audio dict with 'waveform' and 'sample_rate'
        target_sr: Optional target sample rate to resample to

    Returns:
        Tuple of (audio_np, sample_rate) where audio_np is (samples,)
    """
    waveform = audio_dict["waveform"]
    source_sr = audio_dict["sample_rate"]

    # ComfyUI AUDIO format: (batch, channels, samples)
    # Ensure we have a tensor, convert from numpy if needed
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(waveform[0]).float()
    else:
        # It's already a tensor
        wav = waveform[0].float()

    # Handle different channel configurations
    if wav.dim() == 1:
        # Already (samples,) - use as-is
        pass
    elif wav.shape[0] > 1:
        # Multi-channel -> mix down to mono: (samples,)
        wav = wav.mean(dim=0)
    elif wav.shape[0] == 1:
        # Mono -> squeeze to (samples,)
        wav = wav.squeeze(0)
    else:
        # Empty or weird shape - flatten to 1D
        wav = wav.flatten()

    # Convert to numpy, ensuring tensor is on CPU first
    audio_np = wav.cpu().numpy() if hasattr(wav, 'cpu') else np.array(wav)

    # Resample if needed
    if target_sr is not None and source_sr != target_sr:
        import librosa
        audio_np = librosa.resample(audio_np, orig_sr=source_sr, target_sr=target_sr)
        return audio_np, target_sr

    return audio_np, source_sr


def load_model(
    model_name: str,
    device: str,
    precision: str,
    attention: str,
):
    """Load OmniVoice model.

    Args:
        model_name: HuggingFace model name or local folder name
        device: Device choice ("auto", "cuda", "cpu", "mps")
        precision: Precision choice ("auto", "bf16", "fp16", "fp32")
        attention: Attention implementation ("auto", "sdpa", "sage_attention", "flash_attention")

    Returns:
        Tuple of (model, None) - no tokenizer needed for OmniVoice
    """
    from omnivoice import OmniVoice

    model_name = _strip_auto_download_suffix(model_name)
    device_str, _ = resolve_device(device)
    dtype = resolve_precision(precision, device_str)

    # Resolve the actual model identifier to pass to OmniVoice.from_pretrained
    # 1. If it's a known HF model (by display name), download to our folder then use local path
    # 2. If it's a local path, use the path
    # 3. Otherwise pass as-is (might be an unknown HF repo)
    model_identifier = model_name

    if model_name in HF_MODELS:
        # Known HF model - auto-download to our models folder first
        if not _is_model_downloaded(model_name):
            logger.info(f"Model '{model_name}' not found locally. Auto-downloading...")
            success = _auto_download_model(model_name)
            if not success:
                raise RuntimeError(f"Failed to download model '{model_name}'")

        # Use local path
        local_path = _get_models_base() / model_name
        model_identifier = str(local_path)
        logger.info(f"Using local model at: {local_path}")
    else:
        # Check if it's a local folder name
        local_path = _get_models_base() / model_name
        if local_path.is_dir():
            model_identifier = str(local_path)
            logger.info(f"Using local model at: {local_path}")

    logger.info(f"Loading OmniVoice: {model_identifier}")
    logger.info(f"Device: {device_str}, Precision: {dtype}")

    # Determine device_map for OmniVoice
    if device_str == "cuda":
        device_map = "cuda:0"
    elif device_str == "mps":
        device_map = "mps"
    else:
        device_map = "cpu"

    # Load model using OmniVoice's from_pretrained
    model = OmniVoice.from_pretrained(
        model_identifier,
        device_map=device_map,
        dtype=dtype,
    )

    model.eval()

    # Apply VBAR/aimdo detection
    from .model_cache import apply_vbar_detection
    apply_vbar_detection(model, device_str)

    logger.info("OmniVoice model loaded successfully.")
    return model, None  # No tokenizer needed for OmniVoice


def patch_attention(model, attention: str, device: str) -> None:
    """Patch attention implementation (placeholder for future support)."""
    if attention == "auto":
        return

    # TODO: Implement attention patching if OmniVoice supports it
    if attention in ("sage_attention", "flash_attention"):
        logger.warning(
            f"{attention} is requested but OmniVoice uses its own attention. "
            f"The attention setting may not have an effect."
        )
