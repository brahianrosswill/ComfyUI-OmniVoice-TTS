"""Model caching and VRAM management for OmniVoice TTS.

Supports:
  - Model caching with configurable keep_loaded
  - CPU offload to free VRAM between runs
  - VBAR/aimdo detection for ComfyUI dynamic VRAM management
  - Interrupt handling for cancelled generations
"""

import gc
import logging
import threading
from typing import Any

import torch

logger = logging.getLogger("OmniVoice")

# Global cache state protected by lock
_cache_lock = threading.Lock()
_cached_model: Any = None
_cached_key: tuple = ()
_keep_loaded: bool = False
_offloaded: bool = False

# Whisper ASR cache — kept separate from the OmniVoice model since
# it's loaded by a different node and has its own lifecycle.
_whisper_lock = threading.Lock()
_cached_whisper: Any = None
_cached_whisper_key: tuple = ()

# Cancellation event for interrupt handling
cancel_event: threading.Event = threading.Event()


def get_cache_key(model_path: str, device: str, dtype: str, attention: str) -> tuple:
    """Generate a cache key from model configuration."""
    return (model_path, device, dtype, attention)


def get_cached_model() -> tuple[Any, tuple]:
    """Get the cached model and its key."""
    with _cache_lock:
        return _cached_model, _cached_key


def set_cached_model(model: Any, key: tuple, keep_loaded: bool = False) -> None:
    """Cache a model with its configuration key."""
    global _cached_model, _cached_key, _keep_loaded, _offloaded
    with _cache_lock:
        _cached_model = model
        _cached_key = key
        _keep_loaded = keep_loaded
        _offloaded = False


def set_keep_loaded(keep_loaded: bool) -> None:
    """Update the keep_loaded flag for the cached model."""
    global _keep_loaded
    with _cache_lock:
        _keep_loaded = keep_loaded


def is_offloaded() -> bool:
    """Check if the model is currently offloaded to CPU."""
    with _cache_lock:
        return _offloaded


def _detect_vbar() -> tuple[bool, bool]:
    """Detect if ComfyUI's dynamic VRAM management (VBAR/aimdo) is available.

    Returns:
        tuple: (vbar_available, aimdo_available)
            - vbar_available: True if ModelVBAR class can be imported (explicit VBAR mode)
            - aimdo_available: True if comfy_aimdo package is installed (auto memory management)
    """
    try:
        import comfy_aimdo
        from comfy_aimdo.model_vbar import ModelVBAR
        return True, True
    except ImportError:
        pass
    try:
        import comfy_aimdo
        return False, True
    except ImportError:
        pass
    return False, False


def offload_model_to_cpu() -> None:
    """Offload the cached model to CPU to free VRAM."""
    global _offloaded
    with _cache_lock:
        if _cached_model is None:
            return
        if _offloaded:
            return

        # Skip manual offload if VBAR/aimdo is handling VRAM
        if getattr(_cached_model, "_vbar_active", False) or getattr(
            _cached_model, "_aimdo_auto", False
        ):
            _offloaded = True
            mode = "VBAR" if getattr(_cached_model, "_vbar_active", False) else "aimdo auto"
            logger.info(f"{mode} active — skipping manual CPU offload")
            return

        try:
            _cached_model.to("cpu")
            _offloaded = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Model offloaded to CPU. VRAM freed.")
        except Exception as e:
            logger.warning(f"Failed to offload model: {e}")


def resume_model_to_cuda(device: str = "cuda") -> None:
    """Resume an offloaded model back to GPU.

    Also restores the device of any internal ASR pipeline (``_asr_pipe``)
    attached to the model so that input tensors and weights stay on the
    same device.
    """
    global _offloaded
    with _cache_lock:
        if _cached_model is None:
            return
        if not _offloaded:
            return
        try:
            _cached_model.to(device)
            _offloaded = False
            # Restore internal ASR pipeline to the same device
            asr = getattr(_cached_model, "_asr_pipe", None)
            if asr is not None:
                _whisper_to_device(asr, device)
            logger.info(f"Model resumed to {device}.")
        except Exception as e:
            logger.warning(f"Failed to resume model: {e}")


def unload_model() -> None:
    """Fully unload the model from memory."""
    global _cached_model, _cached_key, _keep_loaded, _offloaded
    with _cache_lock:
        if _cached_model is not None:
            logger.info("Unloading OmniVoice model from memory...")
            try:
                _cached_model.to("cpu")
            except Exception:
                pass
            del _cached_model
            _cached_model = None
            _cached_key = ()
            _keep_loaded = False
            _offloaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            logger.info("Model unloaded and VRAM freed.")


def _whisper_to_device(pipe: Any, device: str) -> None:
    """Move a HuggingFace pipeline to the target device.

    Strategy: move the underlying model directly (most reliable across
    transformers versions), then fall back to pipeline-level .to().
    """
    for attr in ("model", "_model"):
        m = getattr(pipe, attr, None)
        if m is not None and hasattr(m, "to"):
            try:
                m.to(device)
                return
            except Exception:
                pass
    # Fallback: pipeline-level .to()
    try:
        pipe.to(device)
    except Exception:
        pass


def _whisper_to_cpu(pipe: Any) -> None:
    """Move a HuggingFace pipeline's model to CPU."""
    _whisper_to_device(pipe, "cpu")


def unload_whisper() -> None:
    """Fully unload the cached Whisper model from memory."""
    global _cached_whisper, _cached_whisper_key
    with _whisper_lock:
        if _cached_whisper is not None:
            logger.info("Unloading Whisper ASR model from memory...")
            _whisper_to_cpu(_cached_whisper)
            del _cached_whisper
            _cached_whisper = None
            _cached_whisper_key = ()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Whisper ASR model unloaded and VRAM freed.")


def offload_whisper_to_cpu() -> None:
    """Offload the cached Whisper model to CPU to free VRAM."""
    with _whisper_lock:
        if _cached_whisper is None:
            return
        try:
            _whisper_to_cpu(_cached_whisper)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Whisper ASR model offloaded to CPU. VRAM freed.")
        except Exception as e:
            logger.warning(f"Failed to offload Whisper model: {e}")


def get_or_cache_whisper(whisper_input: dict | None, model_name: str, device: str, dtype: str) -> Any:
    """Get or load a Whisper pipeline with caching.

    Args:
        whisper_input: Whisper dict from node input (None if not connected).
            Must have "pipeline" and "model_name" keys.
        model_name: Model name from dropdown (for cache key when no input).
        device: Device to load on.
        dtype: Model precision.

    Returns:
        Whisper pipeline or None.
    """
    if whisper_input is None:
        return None

    key = (whisper_input.get("model_name", model_name), device, dtype)

    # Resolve target device so we can restore Whisper to it after CPU offload
    from .loader import resolve_device
    device_str, _ = resolve_device(device)

    global _cached_whisper, _cached_whisper_key
    with _whisper_lock:
        if _cached_whisper is not None and _cached_whisper_key == key:
            # Ensure Whisper is on the target device (may have been offloaded to CPU)
            _whisper_to_device(_cached_whisper, device_str)
            logger.info(f"Reusing cached Whisper ASR model (on {device_str}).")
            return _cached_whisper

        # Settings changed — unload old
        if _cached_whisper is not None:
            logger.info("Whisper settings changed — unloading old Whisper model.")
            _whisper_to_cpu(_cached_whisper)
            del _cached_whisper
            _cached_whisper = None
            _cached_whisper_key = ()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Use the pipeline from the input (already loaded by Whisper Loader node).
        # IMPORTANT: The same pipeline object is shared with ComfyUI's node
        # output cache, so it may have been moved to CPU by a previous
        # unload_whisper() call.  Restore it to the target device now.
        pipe = whisper_input["pipeline"]
        _whisper_to_device(pipe, device_str)
        _cached_whisper = pipe
        _cached_whisper_key = key
        logger.info(f"Whisper ASR model cached (on {device_str}).")
        return _cached_whisper


def apply_vbar_detection(model: Any, device_str: str) -> None:
    """Apply VBAR/aimdo detection flags to the model."""
    model._vbar_active = False
    model._aimdo_auto = False
    if device_str == "cuda":
        vbar_avail, aimdo_avail = _detect_vbar()
        if vbar_avail:
            model._vbar_active = True
            logger.info("ComfyUI Dynamic VRAM (VBAR explicit) detected")
        elif aimdo_avail:
            model._aimdo_auto = True
            logger.info("ComfyUI Dynamic VRAM (aimdo auto-allocator) detected")


def _hook_comfy_model_management() -> None:
    """Hook into ComfyUI's model management for automatic VRAM management."""
    try:
        import comfy.model_management as mm
        _original = mm.soft_empty_cache

        def _patched_soft_empty_cache(*args, **kwargs):
            # Read shared state inside the lock to avoid data races with
            # ComfyUI's scheduler thread.
            with _cache_lock:
                should_offload = _keep_loaded and _cached_model is not None
            if should_offload:
                offload_model_to_cpu()
            else:
                unload_model()
            return _original(*args, **kwargs)

        mm.soft_empty_cache = _patched_soft_empty_cache
        logger.debug("Hooked comfy.model_management.soft_empty_cache for OmniVoice unload.")
    except Exception:
        pass


# Initialize the hook on module load
_hook_comfy_model_management()
