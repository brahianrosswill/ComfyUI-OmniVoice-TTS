"""OmniVoice Voice Design TTS - Text-to-speech with voice design from text description.

This node creates synthetic voices from text descriptions (voice attributes)
and synthesizes speech in that designed voice. No reference audio needed.
"""

import logging
from typing import Tuple

import numpy as np
import torch

from .loader import (
    get_model_names,
    load_model,
    numpy_audio_to_comfy,
    resolve_device,
)
from .model_cache import (
    cancel_event,
    get_cache_key,
    get_cached_model,
    is_offloaded,
    offload_model_to_cpu,
    resume_model_to_cuda,
    set_cached_model,
    set_keep_loaded,
    unload_model,
)

try:
    from comfy.utils import ProgressBar
    _PBAR = True
except ImportError:
    _PBAR = False

try:
    import comfy.model_management as mm
    _MM = True
except ImportError:
    _MM = False

logger = logging.getLogger("OmniVoice")

# OmniVoice outputs at 24kHz
OMNIVOICE_SAMPLE_RATE = 24000

# Voice design attribute hints
VOICE_DESIGN_HINT = (
    "Voice attributes (comma-separated): "
    "gender (male/female), age (child/young/elderly), "
    "pitch (very low/low/medium/high/very high), "
    "style (whisper), "
    "accent (american/british/australian/sichuan/shaanxi/etc.). "
    "Example: 'female, low pitch, british accent'"
)


class OmniVoiceVoiceDesignTTS:
    """OmniVoice Voice Design TTS node."""

    @classmethod
    def INPUT_TYPES(cls):
        model_names = get_model_names()
        return {
            "required": {
                "model": (
                    model_names,
                    {
                        "tooltip": (
                            "OmniVoice model checkpoint. "
                            "Models are stored in ComfyUI/models/omnivoice/"
                        ),
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello! This is a test of voice design with OmniVoice.",
                        "tooltip": (
                            "Text to synthesize in the designed voice. "
                            "Supports inline non-verbal tags like [laughter], [sigh], etc."
                        ),
                    },
                ),
                "voice_instruct": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "female, low pitch, british accent",
                        "tooltip": VOICE_DESIGN_HINT,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 32,
                        "min": 4,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Number of diffusion steps. "
                            "16 = faster, 32 = balanced, 64 = best quality."
                        ),
                    },
                ),
                "speed": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Speaking speed factor. >1.0 = faster, <1.0 = slower.",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": (
                            "Fixed output duration in seconds. "
                            "0 = automatic. Overrides speed if set."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                        "tooltip": "Compute device. 'auto' picks CUDA > MPS > CPU.",
                    },
                ),
                "dtype": (
                    ["auto", "bf16", "fp16", "fp32"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Model precision. 'auto' picks bf16 for CUDA (Ampere+), "
                            "fp16 for older CUDA/MPS, fp32 for CPU."
                        ),
                    },
                ),
                "attention": (
                    ["auto", "sdpa", "sage_attention", "flash_attention"],
                    {
                        "default": "auto",
                        "tooltip": "Attention implementation. 'auto' uses model default.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed. 0 = random.",
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep model loaded between runs. "
                            "Model is automatically offloaded to CPU after generation."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "OmniVoice"
    DESCRIPTION = (
        "OmniVoice Voice Design TTS - Design synthetic voices from text descriptions. "
        "Control gender, age, pitch, accent, and more. No reference audio needed."
    )

    def generate(
        self,
        model: str,
        text: str,
        voice_instruct: str,
        steps: int,
        speed: float,
        duration: float,
        device: str,
        dtype: str,
        attention: str,
        seed: int,
        keep_model_loaded: bool,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        if not voice_instruct.strip():
            logger.warning(
                "No voice instruction provided. A random voice will be used. "
                "Consider adding attributes like 'female, low pitch, british accent'."
            )

        # Load or get cached model
        omnivoice_model, _ = self._get_model(
            model, device, dtype, attention, keep_model_loaded
        )

        pbar = ProgressBar(3) if _PBAR else None

        # Log what we're generating
        logger.info(f"Voice Design TTS: {text[:80]}{'...' if len(text) > 80 else ''}")
        logger.info(f"Voice attributes: {voice_instruct}")

        if pbar:
            pbar.update_absolute(1, 3)

        # Set random seed
        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)

        self._check_interrupt()

        try:
            # Build kwargs for generate
            gen_kwargs = {
                "text": text,
                "instruct": voice_instruct,
                "num_step": steps,
                "speed": speed,
            }
            if duration > 0:
                gen_kwargs["duration"] = duration

            # Generate audio with voice design
            with torch.no_grad():
                audio_list = omnivoice_model.generate(**gen_kwargs)

            if pbar:
                pbar.update_absolute(2, 3)

            # Convert to ComfyUI format
            audio_tensor = audio_list[0]  # (1, T)
            audio_np = audio_tensor.squeeze(0).cpu().numpy()

            result = numpy_audio_to_comfy(audio_np, OMNIVOICE_SAMPLE_RATE)

            logger.info(
                f"Generated {len(audio_np) / OMNIVOICE_SAMPLE_RATE:.2f}s of audio "
                f"at {OMNIVOICE_SAMPLE_RATE}Hz with designed voice"
            )

            if pbar:
                pbar.update_absolute(3, 3)

        finally:
            if not keep_model_loaded:
                unload_model()
            else:
                offload_model_to_cpu()

        return (result,)

    def _get_model(
        self,
        model_name: str,
        device: str,
        dtype: str,
        attention: str,
        keep_loaded: bool = False,
    ):
        """Get or load the OmniVoice model with caching."""
        key = get_cache_key(model_name, device, dtype, attention)
        cached_model, cached_key = get_cached_model()

        # Check if settings changed - force full unload if so
        if cached_model is not None and cached_key != key:
            logger.info(
                f"Settings changed (model/device/dtype/attention) — "
                f"unloading cached model."
            )
            unload_model()

        if cached_model is not None and cached_key == key:
            set_keep_loaded(keep_loaded)
            if is_offloaded():
                device_str, _ = resolve_device(device)
                logger.info(f"Resuming offloaded model to {device_str}...")
                resume_model_to_cuda(device_str)
            else:
                logger.info("Reusing cached OmniVoice model.")
            return cached_model, None

        # Load fresh model
        omnivoice_model, _ = load_model(model_name, device, dtype, attention)
        set_cached_model(omnivoice_model, key, keep_loaded=keep_loaded)
        return omnivoice_model, None

    def _check_interrupt(self):
        """Check if processing was interrupted."""
        if _MM:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                cancel_event.set()
                raise


# Voice attribute reference for documentation
VOICE_ATTRIBUTES = {
    "gender": ["male", "female"],
    "age": ["child", "young", "middle-aged", "elderly"],
    "pitch": ["very low", "low", "medium", "high", "very high"],
    "style": ["whisper"],
    "english_accents": [
        "american accent", "british accent", "australian accent",
        "canadian accent", "indian accent", "irish accent",
        "scottish accent", "south african accent",
    ],
    "chinese_dialects": [
        "四川话", "陕西话", "广东话", "东北话", "山东话",
        "河南话", "上海话", "闽南话", "客家话",
    ],
}

NON_VERBAL_TAGS = [
    "[laughter]",
    "[confirmation-en]",
    "[question-en]", "[question-ah]", "[question-oh]",
    "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]",
    "[dissatisfaction-hnn]",
    "[sniff]",
    "[sigh]",
]
