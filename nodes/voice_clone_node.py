"""OmniVoice Voice Clone TTS - Text-to-speech with voice cloning from reference audio.

This node clones a voice from a reference audio sample and synthesizes new speech
in that voice. Supports 600+ languages with high-quality zero-shot voice cloning.
"""

import logging
from typing import Tuple

import numpy as np
import torch

from .loader import (
    get_model_names,
    load_model,
    numpy_audio_to_comfy,
    comfy_audio_to_numpy,
    resolve_device,
)
from .omnivoice_tts import _smart_chunk_text
from .model_cache import (
    cancel_event,
    get_cache_key,
    get_cached_model,
    get_or_cache_whisper,
    is_offloaded,
    offload_model_to_cpu,
    offload_whisper_to_cpu,
    resume_model_to_cuda,
    set_cached_model,
    set_keep_loaded,
    unload_model,
    unload_whisper,
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


class OmniVoiceVoiceCloneTTS:
    """OmniVoice Voice Clone TTS node."""

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
                        "default": "Hello! This is a test of voice cloning with OmniVoice.",
                        "tooltip": (
                            "Text to synthesize in the cloned voice. "
                            "Supports inline non-verbal tags like [laughter], [sigh], etc."
                        ),
                    },
                ),
                "ref_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Reference audio to clone voice from. "
                            "3-15 seconds of clear speech works best. "
                            "Will be resampled to 24kHz if needed."
                        ),
                    },
                ),
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Transcript of the reference audio. "
                            "Leave empty to auto-transcribe with Whisper ASR. "
                            "Providing the transcript improves quality."
                        ),
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
                            "0 = automatic (uses speed). Overrides speed if set."
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
                    ["auto", "eager", "sage_attention"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Attention implementation. "
                            "'auto' uses model default (eager). "
                            "'sage_attention' uses SageAttention CUDA kernels (requires SM80+ GPU)."
                        ),
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
                "words_per_chunk": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 500,
                        "step": 10,
                        "tooltip": (
                            "Words per chunk for long text. 0 = no chunking. "
                            "Chunks split at sentence boundaries, not mid-word."
                        ),
                    },
                ),
            },
            "optional": {
                "whisper_model": (
                    "WHISPER_ASR",
                    {
                        "tooltip": (
                            "Optional pre-loaded Whisper ASR model. "
                            "Connect from OmniVoice Whisper Loader to avoid "
                            "re-downloading on each run."
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
        "OmniVoice Voice Clone TTS - Clone a voice from reference audio and "
        "synthesize new speech. High-quality zero-shot voice cloning for 600+ languages."
    )

    def generate(
        self,
        model: str,
        text: str,
        ref_audio: dict,
        ref_text: str,
        steps: int,
        speed: float,
        duration: float,
        device: str,
        dtype: str,
        attention: str,
        seed: int,
        words_per_chunk: int,
        keep_model_loaded: bool,
        whisper_model: dict = None,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        # Load or get cached model
        omnivoice_model, _ = self._get_model(
            model, device, dtype, attention, keep_model_loaded
        )

        pbar = ProgressBar(4) if _PBAR else None

        # Convert reference audio from ComfyUI format to numpy at 24kHz
        logger.info("Processing reference audio...")
        ref_audio_np, ref_sr = comfy_audio_to_numpy(ref_audio, target_sr=OMNIVOICE_SAMPLE_RATE)
        ref_audio_tensor = torch.from_numpy(ref_audio_np).float()
        ref_duration = len(ref_audio_np) / OMNIVOICE_SAMPLE_RATE

        # Warn about reference audio length
        if ref_duration < 1:
            logger.warning(
                f"Reference audio is only {ref_duration:.1f}s — "
                "recommend 3-15s for best quality."
            )
        elif ref_duration > 30:
            logger.warning(
                f"Reference audio is {ref_duration:.1f}s — "
                "longer than recommended 15s may cause issues."
            )

        if pbar:
            pbar.update_absolute(1, 4)

        # Log what we're generating
        logger.info(f"Voice Clone TTS: {text[:80]}{'...' if len(text) > 80 else ''}")
        if ref_text.strip():
            logger.info(f"Reference transcript provided — bypassing Whisper ASR")
        elif whisper_model is not None:
            whisper_pipe = get_or_cache_whisper(whisper_model, model, device, dtype)
            if whisper_pipe is not None:
                logger.info("No reference transcript — using pre-loaded Whisper ASR")
                omnivoice_model._asr_pipe = whisper_pipe
        else:
            logger.info("No reference transcript — Whisper will auto-transcribe (will download if not cached)")

        # Set random seed
        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)

        if pbar:
            pbar.update_absolute(2, 4)

        self._check_interrupt()

        # Smart chunk long text at sentence boundaries
        chunks = _smart_chunk_text(text, words_per_chunk)
        if len(chunks) > 1:
            logger.info(f"Long text detected — splitting into {len(chunks)} chunks at sentence boundaries")

        total_chunks = len(chunks)
        pbar = ProgressBar(total_chunks + 1) if _PBAR else None
        audio_chunks = []

        try:
            for chunk_idx, chunk_text in enumerate(chunks):
                self._check_interrupt()

                if len(chunks) > 1:
                    logger.info(f"  Chunk {chunk_idx + 1}/{len(chunks)}: {chunk_text[:50]}{'...' if len(chunk_text) > 50 else ''}")

                gen_kwargs = {
                    "text": chunk_text,
                    "num_step": steps,
                    "speed": speed,
                    "ref_audio": (ref_audio_tensor, OMNIVOICE_SAMPLE_RATE),
                }
                if ref_text.strip():
                    gen_kwargs["ref_text"] = ref_text.strip()
                if duration > 0:
                    gen_kwargs["duration"] = duration

                with torch.no_grad():
                    audio_list = omnivoice_model.generate(**gen_kwargs)

                audio_np = audio_list[0].squeeze(0).cpu().numpy()
                audio_chunks.append(audio_np)

                if pbar:
                    pbar.update_absolute(chunk_idx + 2, total_chunks + 1)

            # Concatenate all chunks
            if len(audio_chunks) == 1:
                audio_out = audio_chunks[0]
            else:
                audio_out = np.concatenate(audio_chunks, axis=0)

            result = numpy_audio_to_comfy(audio_out, OMNIVOICE_SAMPLE_RATE)

            logger.info(
                f"Generated {len(audio_out) / OMNIVOICE_SAMPLE_RATE:.2f}s of audio "
                f"at {OMNIVOICE_SAMPLE_RATE}Hz in cloned voice"
            )

        finally:
            if not keep_model_loaded:
                unload_model()
                unload_whisper()
            else:
                offload_model_to_cpu()
                offload_whisper_to_cpu()

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
