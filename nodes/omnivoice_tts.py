"""OmniVoice Longform TTS - Text-to-speech with smart chunking and optional voice cloning.

This node handles long text by automatically chunking at sentence boundaries.
Supports optional voice cloning from reference audio.
"""

import logging
import re
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

OMNIVOICE_SAMPLE_RATE = 24000


def _smart_chunk_text(text: str, words_per_chunk: int = 100) -> list:
    """Split text into chunks at sentence boundaries, not cutting words.

    Tries to split at sentence-ending punctuation (.!?) followed by space/newline.
    Falls back to word boundaries if no sentence breaks found.
    """
    if words_per_chunk <= 0:
        return [text]

    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]

    chunks = []
    current_chunk = []
    current_word_count = 0

    sentence_end = re.compile(r'[.!?]+(?:\s|$)')

    for word in words:
        current_chunk.append(word)
        current_word_count += 1

        if current_word_count >= words_per_chunk:
            chunk_text = ' '.join(current_chunk)
            matches = list(sentence_end.finditer(chunk_text))

            if matches:
                last_match = matches[-1]
                split_pos = last_match.end()
                final_chunk = chunk_text[:split_pos].strip()
                remaining = chunk_text[split_pos:].strip()

                if final_chunk:
                    chunks.append(final_chunk)

                current_chunk = remaining.split() if remaining else []
                current_word_count = len(current_chunk)
            else:
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                current_chunk = []
                current_word_count = 0

    if current_chunk:
        remaining = ' '.join(current_chunk).strip()
        if remaining:
            chunks.append(remaining)

    return chunks


class OmniVoiceLongformTTS:
    """OmniVoice Longform TTS with smart chunking and optional voice cloning."""

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
                        "default": "Hello! This is a test of OmniVoice text to speech synthesis.",
                        "tooltip": (
                            "Text to synthesize. Supports inline non-verbal tags like "
                            "[laughter], [sigh], [sniff], [question-en], etc. "
                            "Long text will be automatically chunked at sentence boundaries."
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
                            "Leave empty to auto-transcribe with Whisper. "
                            "Only used if 'ref_audio' is connected."
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
                        "tooltip": (
                            "Speaking speed factor. "
                            ">1.0 = faster, <1.0 = slower."
                        ),
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
                    ["auto", "sdpa", "sage_attention", "flash_attention"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Attention implementation. "
                            "'auto' uses model default (SDPA)."
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
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep model loaded between runs. "
                            "Model is automatically offloaded to CPU after generation to free VRAM, "
                            "then resumed to GPU on the next run."
                        ),
                    },
                ),
            },
            "optional": {
                "ref_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional reference audio for voice cloning. "
                            "3-15 seconds of clear speech works best. "
                            "If not connected, uses automatic voice selection."
                        ),
                    },
                ),
                "whisper_model": (
                    "WHISPER_ASR",
                    {
                        "tooltip": (
                            "Optional pre-loaded Whisper ASR model for auto-transcription. "
                            "Connect from OmniVoice Whisper Loader to avoid re-downloading."
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
        "OmniVoice Longform TTS - Zero-shot multilingual TTS with smart chunking. "
        "Handles long text by splitting at sentence boundaries. "
        "Optional voice cloning from reference audio."
    )

    def generate(
        self,
        model: str,
        text: str,
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
        ref_audio: dict = None,
        whisper_model: dict = None,
    ) -> Tuple[dict]:
        cancel_event.clear()
        self._check_interrupt()

        if not text.strip():
            raise ValueError("Text cannot be empty.")

        omnivoice_model, _ = self._get_model(
            model, device, dtype, attention, keep_model_loaded
        )

        use_voice_clone = ref_audio is not None

        ref_audio_tensor = None
        if use_voice_clone:
            logger.info("Processing reference audio for voice cloning...")
            ref_audio_np, _ = comfy_audio_to_numpy(ref_audio, target_sr=OMNIVOICE_SAMPLE_RATE)
            ref_audio_tensor = torch.from_numpy(ref_audio_np).float()

            ref_duration = len(ref_audio_np) / OMNIVOICE_SAMPLE_RATE
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

            if not ref_text.strip() and whisper_model is not None:
                logger.info("Using pre-loaded Whisper ASR for voice transcription")
                omnivoice_model._asr_pipe = whisper_model["pipeline"]
            elif not ref_text.strip():
                logger.info("No ref_text — Whisper will auto-transcribe (downloads if not cached)")

        chunks = _smart_chunk_text(text, words_per_chunk)

        if len(chunks) > 1:
            logger.info(f"Long text detected — splitting into {len(chunks)} chunks at sentence boundaries")

        total_chunks = len(chunks)
        pbar = ProgressBar(total_chunks + 1) if _PBAR else None

        preview = text[:80] + "..." if len(text) > 80 else text
        mode = "voice clone" if use_voice_clone else "auto voice"
        logger.info(f"Longform TTS ({mode}): {preview}")

        if pbar:
            pbar.update_absolute(1, total_chunks + 1)

        actual_seed = seed if seed != 0 else torch.randint(0, 2**31, (1,)).item()
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(actual_seed)

        self._check_interrupt()

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
                }

                if use_voice_clone:
                    gen_kwargs["ref_audio"] = (ref_audio_tensor, OMNIVOICE_SAMPLE_RATE)
                    if ref_text.strip():
                        gen_kwargs["ref_text"] = ref_text.strip()

                if duration > 0:
                    gen_kwargs["duration"] = duration

                with torch.no_grad():
                    audio_list = omnivoice_model.generate(**gen_kwargs)

                audio_tensor = audio_list[0]
                audio_np = audio_tensor.squeeze(0).cpu().numpy()
                audio_chunks.append(audio_np)

                if pbar:
                    pbar.update_absolute(chunk_idx + 2, total_chunks + 1)

            if len(audio_chunks) == 1:
                audio_out = audio_chunks[0]
            else:
                audio_out = np.concatenate(audio_chunks, axis=0)

            result = numpy_audio_to_comfy(audio_out, OMNIVOICE_SAMPLE_RATE)

            logger.info(
                f"Generated {len(audio_out) / OMNIVOICE_SAMPLE_RATE:.2f}s of audio "
                f"at {OMNIVOICE_SAMPLE_RATE}Hz"
            )

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

        if cached_model is not None and cached_key != key:
            logger.info(
                f"Settings changed (model/device/dtype/attention) — "
                f"unloading cached model. Old: {cached_key}, New: {key}"
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
