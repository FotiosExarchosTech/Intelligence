"""Text-to-speech with Piper: one voice per language, chosen per sentence."""

from __future__ import annotations

import asyncio
import logging
import re

import numpy as np
from piper import PiperVoice, SynthesisConfig

from .config import Settings
from .language import text_language
from .pronounce import Pronouncer

log = logging.getLogger(__name__)

_GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
_LATIN = re.compile(r"[A-Za-z]")


def _voice_for(text: str, language: str) -> str:
    """The voice for a sentence: the conversation's language whenever the sentence has some words
    in it (Greek sentences often contain English terms, and vice versa), else the other one."""
    if (language == "el" and _GREEK.search(text)) or (language == "en" and _LATIN.search(text)):
        return language
    return text_language(text, language)


class TextToSpeech:
    def __init__(self, settings: Settings) -> None:
        self._voices: dict[str, PiperVoice] = {}
        for language, path in (("en", settings.piper_voice_en), ("el", settings.piper_voice_el)):
            if not path.exists():
                raise FileNotFoundError(f"Piper voice not found: {path} (see README: downloading voices)")
            self._voices[language] = PiperVoice.load(path)
            log.info("Loaded Piper voice %s for %r", path.name, language)
        self._config = SynthesisConfig(length_scale=settings.tts_length_scale)
        english = self._voices["en"]
        self._pronounce = Pronouncer(lambda word: " ".join("".join(p) for p in english.phonemize(word)))

    async def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """Returns (int16 mono samples, sample rate)."""
        return await asyncio.to_thread(self._synthesize, text, language)

    def _synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        voice_language = _voice_for(text, language)
        voice = self._voices[voice_language]
        chunks = list(voice.synthesize(self._pronounce(text, voice_language), syn_config=self._config))
        if not chunks:
            return np.zeros(0, dtype=np.int16), voice.config.sample_rate
        return np.concatenate([c.audio_int16_array for c in chunks]), chunks[0].sample_rate
