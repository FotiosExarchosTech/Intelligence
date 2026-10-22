"""Speech-to-text: Whisper large-v3 on Groq (default), or locally NVIDIA Canary, Parakeet
or faster-whisper.

Locally, NVIDIA Canary must be told the language, and told the wrong one it *translates*
(Greek speech told "English" comes out as an English translation). See CanarySpeechToText
for how the language is chosen.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import Settings
from .language import SUPPORTED_LANGUAGES, text_language

log = logging.getLogger(__name__)

# Very short replies ("ναι", "yes", "ok") can't be told apart reliably: they always keep the
# conversation's language. (Utterances include ~0.5 s of silence around the words.)
_SHORT_S = 1.4


def _load_whisper(settings: Settings, name: str):
    from faster_whisper import WhisperModel

    log.info("Loading Whisper model %r (first run downloads it)...", name)
    return WhisperModel(
        name,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        download_root=str(settings.whisper_download_dir),
    )


class LanguageDetector:
    """Greek or English? Uses a small Whisper model (~0.3 s)."""

    def __init__(self, settings: Settings, model=None) -> None:
        self._model = model or _load_whisper(settings, settings.whisper_language_model)
        self._previous: str | None = None

    def probabilities(self, audio: np.ndarray) -> dict[str, float]:
        _, _, all_probs = self._model.detect_language(audio)
        return {code: p for code, p in all_probs if code in SUPPORTED_LANGUAGES}

    def detect(self, audio: np.ndarray) -> str:
        previous = self._previous
        if previous and len(audio) < _SHORT_S * 16000:
            log.info("Language: %s (short reply, kept)", previous)
            return previous
        # Every sentence is judged on its own: a sticky wrong guess would make the recognizer
        # translate the person's Greek into English for the rest of the conversation.
        probs = self.probabilities(audio)
        best = max(SUPPORTED_LANGUAGES, key=lambda code: probs.get(code, 0.0))
        log.info("Language: %s (el %.2f, en %.2f)", best, probs.get("el", 0), probs.get("en", 0))
        self._previous = best
        return best


class SpeechToText:
    async def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """16 kHz float32 audio -> (text, language code). Runs in a worker thread."""
        return await asyncio.to_thread(self._transcribe, audio)

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        raise NotImplementedError


# The NVIDIA models' vocabulary has no final sigma (and a few rare accented letters): they write <unk>.
_UNK_FINAL_SIGMA = re.compile(r"(?<=[Ͱ-Ͽἀ-῿])<unk>(?!\w)")
_UNK = re.compile(r"<unk>")


def _fix_unknown_letters(text: str) -> str:
    return _UNK.sub("", _UNK_FINAL_SIGMA.sub("ς", text)).strip()


# Below this difference in average confidence, Canary's two readings count as a tie and the
# small Whisper model's language guess decides.
_CONFIDENCE_TIE = 0.05


class CanarySpeechToText(SpeechToText):
    """NVIDIA Canary 1B v2, which must be told the language. Told the wrong one, it translates
    (English speech told "Greek" comes out as Greek), and a small language guesser is easily
    fooled by a Greek accent. So it transcribes both ways at once and keeps the reading it is
    clearly more confident in (translating costs it confidence); only near-ties go to the
    guesser. ~1.2 s on a laptop CPU."""

    def __init__(self, settings: Settings) -> None:
        import onnx_asr

        self._settings = settings
        path = settings.canary_model_dir
        log.info("Loading Canary speech recognition%s...", "" if path.exists() else " (first run downloads ~1 GB)")
        self._model = onnx_asr.load_model("nemo-canary-1b-v2", path, quantization="int8").with_timestamps()
        self._detector = LanguageDetector(settings) if settings.language == "auto" else None
        self._pool = ThreadPoolExecutor(max_workers=3)
        self._previous: str | None = None

    def _read(self, audio: np.ndarray, language: str) -> tuple[str, float]:
        """Transcription in `language` and the model's average confidence (log-probability)."""
        result = self._model.recognize(audio, sample_rate=16000, language=language)
        confidence = float(np.mean(result.logprobs)) if result.logprobs else -10.0
        return _fix_unknown_letters(result.text), confidence

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        if self._detector is None:
            return self._read(audio, self._settings.language)[0], self._settings.language
        if self._previous and len(audio) < _SHORT_S * 16000:
            # One or two words: too little to judge, keep the conversation's language.
            log.info("Language: %s (short reply, kept)", self._previous)
            return self._read(audio, self._previous)[0], self._previous

        readings = {lang: self._pool.submit(self._read, audio, lang) for lang in SUPPORTED_LANGUAGES}
        guess = self._pool.submit(self._detector.probabilities, audio)
        readings = {lang: future.result() for lang, future in readings.items()}
        probs = guess.result()
        el, en = readings["el"][1], readings["en"][1]
        if abs(el - en) >= _CONFIDENCE_TIE:
            language = "el" if el > en else "en"
        else:
            language = "el" if probs.get("el", 0) >= probs.get("en", 0) else "en"
        log.info(
            "Language: %s (confidence el %.2f / en %.2f, guess el %.2f / en %.2f)",
            language, el, en, probs.get("el", 0), probs.get("en", 0),
        )
        self._previous = language
        return readings[language][0], language


class ParakeetSpeechToText(SpeechToText):
    """NVIDIA Parakeet v3: the fastest (~0.4 s), but it guesses the language on its own and may
    turn Greek into English. Best with LANGUAGE=en."""

    def __init__(self, settings: Settings) -> None:
        import onnx_asr

        self._settings = settings
        path = settings.parakeet_model_dir
        log.info("Loading Parakeet speech recognition%s...", "" if path.exists() else " (first run downloads ~670 MB)")
        self._model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path, quantization="int8")

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        text = _fix_unknown_letters(self._model.recognize(audio, sample_rate=16000))
        language = self._settings.language
        if language == "auto":
            language = text_language(text, fallback="en")
        return text, language


class WhisperSpeechToText(SpeechToText):
    """faster-whisper: accurate but slow on CPU (~3 s with "small")."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = _load_whisper(settings, settings.whisper_model)
        self._detector = None
        if settings.language == "auto":
            same = settings.whisper_language_model == settings.whisper_model
            self._detector = LanguageDetector(settings, self._model if same else None)

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        language = self._detector.detect(audio) if self._detector else self._settings.language
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            beam_size=self._settings.whisper_beam_size,
            hotwords=self._settings.stt_hotwords or None,
            condition_on_previous_text=False,
            vad_filter=False,  # the microphone's VAD already cut the utterance
        )
        # Whisper sometimes "hears" words in noise; drop segments it is unsure contain speech.
        text = " ".join(
            segment.text.strip()
            for segment in segments
            if not (segment.no_speech_prob > 0.6 and segment.avg_logprob < -1.0)
        ).strip()
        return text, language


# Whisper's well-known "subtitles" hallucinations on noise or silence.
_HALLUCINATIONS = {
    "υπότιτλοι authorwave",
    "ευχαριστούμε που παρακολουθήσατε",
    "ευχαριστώ που παρακολουθήσατε",
    "thank you for watching",
    "thanks for watching",
    "thank you",
}
_LANGUAGE_CODES = {"greek": "el", "english": "en", "el": "el", "en": "en"}


class GroqSpeechToText(SpeechToText):
    """Whisper large-v3 on Groq's servers: much better Greek than any model that runs well on
    a laptop CPU, in ~0.5 s. Falls back to a local engine when offline or over the free limit."""

    def __init__(self, settings: Settings, fallback: SpeechToText | None) -> None:
        import groq

        self._settings = settings
        self._client = groq.Groq(timeout=6.0, max_retries=0)  # reads GROQ_API_KEY
        self._fallback = fallback
        self._previous: str | None = None
        log.info("Speech recognition: Groq %s%s", settings.groq_stt_model, " (local fallback ready)" if fallback else "")

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        try:
            return self._groq(audio)
        except Exception as exc:
            if self._fallback is None:
                raise
            log.warning("Groq speech recognition failed (%s); using the local recognizer", exc)
            return self._fallback._transcribe(audio)

    def _groq(self, audio: np.ndarray) -> tuple[str, str]:
        forced = self._settings.language if self._settings.language != "auto" else None
        if forced is None and self._previous and len(audio) < _SHORT_S * 16000:
            forced = self._previous  # too short to judge the language: keep the conversation's
        text, language = self._request(audio, forced)
        if language not in SUPPORTED_LANGUAGES:
            # Whisper heard some other language; ask again in the conversation's language.
            text, language = self._request(audio, self._previous or "el")
        self._previous = language
        if text.lower().strip(" .!") in _HALLUCINATIONS:
            return "", language
        return text, language

    def _request(self, audio: np.ndarray, language: str | None) -> tuple[str, str]:
        result = self._client.audio.transcriptions.create(
            model=self._settings.groq_stt_model,
            file=("speech.wav", _wav_bytes(audio)),
            response_format="verbose_json",
            temperature=0.0,
            **({"language": language} if language else {}),
        )
        data = result.model_dump()
        detected = _LANGUAGE_CODES.get(str(data.get("language") or language).lower(), str(data.get("language")))
        # Whisper sometimes "hears" words in noise; drop segments it is unsure contain speech.
        segments = data.get("segments") or []
        if segments:
            text = " ".join(
                s["text"].strip()
                for s in segments
                if not (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)
            )
        else:
            text = data.get("text") or ""
        log.debug("Groq heard (%s): %s", detected, text)
        return text.strip(), detected


def _wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


_LOCAL_ENGINES = {"canary": CanarySpeechToText, "parakeet": ParakeetSpeechToText, "whisper": WhisperSpeechToText}


def create_speech_to_text(settings: Settings) -> SpeechToText:
    engine = settings.stt_engine
    if engine == "groq":
        fallback_name = settings.stt_fallback
        if fallback_name not in (*_LOCAL_ENGINES, "none"):
            raise ValueError(f"STT_FALLBACK must be one of {', '.join(_LOCAL_ENGINES)} or none")
        fallback = _LOCAL_ENGINES[fallback_name](settings) if fallback_name != "none" else None
        if not os.getenv("GROQ_API_KEY"):
            if fallback is None:
                raise ValueError("STT_ENGINE=groq needs GROQ_API_KEY in .env")
            log.warning("GROQ_API_KEY is not set (add it to .env) - using the local %s recognizer", fallback_name)
            return fallback
        return GroqSpeechToText(settings, fallback)
    if engine not in _LOCAL_ENGINES:
        raise ValueError(f"STT_ENGINE must be groq or one of {', '.join(_LOCAL_ENGINES)} (got {engine!r})")
    return _LOCAL_ENGINES[engine](settings)
