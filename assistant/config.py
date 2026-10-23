"""Settings loaded from environment variables / the project's .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    return int(_str(name) or default)


def _float(name: str, default: float) -> float:
    return float(_str(name) or default)


def _bool(name: str, default: bool) -> bool:
    value = _str(name).lower()
    return default if not value else value in ("1", "true", "yes", "on")


def _path(name: str, default: str) -> Path:
    p = Path(_str(name) or default)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _device(name: str) -> int | str | None:
    """Audio device: empty = system default, a number = device index, text = name substring."""
    value = _str(name)
    if not value:
        return None
    return int(value) if value.isdigit() else value


def _think(value: str) -> bool | str | None:
    """Ollama 'think': low/medium/high for gpt-oss, true/false for other thinking models, empty = model default."""
    lowered = value.lower()
    if not lowered:
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    return lowered


@dataclass(frozen=True)
class Settings:
    # Conversation
    assistant_name: str
    knowledge_dir: Path
    language: str  # "auto", "el" or "en"
    conversation_timeout_s: float

    # LLM: "ollama" or "claude"
    llm_provider: str

    # Ollama
    ollama_host: str
    ollama_model: str
    ollama_think: bool | str | None

    # Claude
    llm_model: str
    llm_effort: str
    llm_max_tokens: int

    # Speech-to-text: "groq" (default), or local "canary", "parakeet" or "whisper"
    stt_engine: str
    stt_fallback: str
    groq_stt_model: str
    canary_model_dir: Path
    parakeet_model_dir: Path
    whisper_model: str
    whisper_language_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_beam_size: int
    whisper_download_dir: Path
    stt_hotwords: str

    # Text-to-speech (Piper)
    piper_voice_en: Path
    piper_voice_el: Path
    tts_length_scale: float

    # Face page + browser
    face_port: int
    face_fullscreen: bool
    face_captions: bool
    browser_profile_dir: Path
    browse_idle_return_s: float

    # Ridebly app + demos
    ridebly_app_url: str
    ridebly_app_test_server: bool
    ridebly_approval_wait_s: float
    demos_dir: Path

    # Camera (vision)
    camera_enabled: bool
    camera_index: int
    camera_mirror: bool
    camera_fps: float
    gaze_gain: float
    face_detection_model: Path

    # Audio devices
    audio_input_device: int | str | None
    audio_output_device: int | str | None

    # Voice activity detection (Silero)
    vad_threshold: float
    vad_min_silence_ms: int
    vad_min_speech_ms: int
    vad_max_utterance_s: float

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env")
        language = _str("LANGUAGE", "auto").lower()
        if language not in ("auto", "el", "en"):
            raise ValueError(f"LANGUAGE must be auto, el or en (got {language!r})")
        llm_provider = _str("LLM_PROVIDER", "ollama").lower()
        if llm_provider not in ("ollama", "claude"):
            raise ValueError(f"LLM_PROVIDER must be ollama or claude (got {llm_provider!r})")
        return cls(
            assistant_name=_str("ASSISTANT_NAME", "Anna"),
            knowledge_dir=_path("KNOWLEDGE_DIR", "knowledge"),
            language=language,
            conversation_timeout_s=_float("CONVERSATION_TIMEOUT_S", 300),
            llm_provider=llm_provider,
            ollama_host=_str("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=_str("OLLAMA_MODEL", "gemma4:31b-cloud"),
            ollama_think=_think(_str("OLLAMA_THINK", "true")),
            llm_model=_str("LLM_MODEL", "claude-opus-5-5"),
            llm_effort=_str("LLM_EFFORT", "low"),
            llm_max_tokens=_int("LLM_MAX_TOKENS", 16000),
            stt_engine=_str("STT_ENGINE", "groq").lower(),
            stt_fallback=_str("STT_FALLBACK", "canary").lower(),
            groq_stt_model=_str("GROQ_STT_MODEL", "whisper-large-v3"),
            canary_model_dir=_path("CANARY_MODEL_DIR", "models/canary-1b-v2"),
            parakeet_model_dir=_path("PARAKEET_MODEL_DIR", "models/parakeet-tdt-0.6b-v3"),
            whisper_model=_str("WHISPER_MODEL", "small"),
            whisper_language_model=_str("WHISPER_LANGUAGE_MODEL", "tiny"),
            whisper_device=_str("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", "int8"),
            whisper_beam_size=_int("WHISPER_BEAM_SIZE", 1),
            whisper_download_dir=_path("WHISPER_DOWNLOAD_DIR", "models/whisper"),
            stt_hotwords=_str("STT_HOTWORDS", "Ridebly, Flarmio, VisionArrays, Quickord, Roomismo, Rate Manager, Desmos"),
            piper_voice_en=_path("PIPER_VOICE_EN", "models/piper/en_US-lessac-medium.onnx"),
            piper_voice_el=_path("PIPER_VOICE_EL", "models/piper/el_GR-rapunzelina-medium.onnx"),
            tts_length_scale=_float("TTS_LENGTH_SCALE", 1.0),
            face_port=_int("FACE_PORT", 8765),
            face_fullscreen=_bool("FACE_FULLSCREEN", True),
            face_captions=_bool("FACE_CAPTIONS", True),
            browser_profile_dir=_path("BROWSER_PROFILE_DIR", "data/browser-profile"),
            browse_idle_return_s=_float("BROWSE_IDLE_RETURN_S", 90),
            ridebly_app_url=_str("RIDEBLY_APP_URL", "https://dev.app.ridebly.com/"),
            ridebly_app_test_server=_bool("RIDEBLY_APP_TEST_SERVER", True),
            ridebly_approval_wait_s=_float("RIDEBLY_APPROVAL_WAIT_S", 300),
            demos_dir=_path("DEMOS_DIR", "demos"),
            camera_enabled=_bool("CAMERA_ENABLED", True),
            camera_index=_int("CAMERA_INDEX", 0),
            camera_mirror=_bool("CAMERA_MIRROR", False),
            camera_fps=_float("CAMERA_FPS", 15),
            gaze_gain=_float("GAZE_GAIN", 1.3),
            face_detection_model=_path("FACE_DETECTION_MODEL", "models/mediapipe/blaze_face_short_range.tflite"),
            audio_input_device=_device("AUDIO_INPUT_DEVICE"),
            audio_output_device=_device("AUDIO_OUTPUT_DEVICE"),
            vad_threshold=_float("VAD_THRESHOLD", 0.5),
            vad_min_silence_ms=_int("VAD_MIN_SILENCE_MS", 1000),
            vad_min_speech_ms=_int("VAD_MIN_SPEECH_MS", 250),
            vad_max_utterance_s=_float("VAD_MAX_UTTERANCE_S", 30),
        )
