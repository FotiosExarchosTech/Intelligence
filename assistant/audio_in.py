"""Microphone capture + voice activity detection that cuts the stream into utterances."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .audio_devices import find_device
from .config import Settings
from .vad import FRAME_SAMPLES, SAMPLE_RATE, SileroVAD

log = logging.getLogger(__name__)

_FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32 ms
_PRE_ROLL_MS = 300  # audio kept from just before speech started, so first syllables aren't cut
_TAIL_KEEP_MS = 200  # silence kept after the last word
_SILENCE_CHECK_FRAMES = 3000 // _FRAME_MS  # the first 3 s after starting
_SILENT_PEAK = 0.004


@dataclass(frozen=True)
class SpeechStarted:
    pass


@dataclass(frozen=True)
class SpeechDiscarded:
    """Speech started but was too short to be an utterance (a cough, a click...)."""


@dataclass(frozen=True)
class Utterance:
    audio: np.ndarray  # float32 mono, 16 kHz

    @property
    def seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE


VoiceEvent = SpeechStarted | SpeechDiscarded | Utterance


class Microphone:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._vad = SileroVAD()
        self._frames: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._device_rate = SAMPLE_RATE
        self._resampled = np.zeros(0, dtype=np.float32)
        # While paused (the assistant is talking) incoming audio is dropped, so the
        # assistant doesn't hear and answer itself through the laptop speakers.
        self.paused = False

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        device = find_device(self._settings.audio_input_device, "input")
        try:
            sd.check_input_settings(device, channels=1, samplerate=SAMPLE_RATE, dtype="float32")
            self._device_rate = SAMPLE_RATE
        except sd.PortAudioError:
            self._device_rate = int(sd.query_devices(device, "input")["default_samplerate"])
            log.info("Microphone doesn't support 16 kHz, capturing at %d Hz and resampling", self._device_rate)

        def callback(indata: np.ndarray, frames: int, time, status) -> None:  # audio thread
            loop.call_soon_threadsafe(self._frames.put_nowait, indata[:, 0].copy())

        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=self._device_rate,
            dtype="float32",
            blocksize=round(FRAME_SAMPLES * self._device_rate / SAMPLE_RATE),
            callback=callback,
        )
        self._stream.start()
        log.info("Microphone: %s", sd.query_devices(self._stream.device)["name"])

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def _next_frame(self) -> np.ndarray:
        """Next 512-sample frame at 16 kHz, resampling if the device runs at another rate."""
        while len(self._resampled) < FRAME_SAMPLES:
            block = await self._frames.get()
            if self._device_rate != SAMPLE_RATE:
                n_out = round(len(block) * SAMPLE_RATE / self._device_rate)
                x_old = np.linspace(0, 1, len(block), endpoint=False)
                x_new = np.linspace(0, 1, n_out, endpoint=False)
                block = np.interp(x_new, x_old, block).astype(np.float32)
            self._resampled = np.concatenate([self._resampled, block])
        frame, self._resampled = self._resampled[:FRAME_SAMPLES], self._resampled[FRAME_SAMPLES:]
        return frame

    async def events(self) -> AsyncIterator[VoiceEvent]:
        s = self._settings
        threshold = s.vad_threshold
        silence_threshold = max(threshold - 0.15, 0.05)  # hysteresis: harder to stop than to start
        min_silence_frames = s.vad_min_silence_ms // _FRAME_MS
        min_speech_frames = s.vad_min_speech_ms // _FRAME_MS
        max_frames = int(s.vad_max_utterance_s * 1000 / _FRAME_MS)

        pre_roll: deque[np.ndarray] = deque(maxlen=_PRE_ROLL_MS // _FRAME_MS)
        speech: list[np.ndarray] = []
        in_speech = False
        speech_frames = silence_frames = 0
        startup_frames, startup_peak = 0, 0.0

        while True:
            frame = await self._next_frame()

            # A working microphone always picks up some room noise; a silent one (e.g. an
            # empty headset jack) makes the assistant mishear everything.
            if startup_frames < _SILENCE_CHECK_FRAMES:
                startup_frames += 1
                startup_peak = max(startup_peak, float(np.abs(frame).max()))
                if startup_frames == _SILENCE_CHECK_FRAMES and startup_peak < _SILENT_PEAK:
                    log.warning(
                        "The microphone is almost silent (peak %.4f). If the assistant doesn't understand you, "
                        "set AUDIO_INPUT_DEVICE in .env (run with --list-devices to see the microphones)",
                        startup_peak,
                    )

            if self.paused:
                if in_speech or pre_roll:
                    self._vad.reset()
                    pre_roll.clear()
                    speech, in_speech = [], False
                continue

            prob = self._vad(frame)

            if not in_speech:
                pre_roll.append(frame)
                if prob >= threshold:
                    in_speech = True
                    speech = list(pre_roll)
                    speech_frames, silence_frames = 1, 0
                    yield SpeechStarted()
                continue

            speech.append(frame)
            if prob >= threshold:
                speech_frames += 1
                silence_frames = 0
            elif prob < silence_threshold:
                silence_frames += 1

            if silence_frames >= min_silence_frames or len(speech) >= max_frames:
                in_speech = False
                pre_roll.clear()
                if speech_frames >= min_speech_frames:
                    # Most of the trailing silence only told us the turn was over; recognizers
                    # are faster and a little more accurate without it.
                    trim = max(0, silence_frames - _TAIL_KEEP_MS // _FRAME_MS)
                    yield Utterance(np.concatenate(speech[: len(speech) - trim]))
                else:
                    yield SpeechDiscarded()
                speech = []
