"""Speaker output that reports the loudness of what it plays (for the mouth animation)."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from .audio_devices import find_device
from .config import Settings

log = logging.getLogger(__name__)

_CHUNK_MS = 30  # granularity of writes and loudness updates


class Speaker:
    def __init__(self, settings: Settings, on_level: Callable[[float], None]) -> None:
        self._device = find_device(settings.audio_output_device, "output")
        self._on_level = on_level
        self._stream: sd.OutputStream | None = None
        self._stop = threading.Event()

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Play int16 mono samples; returns when done or when stop() is called."""
        self._stop.clear()
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._play_blocking, samples, sample_rate, loop)

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _play_blocking(self, samples: np.ndarray, sample_rate: int, loop: asyncio.AbstractEventLoop) -> None:
        # Keep one stream open between sentences (no clicks, no reopen latency); reopen
        # only when a voice with a different sample rate speaks.
        if self._stream is None or self._stream.samplerate != sample_rate:
            self.close()
            self._stream = sd.OutputStream(device=self._device, samplerate=sample_rate, channels=1, dtype="int16")
            self._stream.start()
            log.info("Speaker: %s @ %d Hz", sd.query_devices(self._stream.device)["name"], sample_rate)

        # A block becomes audible only after it passes the output buffer, so report its
        # loudness that much later; otherwise the mouth moves ahead of the voice.
        latency = self._stream.latency

        def report(level: float) -> None:
            loop.call_soon_threadsafe(loop.call_later, latency, self._on_level, level)

        chunk = sample_rate * _CHUNK_MS // 1000
        for start in range(0, len(samples), chunk):
            if self._stop.is_set():
                break
            block = samples[start : start + chunk]
            rms = float(np.sqrt(np.mean((block.astype(np.float32) / 32768) ** 2)))
            report(min(1.0, rms * 4))
            self._stream.write(block.reshape(-1, 1))
        report(0.0)
