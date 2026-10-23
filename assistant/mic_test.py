"""`assistant --mic-test`: record a few seconds and show what the assistant hears and understands.

Nothing is saved: the recording is analysed, played back and discarded.
"""

from __future__ import annotations

import asyncio
import sys

import numpy as np
import sounddevice as sd

from .audio_devices import find_device
from .audio_in import Microphone, SpeechDiscarded, Utterance
from .config import Settings
from .vad import FRAME_SAMPLES, SAMPLE_RATE

_SECONDS = 6


def _record(device: int | None) -> np.ndarray:
    """Record _SECONDS at 16 kHz (resampled if the microphone can't do 16 kHz), with a level bar."""
    try:
        sd.check_input_settings(device, channels=1, samplerate=SAMPLE_RATE)
        rate = SAMPLE_RATE
    except sd.PortAudioError:
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
    blocks: list[np.ndarray] = []

    def meter(indata, frames, time, status) -> None:
        blocks.append(indata[:, 0].copy())
        level = float(np.abs(indata).max())
        bar = "#" * min(40, int(level * 200))
        sys.stdout.write(f"\r  level |{bar:<40}| {level:.3f} ")
        sys.stdout.flush()

    with sd.InputStream(device=device, channels=1, samplerate=rate, dtype="float32", callback=meter):
        sd.sleep(_SECONDS * 1000)
    print()
    audio = np.concatenate(blocks)
    if rate != SAMPLE_RATE:
        n = round(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


async def _detect(settings: Settings, audio: np.ndarray) -> list:
    """Run the recording through the assistant's own speech detector."""
    mic = Microphone(settings)
    padded = np.concatenate([audio, np.zeros(SAMPLE_RATE + (-len(audio) % FRAME_SAMPLES), np.float32)])
    for i in range(0, len(padded), FRAME_SAMPLES):
        mic._frames.put_nowait(padded[i : i + FRAME_SAMPLES])
    events: list = []

    async def collect() -> None:
        async for event in mic.events():
            events.append(event)

    try:
        await asyncio.wait_for(collect(), timeout=5)
    except TimeoutError:
        pass
    return events


def run() -> None:
    settings = Settings.load()
    device = find_device(settings.audio_input_device, "input")
    name = sd.query_devices(device if device is not None else sd.default.device[0], "input")["name"]
    print(f"Microphone: {name}  (AUDIO_INPUT_DEVICE={settings.audio_input_device or 'system default'})")
    print(f"Speak normally for {_SECONDS} seconds, e.g. a sentence in Greek and one in English...")
    audio = _record(device)

    peak, rms = float(np.abs(audio).max()), float(np.sqrt(np.mean(audio**2)))
    print(f"\nLoudness: peak {peak:.3f}, average {rms:.4f}")
    if peak < 0.02:
        print("  -> VERY QUIET. This microphone barely hears you: try another one (AUDIO_INPUT_DEVICE),")
        print("     raise its level in Windows sound settings, or check the mute key.")
    elif peak > 0.98:
        print("  -> TOO LOUD (clipping): lower the microphone level in Windows sound settings.")
    else:
        print("  -> OK")

    events = asyncio.run(_detect(settings, audio))
    utterances = [e for e in events if isinstance(e, Utterance)]
    discarded = sum(isinstance(e, SpeechDiscarded) for e in events)
    print(f"\nSpeech detector: {len(utterances)} sentence(s) found"
          + (f", {discarded} short sound(s) ignored" if discarded else ""))
    if not utterances:
        print("  -> No speech detected: the assistant would just go back to idle.")
    elif len(utterances) > 1:
        print(f"  -> If you said one sentence, it was cut at pauses longer than {settings.vad_min_silence_ms} ms;")
        print("     raise VAD_MIN_SILENCE_MS in .env to let people pause longer.")

    if utterances:
        from .stt import create_speech_to_text

        print("\nLoading speech recognition...")
        stt = create_speech_to_text(settings)
        for utterance in utterances:
            text, language = stt._transcribe(utterance.audio)
            print(f"  ({language}, {utterance.seconds:.1f}s) {text or '(nothing understood)'}")

    print("\nPlaying back what the microphone recorded...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
