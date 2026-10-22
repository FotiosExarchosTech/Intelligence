"""Turning the AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE settings into a sound device."""

from __future__ import annotations

import sounddevice as sd


def find_device(spec: int | str | None, kind: str) -> int | None:
    """None = system default, a number = device index, text = part of the device name.

    On Windows the same microphone is listed once per audio system (MME, DirectSound,
    WASAPI, WDM-KS), so a name usually matches several entries: take the one on the
    default audio system.
    """
    if spec is None or isinstance(spec, int):
        return spec
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [d for d in sd.query_devices() if spec.lower() in d["name"].lower() and d[channels] > 0]
    if not matches:
        raise ValueError(f"No {kind} device matches {spec!r}. Run `assistant --list-devices` to see them.")
    default_api = sd.default.hostapi
    matches.sort(key=lambda d: d["hostapi"] != default_api)
    return matches[0]["index"]
