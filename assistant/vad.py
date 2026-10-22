"""Streaming Silero VAD running directly on onnxruntime (no PyTorch needed).

Uses the Silero v6 ONNX model bundled with faster-whisper. The model takes 512-sample
frames at 16 kHz, plus the last 64 samples of the previous frame as context, and keeps
an LSTM state (h, c) between frames.
"""

from __future__ import annotations

import os

import faster_whisper
import numpy as np
import onnxruntime

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms
_CONTEXT_SAMPLES = 64

MODEL_PATH = os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad_v6.onnx")


class SileroVAD:
    def __init__(self, model_path: str = MODEL_PATH) -> None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability (0..1) of one 512-sample float32 frame at 16 kHz."""
        x = np.concatenate([self._context, frame])[np.newaxis, :]
        prob, self._h, self._c = self._session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._context = frame[-_CONTEXT_SAMPLES:]
        return float(np.ravel(prob)[0])
