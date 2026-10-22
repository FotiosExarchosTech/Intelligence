"""Camera + MediaPipe face detection: tells the face where the person in front of it is.

Frames are processed in memory only; nothing is stored or sent anywhere.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time

# Quiet MediaPipe's native INFO logging; must be set before it is imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode  # noqa: E402

from .config import Settings
from .events import EventBus, Gaze

log = logging.getLogger(__name__)

_LOST_AFTER_S = 1.5  # keep looking where the person was for a moment before wandering again
_MIN_MOVE = 0.02  # don't flood the face page with tiny changes


class Vision:
    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self._settings = settings
        self._bus = bus
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_time = 0.0

    def snapshot_jpeg(self) -> bytes | None:
        """The current camera picture as JPEG, or None if the camera isn't working."""
        with self._frame_lock:
            frame, taken = self._latest_frame, self._latest_time
        if frame is None or time.monotonic() - taken > 2:
            return None
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes() if ok else None

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        try:
            await asyncio.to_thread(self._track, loop, stop)
        finally:
            stop.set()

    def _track(self, loop: asyncio.AbstractEventLoop, stop: threading.Event) -> None:
        s = self._settings
        # DirectShow opens webcams on Windows much faster than the default backend.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        camera = cv2.VideoCapture(s.camera_index, backend)
        if not camera.isOpened():
            log.warning("Camera %s not available - the eyes will just look around", s.camera_index)
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(s.face_detection_model)),
                running_mode=RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
        )
        log.info("Camera %s: tracking faces", s.camera_index)

        def publish(gaze: Gaze) -> None:
            loop.call_soon_threadsafe(self._bus.publish, gaze)

        interval = 1 / s.camera_fps
        last_seen = 0.0
        sent: tuple[float, float] | None = None
        try:
            while not stop.is_set():
                started = time.monotonic()
                ok, frame = camera.read()
                if not ok:
                    time.sleep(0.2)
                    continue
                with self._frame_lock:
                    self._latest_frame, self._latest_time = frame, started
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = detector.detect_for_video(image, int(started * 1000))

                if result.detections:
                    # The biggest face is the closest person: that's who we look at.
                    box = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
                    height, width = frame.shape[:2]
                    x, y = self._to_gaze((box.origin_x + box.width / 2) / width, (box.origin_y + box.height / 2) / height)
                    last_seen = started
                    if sent is None:
                        log.debug("Person in view")
                    if sent is None or abs(x - sent[0]) + abs(y - sent[1]) > _MIN_MOVE:
                        sent = (x, y)
                        publish(Gaze(x, y))
                elif sent is not None and started - last_seen > _LOST_AFTER_S:
                    log.debug("Nobody in view")
                    sent = None
                    publish(Gaze())

                time.sleep(max(0.0, interval - (time.monotonic() - started)))
        finally:
            detector.close()
            camera.release()

    def _to_gaze(self, cx: float, cy: float) -> tuple[float, float]:
        """Face center in the image (0..1) -> where the eyes look (-1..1)."""
        s = self._settings
        # A webcam image is unmirrored: a person stepping to their right shows up on the
        # image's left, and the eyes should follow them to the right.
        x = (2 * cx - 1) if s.camera_mirror else (1 - 2 * cx)
        y = 2 * cy - 1
        clamp = lambda v: max(-1.0, min(1.0, v * s.gaze_gain))  # noqa: E731
        return clamp(x), clamp(y)
