"""Serves the face page and pushes assistant events to it over a websocket."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..events import (
    AssistantSaid,
    AudioLevel,
    EventBus,
    Gaze,
    InterruptRequested,
    LanguageChanged,
    LanguageRequested,
    State,
    StateChanged,
    UserSaid,
)

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class _Server(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        # Leave Ctrl+C to our own asyncio.run(); uvicorn would otherwise swallow it.
        yield


def _message(event: object) -> dict | None:
    """The JSON message the face page understands for an event, if any."""
    match event:
        case StateChanged(state=state):
            return {"type": "state", "state": state.value}
        case AudioLevel(level=level):
            return {"type": "level", "level": round(level, 3)}
        case Gaze(x=None) | Gaze(y=None):
            return {"type": "gaze", "x": None, "y": None}
        case Gaze(x=x, y=y):
            return {"type": "gaze", "x": round(x, 3), "y": round(y, 3)}
        case UserSaid(text=text):
            return {"type": "say", "who": "user", "text": text}
        case AssistantSaid(text=text):
            return {"type": "say", "who": "assistant", "text": text}
        case LanguageChanged(language=language):
            return {"type": "language", "language": language}
    return None


class FaceServer:
    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self._bus = bus
        self._clients: set[WebSocket] = set()
        self._state = State.LOADING
        self._language = settings.language
        self._captions = settings.face_captions

        app = FastAPI()

        @app.middleware("http")
        async def no_cache(request, call_next):
            # The browser profile persists between runs: never let it show an old face page.
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response

        app.add_api_websocket_route("/ws", self._websocket)
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True))
        self._server = _Server(
            uvicorn.Config(app, host="127.0.0.1", port=settings.face_port, log_level="warning", lifespan="off")
        )
        self.url = f"http://127.0.0.1:{settings.face_port}/"

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self._server.serve())
            tasks.create_task(self._broadcast())

    async def wait_started(self) -> None:
        while not self._server.started:
            await asyncio.sleep(0.05)
        log.info("Face page: %s", self.url)

    async def _websocket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "config", "captions": self._captions})
        await websocket.send_json({"type": "state", "state": self._state.value})
        await websocket.send_json({"type": "language", "language": self._language})
        self._clients.add(websocket)
        try:
            while True:
                try:
                    message = json.loads(await websocket.receive_text())
                except json.JSONDecodeError:
                    continue
                if message.get("type") == "interrupt":
                    self._bus.publish(InterruptRequested())
                elif message.get("type") == "language" and message.get("language") in ("el", "en"):
                    self._bus.publish(LanguageRequested(message["language"]))
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)

    async def _broadcast(self) -> None:
        events = self._bus.subscribe()
        while True:
            event = await events.get()
            if isinstance(event, StateChanged):
                self._state = event.state
            elif isinstance(event, LanguageChanged):
                self._language = event.language
            if (message := _message(event)) is None or not self._clients:
                continue
            text = json.dumps(message, ensure_ascii=False)
            clients = list(self._clients)
            results = await asyncio.gather(*(c.send_text(text) for c in clients), return_exceptions=True)
            for client, result in zip(clients, results):
                if isinstance(result, Exception):
                    self._clients.discard(client)
