"""A tiny async publish/subscribe event bus plus the events and states it carries.

Modules never call each other directly for status updates: they publish events here,
and anyone interested (console logger now, the face UI in step 2) subscribes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class State(str, Enum):
    LOADING = "loading"  # starting up; the face keeps its eyes closed
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    DEMO = "demo"


@dataclass(frozen=True)
class StateChanged:
    state: State


@dataclass(frozen=True)
class UserSaid:
    text: str
    language: str


@dataclass(frozen=True)
class AssistantSaid:
    text: str


@dataclass(frozen=True)
class AudioLevel:
    """Loudness (0..1) of the speech currently playing; drives the mouth animation."""

    level: float


@dataclass(frozen=True)
class LanguageRequested:
    """Someone used the language switch (or asked by voice): "el", "en" or "auto"."""

    language: str


@dataclass(frozen=True)
class LanguageChanged:
    """The language the assistant now listens and speaks in: "el", "en" or "auto"."""

    language: str


@dataclass(frozen=True)
class InterruptRequested:
    """Someone pressed the stop button: stop thinking and talking right away."""


@dataclass(frozen=True)
class Gaze:
    """Where the eyes should look, -1..1 on each axis (x: left..right, y: up..down).

    None means nobody is in view: the eyes wander on their own.
    """

    x: float | None = None
    y: float | None = None


class EventBus:
    """Fan-out bus: every subscriber gets its own queue with every published event."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Any]] = []

    def subscribe(self, maxsize: int = 256) -> asyncio.Queue[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Any]) -> None:
        self._subscribers.remove(queue)

    def publish(self, event: Any) -> None:
        for queue in self._subscribers:
            if queue.full():
                # A slow subscriber must never block the publisher: drop its oldest event.
                queue.get_nowait()
            queue.put_nowait(event)
