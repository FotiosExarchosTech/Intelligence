"""Scripted, narrated demos: YAML files in demos/ listing steps to show and lines to say.

A demo file looks like this (see demos/_template.yaml for every kind of step):

    title: Ridebly website tour
    description: What Ridebly does, how a booking flows and what it costs.
    steps:
      - open: https://www.ridebly.com/how-it-works
        say:
          en: Here's how Ridebly works, step by step.
          el: Δείτε πώς λειτουργεί το Ridebly, βήμα βήμα.
      - point_at: STEP 1
        say: {en: ..., el: ...}

Each step does one thing on screen while the assistant says its line, and the next step starts once
both are done. Files whose name starts with "_" are ignored.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .browser import Browser

log = logging.getLogger(__name__)

_ACTIONS = ("open", "open_app", "click", "point_at", "scroll", "type", "press", "pricing", "wait", "show_face")


@dataclass(frozen=True)
class Demo:
    name: str
    title: str
    description: str
    steps: list[dict]


def load_demos(folder: Path) -> dict[str, Demo]:
    demos = {}
    for path in sorted(folder.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            steps = data["steps"]
            for number, step in enumerate(steps, 1):
                actions = [a for a in _ACTIONS if a in step]
                if len(actions) > 1:
                    raise ValueError(f"step {number} has more than one action: {actions}")
            demos[path.stem] = Demo(path.stem, data["title"], data.get("description", ""), steps)
        except Exception as exc:
            log.warning("Skipping demo %s: %s", path.name, exc)
    if demos:
        log.info("Demos: %s", ", ".join(demos))
    return demos


class NotLoggedIn(Exception):
    pass


class DemoRunner:
    def __init__(self, browser: Browser, app_url: str, say: Callable[[str, str], Awaitable[None]]) -> None:
        self._browser = browser
        self._app_url = app_url.rstrip("/")
        self._say = say

    async def run(self, demo: Demo, language: str) -> str:
        """Runs the demo; returns a short report for the LLM."""
        log.info("Demo %s (%s)", demo.name, language)
        for number, step in enumerate(demo.steps, 1):
            line = _line(step.get("say"), language)
            action = asyncio.create_task(self._do(step, language))
            try:
                if line:
                    await self._say(line, language)
                await action
            except NotLoggedIn:
                return (f"The demo stopped at step {number}: the Ridebly app's login page is on screen. A live "
                        "demo of the app needs Fotis Exarchos's approval, given by logging in here. Tell the "
                        "person in a sentence, call wait_for_approval, then start the demo again.")
            except Exception as exc:
                log.warning("Demo %s failed at step %d: %s", demo.name, number, exc)
                return (f"The demo stopped at step {number} of {len(demo.steps)} because something on the page "
                        f"didn't work ({str(exc).splitlines()[0]}). Apologise briefly and carry on from the "
                        "page on screen.")
            finally:
                if not action.done():
                    action.cancel()
            await asyncio.sleep(0.3)
        return "The demo is finished; the last page is still on screen. Ask if they have any questions."

    async def _do(self, step: dict, language: str) -> None:
        b = self._browser
        if "open" in step:
            await b.open(_localized(step["open"], language))
        elif "open_app" in step:
            await self.open_app(step["open_app"])
        elif "click" in step:
            await b.click(_line(step["click"], language))
        elif "point_at" in step:
            await b.point_at(_line(step["point_at"], language))
        elif "scroll" in step:
            await b.scroll(step["scroll"])
        elif "type" in step:
            spec = step["type"]
            await b.type_text(_line(spec.get("field", ""), language), _line(spec["text"], language), spec.get("submit", False))
        elif "press" in step:
            await b.press_key(step["press"])
        elif "pricing" in step:
            spec = step["pricing"]
            await b.ridebly_pricing(spec["business"], spec.get("vehicles"), spec.get("users"), language)
        elif "wait" in step:
            await asyncio.sleep(float(step["wait"]))
        elif "show_face" in step:
            await b.show_face()

    async def open_app(self, path: str = "") -> dict:
        page = await self._browser.open(f"{self._app_url}/{path.lstrip('/')}")
        if self._is_login(page["url"]):
            raise NotLoggedIn
        return page

    async def wait_for_login(self, timeout_s: float) -> dict:
        """Wait until someone logs in to the app on screen (that's how approval is given)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(1)
            url = self._browser.current_url()
            if url and url.startswith(self._app_url) and not self._is_login(url):
                await asyncio.sleep(2)  # let the app finish loading after logging in
                return await self._browser.describe()
        raise NotLoggedIn

    @staticmethod
    def _is_login(url: str) -> bool:
        return "/login" in url


def _line(value, language: str) -> str:
    """A step's text: a plain string, or {en: ..., el: ...} picked by language."""
    if isinstance(value, dict):
        return value.get(language) or value.get("en") or next(iter(value.values()), "")
    return value or ""


def _localized(url: str, language: str) -> str:
    """ridebly.com remembers the last language in a cookie, so always ask for one explicitly:
    /en/ for English, /el/ for Greek."""
    base = "https://www.ridebly.com/"
    if not url.startswith(base) or url.startswith((base + "el/", base + "en/")):
        return url
    return base + language + "/" + url[len(base):]
