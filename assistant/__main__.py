"""Entry point: `uv run assistant` (or `python -m assistant`)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import PROJECT_ROOT, Settings
from .events import AssistantSaid, EventBus, StateChanged, UserSaid


async def print_conversation(bus: EventBus) -> None:
    """Console view of what is happening (the face UI replaces this in step 2)."""
    events = bus.subscribe()
    while True:
        match await events.get():
            case StateChanged(state=state):
                logging.getLogger("state").debug("-> %s", state.value)
            case UserSaid(text=text, language=language):
                print(f"\nYou ({language}): {text}", flush=True)
            case AssistantSaid(text=text):
                print(f"Assistant: {text}", flush=True)


def _quiet_connection_resets(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # Windows reports a browser tab closing its websocket as an error; it's harmless.
    if not isinstance(context.get("exception"), ConnectionResetError):
        loop.default_exception_handler(context)


def _helper(coroutine) -> asyncio.Task:
    """A background task whose crash is logged instead of silently lost."""

    def report_crash(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger("assistant").error("Background task crashed", exc_info=task.exception())

    task = asyncio.create_task(coroutine)
    task.add_done_callback(report_crash)
    return task


async def run(args: argparse.Namespace) -> None:
    # Heavy imports (whisper, piper, playwright) only once we really start.
    from .app import Assistant
    from .browser import Browser
    from .face import FaceServer
    from .tools import Narrator, camera_tools, demo_tools, ridebly_tools, web_tools

    asyncio.get_running_loop().set_exception_handler(_quiet_connection_resets)
    settings = Settings.load()
    bus = EventBus()
    face = FaceServer(settings, bus)
    browser = None if args.no_browser else Browser(settings, bus)
    # Background helpers may end on their own (e.g. no camera); the program keeps running.
    helpers = [asyncio.create_task(print_conversation(bus))]
    # When any of these ends, the program ends.
    essentials = [asyncio.create_task(face.run())]
    try:
        await face.wait_started()
        tools = []
        narrator = Narrator()  # lets demos speak; connected once the assistant exists
        if browser:
            await browser.start(face.url)
            essentials.append(asyncio.create_task(browser.closed.wait()))
            helpers.append(_helper(browser.follow_state()))
            tools += web_tools(browser) + ridebly_tools(browser)
            tools += demo_tools(
                browser, settings.ridebly_app_url, settings.demos_dir, narrator, settings.ridebly_approval_wait_s
            )
        else:
            print(f"Open {face.url} in a browser to see the face.")
        if settings.camera_enabled:
            from .vision import Vision

            vision = Vision(settings, bus)
            helpers.append(_helper(vision.run()))
            tools += camera_tools(vision)
        # Load the models in a thread, so the face is already on screen (eyes closed) meanwhile.
        record_dir = PROJECT_ROOT / "data" / "recordings" if args.record else None
        if record_dir:
            print(f"Recording what you say to {record_dir}")
        assistant = await asyncio.to_thread(Assistant, settings, bus, tools, record_dir)
        narrator.say = assistant.narrate
        essentials.append(asyncio.create_task(assistant.run_text() if args.text else assistant.run()))

        # Run until the conversation ends, the window is closed, or something fails.
        done, _ = await asyncio.wait(essentials, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()  # re-raise errors
    finally:
        tasks = helpers + essentials
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if browser:
            await browser.close()


async def login() -> None:
    """Open the assistant's browser on the Ridebly app so staff can log in once; the login is kept in
    the assistant's browser profile for later runs."""
    from playwright.async_api import async_playwright

    settings = Settings.load()
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(settings.browser_profile_dir),
            headless=False,
            no_viewport=True,
            args=["--start-maximized"],
            ignore_default_args=["--enable-automation"],
        )
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(settings.ridebly_app_url)
        print("Log in to the Ridebly app in the browser window, then close the window.")
        closed = asyncio.Event()
        context.on("close", lambda _: closed.set())
        await closed.wait()
    print("Saved. The assistant will use this login from now on.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="assistant", description=__doc__)
    parser.add_argument("--text", action="store_true", help="type instead of using the microphone")
    parser.add_argument("--no-browser", action="store_true", help="don't open the face window (open the URL yourself)")
    parser.add_argument("--record", action="store_true", help="save what you say (and what was understood) to data/recordings")
    parser.add_argument("--login", action="store_true", help="open the Ridebly app to log in once (the assistant remembers it)")
    parser.add_argument("--mic-test", action="store_true", help="record a few seconds and show what is heard and understood")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="show timings and state changes")
    args = parser.parse_args()

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # Greek text on the Windows console

    if args.mic_test:
        logging.basicConfig(level=logging.WARNING)
        from .mic_test import run as mic_test

        mic_test()
        return

    if args.login:
        asyncio.run(login())
        return

    if args.list_devices:
        import sounddevice

        print(sounddevice.query_devices())
        return

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for ours in ("assistant", "state"):
        logging.getLogger(ours).setLevel(logging.DEBUG if args.verbose else logging.INFO)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
