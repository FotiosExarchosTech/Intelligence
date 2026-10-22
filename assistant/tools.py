"""The tools the LLM can use: the camera, and the web browser next to the face."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from .browser import Browser
from .demos import DemoRunner, NotLoggedIn, load_demos
from .llm import Tool, ToolResult
from .vision import Vision

_NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _page_result(page: dict, intro: str) -> ToolResult:
    return ToolResult(f"{intro}\n{json.dumps(page, ensure_ascii=False)}")


def camera_tools(vision: Vision) -> list[Tool]:
    async def look(arguments: dict) -> ToolResult:
        jpeg = await asyncio.to_thread(vision.snapshot_jpeg)
        if jpeg is None:
            return ToolResult("The camera isn't working right now, so you can't see anything.")
        return ToolResult("Here is what the camera sees right now.", jpeg)

    return [
        Tool(
            name="look",
            description="Take a picture with your camera to see the person in front of you and their surroundings.",
            parameters=_NO_ARGS,
            run=look,
        )
    ]


class Narrator:
    """Lets tools speak with the assistant's voice, in step with what they show. Connected to the
    assistant once it exists (tools are created before it)."""

    def __init__(self) -> None:
        self.say: Callable[[str, str], Awaitable[None]] | None = None

    async def __call__(self, text: str, language: str) -> None:
        if self.say is not None:
            await self.say(text, language)


def demo_tools(
    browser: Browser, app_url: str, demos_dir: Path, narrator: Narrator, approval_wait_s: float
) -> list[Tool]:
    runner = DemoRunner(browser, app_url, narrator)
    demos = load_demos(demos_dir)

    async def open_ridebly_app(args: dict) -> ToolResult:
        try:
            page = await runner.open_app(args.get("path", ""))
        except NotLoggedIn:
            return ToolResult(
                "The app's login page is on screen. A live demo of the Ridebly app needs approval from Fotis "
                "Exarchos, who gives it by logging in here; you never log in yourself. Tell the person this in "
                "a sentence or two, then call wait_for_approval."
            )
        return _page_result(page, "The Ridebly app is on screen:")

    async def wait_for_approval(args: dict) -> ToolResult:
        try:
            page = await runner.wait_for_login(approval_wait_s)
        except NotLoggedIn:
            minutes = round(approval_wait_s / 60)
            return ToolResult(
                f"Nobody logged in within {minutes} minutes, so the demo isn't approved for now. Tell the person "
                "they can ask again later, and offer the website tour or the prices instead."
            )
        return _page_result(page, "Approved: someone logged in, and the Ridebly app is on screen. Carry on with the demo:")

    async def start_demo(args: dict) -> ToolResult:
        demo = demos.get(args["name"])
        if demo is None:
            return ToolResult(f"There is no demo called {args['name']!r}. Available: {', '.join(demos) or 'none'}.")
        return ToolResult(await runner.run(demo, args.get("language", "en")))

    tools = [
        Tool(
            "open_ridebly_app",
            "Open the Ridebly app itself (the management panel businesses use) to show it live. Returns the "
            "screen's text, links and buttons; then use click, read_page, point_at, scroll and type_text to "
            "walk through it while explaining. Tells you when approval (a login) is needed first.",
            _schema({"path": {"type": "string", "description": "Optional page path within the app"}}, []),
            open_ridebly_app,
        ),
        Tool(
            "wait_for_approval",
            f"Wait (up to {round(approval_wait_s / 60)} minutes) for Fotis Exarchos to approve a live app demo by "
            "logging in on screen. Only after telling the person you're waiting for approval.",
            _NO_ARGS,
            wait_for_approval,
        ),
    ]
    if demos:
        catalogue = "; ".join(f"{d.name}: {d.title}. {d.description}".strip() for d in demos.values())
        tools.append(
            Tool(
                "start_demo",
                "Run a prepared, narrated demo: it shows each step on screen and speaks its lines itself, so "
                "don't repeat them; afterwards ask if they have questions. Say one short sentence first. "
                f"Available demos: {catalogue}",
                _schema(
                    {
                        "name": {"type": "string", "enum": list(demos)},
                        "language": {"type": "string", "enum": ["en", "el"], "description": "The person's language"},
                    },
                    ["name"],
                ),
                start_demo,
            )
        )
    return tools


def ridebly_tools(browser: Browser) -> list[Tool]:
    async def show_ridebly_pricing(args: dict) -> ToolResult:
        result = await browser.ridebly_pricing(
            args["business"], args.get("vehicles"), args.get("users"), args.get("language", "en")
        )
        return ToolResult(
            "The pricing page is on screen with the calculator set and the monthly price highlighted. "
            "Calculator now shows (the plan steps may differ slightly from what was asked):\n"
            + json.dumps(result, ensure_ascii=False)
        )

    return [
        Tool(
            "show_ridebly_pricing",
            "Show Ridebly's prices on its pricing page: opens the page in the person's language, sets the price "
            "calculator for their business (watch it count up) and highlights the monthly price. Use it whenever "
            "someone asks what Ridebly costs, and read the exact numbers it returns. Transport plans come in steps "
            "of N vehicles with 2N users, so the plan shown is the smallest one covering both numbers; say what it "
            "includes.",
            _schema(
                {
                    "business": {
                        "type": "string",
                        "enum": ["transport", "car_rental"],
                        "description": "transport = transfers, shuttles, tourist transport; car_rental = rent a car",
                    },
                    "vehicles": {"type": "integer", "description": "Fleet size, if the person said it"},
                    "users": {"type": "integer", "description": "Staff users (transport only), if the person said it"},
                    "language": {"type": "string", "enum": ["en", "el"], "description": "The person's language"},
                },
                ["business"],
            ),
            show_ridebly_pricing,
        )
    ]


def web_tools(browser: Browser) -> list[Tool]:
    async def search_web(args: dict) -> ToolResult:
        results = await browser.search(args["query"])
        if not results:
            return ToolResult("The search found nothing; the results page is on screen.")
        return ToolResult("The search results are on screen. Top results:\n" + json.dumps(results, ensure_ascii=False))

    async def open_website(args: dict) -> ToolResult:
        return _page_result(await browser.open(args["url"]), "The page is on screen now:")

    async def read_page(args: dict) -> ToolResult:
        return _page_result(await browser.describe(), "The page on screen:")

    async def click(args: dict) -> ToolResult:
        return _page_result(await browser.click(args["text"]), "Clicked. The page on screen now:")

    async def scroll(args: dict) -> ToolResult:
        view = await browser.scroll(args["direction"])
        return ToolResult("Scrolled. Now visible:\n" + json.dumps(view, ensure_ascii=False))

    async def point_at(args: dict) -> ToolResult:
        await browser.point_at(args["text"])
        return ToolResult("It is highlighted on screen for a few seconds.")

    async def type_text(args: dict) -> ToolResult:
        page = await browser.type_text(args.get("field", ""), args["text"], args.get("submit", False))
        return _page_result(page, "Typed. The page on screen now:")

    async def press_key(args: dict) -> ToolResult:
        return _page_result(await browser.press_key(args["key"]), f"Pressed {args['key']}. The page on screen now:")

    async def desmos(args: dict) -> ToolResult:
        results = await browser.desmos(
            args["expressions"], args.get("mode", "2d"), args.get("clear", False), args.get("bounds")
        )
        rejected = sum(r["result"] != "shown" for r in results)
        summary = "All expressions are on screen." if not rejected else (
            f"{rejected} expression(s) were rejected and removed; fix them and add them again."
        )
        return ToolResult(f"{summary}\n{json.dumps(results, ensure_ascii=False)}")

    async def look_at_screen(args: dict) -> ToolResult:
        return ToolResult("Here is what the web page on screen looks like right now.", await browser.screenshot_jpeg())

    async def go_back(args: dict) -> ToolResult:
        return _page_result(await browser.back(), "Went back. The page on screen now:")

    async def show_face(args: dict) -> ToolResult:
        await browser.show_face()
        return ToolResult("The web page is hidden; your face is on screen again.")

    text = {"type": "string", "description": "Visible text of the element, as on the page"}
    return [
        Tool(
            "search_web",
            "Search the web and show the results page on the screen. Returns the top results (title and url).",
            _schema({"query": {"type": "string", "description": "What to search for"}}, ["query"]),
            search_web,
        ),
        Tool(
            "open_website",
            "Open a web page on the screen for the person to see. Returns its title, the start of its text "
            "and the links and buttons on it.",
            _schema({"url": {"type": "string", "description": "Web address, e.g. https://www.example.com"}}, ["url"]),
            open_website,
        ),
        Tool("read_page", "Read the page on screen again: its text, links, buttons and input fields.", _NO_ARGS, read_page),
        Tool("click", "Click a link or button on the page on screen, by its visible text.", _schema({"text": text}, ["text"]), click),
        Tool(
            "scroll",
            "Scroll the page on screen and get the text that is now visible.",
            _schema({"direction": {"type": "string", "enum": ["down", "up", "top", "bottom"]}}, ["direction"]),
            scroll,
        ),
        Tool(
            "point_at",
            "Scroll to some text on the page and highlight it, to show the person exactly what you're talking about.",
            _schema({"text": text}, ["text"]),
            point_at,
        ),
        Tool(
            "type_text",
            "Type into a field on the page, key by key like a person: search boxes, forms, text editors "
            "and math editors such as Desmos (there '^' starts an exponent and '/' a fraction; press "
            "ArrowRight to leave them). Replaces what the field contained.",
            _schema(
                {
                    "field": {
                        "type": "string",
                        "description": "The field's label or placeholder, as listed in the page's fields. "
                        "Leave empty for the page's main field.",
                    },
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "description": "Press Enter afterwards (default false)"},
                },
                ["text"],
            ),
            type_text,
        ),
        Tool(
            "press_key",
            "Press a key on the page on screen, e.g. Enter to confirm, ArrowRight to leave an exponent, Escape to close a pop-up.",
            _schema(
                {"key": {"type": "string", "enum": ["Enter", "Tab", "Escape", "Backspace", "Delete", "Space", "ArrowUp",
                                                   "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"]}},
                ["key"],
            ),
            press_key,
        ),
        Tool(
            "desmos",
            "Draw on the Desmos graphing calculator (opens it if needed): adds many expressions at once and "
            "reports any Desmos rejects, so you can fix them. Use this instead of typing for anything in Desmos. "
            "Expressions are Desmos LaTeX, e.g. y=x^{2}+3, y=\\sin(x), (x-1)^{2}+y^{2}=4. "
            "Build people, animals and objects in 3d from solid, colored shapes (z is up): "
            "ball: \\operatorname{sphere}((x_0,y_0,z_0),r); "
            "upright cylinder: (x-a)^{2}+(y-b)^{2}=r^{2}\\left\\{z_{1}<z<z_{2}\\right\\}; "
            "box centred at (a,b,c) with half-sizes w,d,h: "
            "\\max\\left(\\frac{\\left|x-a\\right|}{w},\\frac{\\left|y-b\\right|}{d},\\frac{\\left|z-c\\right|}{h}\\right)=1; "
            "thin line: \\operatorname{segment}((x_1,y_1,z_1),(x_2,y_2,z_2)). "
            "Write absolute values as \\left|...\\right|, never a bare |. There are no cylinder() or cone() "
            "functions. For a drawing of something, look_at_screen afterwards to check it and fix what looks wrong.",
            _schema(
                {
                    "expressions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "latex": {"type": "string"},
                                "color": {"type": "string", "description": "Optional, like #c74440"},
                            },
                            "required": ["latex"],
                            "additionalProperties": False,
                        },
                    },
                    "mode": {"type": "string", "enum": ["2d", "3d"], "description": "Calculator to use (default 2d)"},
                    "clear": {"type": "boolean", "description": "Start from an empty graph first (default false)"},
                    "bounds": {
                        "type": "object",
                        "description": "Optional zoom: the visible range, e.g. {xmin:-2,xmax:2,ymin:-2,ymax:2,zmin:0,zmax:3} (z only in 3d). In 3d it fits the numbers used by default.",
                        "properties": {k: {"type": "number"} for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")},
                        "additionalProperties": False,
                    },
                },
                ["expressions"],
            ),
            desmos,
        ),
        Tool(
            "look_at_screen",
            "See the web page on screen as a picture, e.g. to check a drawing you made or a page's layout.",
            _NO_ARGS,
            look_at_screen,
        ),
        Tool("go_back", "Go back to the previous page.", _NO_ARGS, go_back),
        Tool("show_face", "Hide the web page and show your face again. Use when the person is done looking.", _NO_ARGS, show_face),
    ]
