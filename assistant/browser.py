"""One Chromium window (via Playwright): the face in one tab, web pages the assistant shows in another."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from urllib.parse import quote_plus, urlparse

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from .config import Settings
from .events import EventBus, InterruptRequested, State, StateChanged

log = logging.getLogger(__name__)

_SEARCH_URL = "https://duckduckgo.com/?q={query}&kl=gr-el"
_SEARCH_RESULT = 'a[data-testid="result-title-a"]'
_MAX_TEXT = 2500  # characters of page text handed to the LLM
_MAX_LINKS = 30
# Cookie banner buttons, most privacy-friendly first. Clicked automatically so banners
# don't cover what the assistant is showing.
_CONSENT_BUTTONS = [
    "Reject all", "Απόρριψη όλων", "Decline all", "Decline", "Απόρριψη",
    "Accept all", "Αποδοχή όλων", "Accept", "Agree", "I agree", "Αποδοχή", "Συμφωνώ", "Got it", "OK",
]

# A small glowing orb with the assistant's eyes in the corner of every web page, so people can see
# when he is listening, thinking or speaking while his face is hidden.
_OVERLAY_SCRIPT = """
(() => {
  if (location.hostname === "127.0.0.1" || window.top !== window) return;
  const COLORS = { idle: "90 209 255", listening: "110 240 160", thinking: "90 209 255", speaking: "90 209 255" };
  let host, orb;
  function ensure() {
    if (host && host.isConnected) return;
    host = document.createElement("assistant-orb");
    const root = host.attachShadow({ mode: "closed" });
    root.innerHTML = `
      <style>
        .orb { position: fixed; right: 24px; bottom: 24px; width: 72px; height: 72px; border-radius: 50%;
               background: #05070d; display: flex; gap: 12px; align-items: center; justify-content: center;
               z-index: 2147483647; cursor: pointer; border: 0; padding: 0; --c: 90 209 255;
               box-shadow: 0 0 18px rgb(var(--c) / 0.7), 0 4px 18px #0008; transition: box-shadow .3s, scale .3s; }
        .orb:hover { scale: 1.08; }
        .eye { width: 11px; height: 22px; border-radius: 6px; background: rgb(var(--c));
               box-shadow: 0 0 8px rgb(var(--c)); transition: background .3s, height .3s; }
        .stop { display: none; width: 24px; height: 24px; border-radius: 5px; background: rgb(var(--c)); }
        .orb.listening { scale: 1.12; }
        .orb.thinking .eye, .orb.speaking .eye { display: none; }
        .orb.thinking .stop, .orb.speaking .stop { display: block; }
        .orb.speaking { animation: talk .5s ease-in-out infinite alternate; }
        @keyframes talk { to { box-shadow: 0 0 30px rgb(var(--c) / 0.95), 0 4px 18px #0008; } }
        @media (prefers-reduced-motion: reduce) { .orb.speaking { animation: none; } }
      </style>
      <button class="orb" type="button"><div class="eye"></div><div class="eye"></div><div class="stop"></div></button>`;
    orb = root.querySelector(".orb");
    // While the assistant is busy the orb is a stop button; otherwise it brings the face back.
    orb.addEventListener("click", () => window.__assistantOrbClicked && window.__assistantOrbClicked());
    document.documentElement.append(host);
  }
  window.__assistantState = (state) => {
    ensure();
    orb.className = "orb " + state;
    orb.style.setProperty("--c", COLORS[state] || COLORS.idle);
    const busy = state === "thinking" || state === "speaking";
    orb.title = orb.ariaLabel = busy ? "Stop" : "Back";
  };
  document.addEventListener("DOMContentLoaded", () => window.__assistantState(window.__assistantLastState || "idle"));
})();
"""

# Anything a person could type into: inputs, text areas, search/combo boxes, rich editors
# and math editors (Desmos and similar draw their own field around a hidden textarea).
_EDITABLE = (
    "input:not([type=hidden]):not([type=submit]):not([type=button]):not([type=checkbox]):not([type=radio])"
    ":not([type=file]):not([type=range]), textarea, [contenteditable=''], [contenteditable=true],"
    " [role=textbox], [role=searchbox], [role=combobox], [class*=math-field]"
)
_DESMOS_URLS = {"2d": "https://www.desmos.com/calculator", "3d": "https://www.desmos.com/3d"}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# The site remembers the last language in a cookie, so always ask for one explicitly.
_PRICING_URLS = {"en": "https://www.ridebly.com/en/pricing", "el": "https://www.ridebly.com/el/pricing"}
_PRICING_SECTIONS = {"transport": "#transfers-pricing", "car_rental": "#rent-a-car-pricing"}
# Outline the calculator's monthly price: the biggest "€ 12.34"-style text in the section.
_HIGHLIGHT_PRICE_SCRIPT = """
(section) => {
  const prices = [...section.querySelectorAll("*")].filter((el) =>
    el.children.length === 0 && /^€\\s?[\\d.,]+$/.test(el.textContent.trim()));
  const price = prices.sort((a, b) => parseFloat(getComputedStyle(b).fontSize) - parseFloat(getComputedStyle(a).fontSize))[0];
  if (!price) return;
  const old = price.style.outline, oldOffset = price.style.outlineOffset;
  price.style.outline = "4px solid rgb(90 209 255)";
  price.style.outlineOffset = "6px";
  setTimeout(() => { price.style.outline = old; price.style.outlineOffset = oldOffset; }, 6000);
}
"""


def _fit_bounds(latexes) -> dict | None:
    """A 3D view just big enough for the coordinates used, so drawings fill the screen
    (Desmos's default view is -5..5, where a 2-unit-tall figure looks tiny)."""
    numbers = [float(n) for latex in latexes for n in _NUMBER.findall(latex)]
    if not numbers:
        return None
    size = min(max(max(numbers) * 1.25, 1.0), 50.0)
    return {"xmin": -size, "xmax": size, "ymin": -size, "ymax": size, "zmin": -size, "zmax": size}


_FOCUS_IS_EDITABLE = """
(editable) => {
  const el = document.activeElement;
  return !!el && el !== document.body && (el.isContentEditable || el.matches(editable) || !!el.closest("[class*=math-field]"));
}
"""
_KEYS = {"Enter", "Tab", "Escape", "Backspace", "Delete", "Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
         "Home", "End", "PageUp", "PageDown"}

# Text and clickable things of the current page, for the LLM.
_DESCRIBE_SCRIPT = """
([maxText, maxLinks, EDITABLE]) => {
  const visible = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const text = (document.querySelector("main") || document.body).innerText.replace(/\\n{3,}/g, "\\n\\n").trim();
  const seen = new Set();
  const links = [];
  for (const el of document.querySelectorAll("a[href], button, [role=button], input[type=submit]")) {
    const label = (el.innerText || el.value || el.getAttribute("aria-label") || "").trim().replace(/\\s+/g, " ");
    if (!label || label.length > 80 || seen.has(label) || !visible(el)) continue;
    seen.add(label);
    links.push(label);
    if (links.length >= maxLinks) break;
  }
  const fields = [...document.querySelectorAll(EDITABLE)]
    .filter(visible)
    .map((el) => el.getAttribute("aria-label") || el.placeholder || el.name ||
                 (el.className.toString().includes("math-field") ? "math input" : el.getAttribute("role") || "text field"))
    .slice(0, 8);
  return { title: document.title, url: location.href, text: text.slice(0, maxText), links, fields };
}
"""

_HIGHLIGHT_SCRIPT = """
(el) => {
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  const old = el.style.outline, oldOffset = el.style.outlineOffset, oldRadius = el.style.borderRadius;
  el.style.outline = "4px solid rgb(90 209 255)";
  el.style.outlineOffset = "4px";
  el.style.borderRadius = "6px";
  setTimeout(() => { el.style.outline = old; el.style.outlineOffset = oldOffset; el.style.borderRadius = oldRadius; }, 5000);
}
"""


class Browser:
    def __init__(self, settings: Settings, bus: EventBus) -> None:
        self._settings = settings
        self._bus = bus
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self.face: Page | None = None
        self._page: Page | None = None  # the tab with web pages the assistant shows
        self.showing = False
        self._state = State.IDLE
        self._desmos_count = 0  # for unique Desmos expression ids
        self.closed = asyncio.Event()  # set when the user closes the window

    async def start(self, face_url: str) -> None:
        s = self._settings
        self._playwright = await async_playwright().start()
        # A persistent profile keeps cookies, so logins and cookie choices survive restarts.
        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(s.browser_profile_dir),
            headless=False,
            no_viewport=True,  # pages take the real window size
            accept_downloads=False,
            args=["--kiosk"] if s.face_fullscreen else ["--start-maximized"],
            # Hides the "Chrome is being controlled by automated test software" bar.
            ignore_default_args=["--enable-automation"],
        )
        await self._context.expose_function("__assistantOrbClicked", self._orb_clicked)
        await self._context.add_init_script(_OVERLAY_SCRIPT)
        self._context.on("close", lambda _: self.closed.set())
        self._context.on("page", self._on_new_page)
        pages = self._context.pages
        self.face = pages[0] if pages else await self._context.new_page()
        # Drop cached files from earlier runs so an updated face page is always used.
        # (Cookies and logins are kept.)
        cdp = await self._context.new_cdp_session(self.face)
        await cdp.send("Network.clearBrowserCache")
        await cdp.detach()
        await self.face.goto(face_url)
        log.info("Browser window open%s", " (fullscreen: Alt+F4 closes it)" if s.face_fullscreen else "")

    async def close(self) -> None:
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass  # already closed by the user
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def follow_state(self) -> None:
        """Keep the orb on web pages in sync, and go back to the face when nobody talks for a while."""
        events = self._bus.subscribe()
        last_active = time.monotonic()
        while True:
            try:
                event = await asyncio.wait_for(events.get(), timeout=5)
            except TimeoutError:
                event = None
            if isinstance(event, StateChanged):
                self._state = event.state
                await self._update_orb()
            if self._state != State.IDLE:
                last_active = time.monotonic()
            elif self.showing and time.monotonic() - last_active > self._settings.browse_idle_return_s:
                log.info("Nobody talked for a while: back to the face")
                await self.show_face()

    # ---------- actions used by the LLM's web tools ----------

    async def open(self, url: str) -> dict:
        if "://" not in url:
            url = "https://" + url
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("Only http and https addresses can be opened.")
        page = await self._content_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        except Exception as exc:
            if "Timeout" not in type(exc).__name__:
                raise
            log.info("Page is slow to load; showing what is there")
        await self._show_content()
        await page.wait_for_timeout(800)  # let scripts render the first view
        await self._dismiss_cookie_banner(page)
        return await self.describe()

    async def search(self, query: str) -> list[dict]:
        page = await self._content_page()
        await page.goto(_SEARCH_URL.format(query=quote_plus(query)), wait_until="domcontentloaded", timeout=20000)
        await self._show_content()
        await page.wait_for_selector(_SEARCH_RESULT, timeout=10000)
        return await page.eval_on_selector_all(
            _SEARCH_RESULT, "els => els.slice(0, 6).map(e => ({ title: e.innerText.trim(), url: e.href }))"
        )

    async def describe(self) -> dict:
        page = self._require_page()
        return await page.evaluate(_DESCRIBE_SCRIPT, [_MAX_TEXT, _MAX_LINKS, _EDITABLE])

    async def click(self, text: str) -> dict:
        page = self._require_page()
        target = await self._find(text, clickable=True)
        await target.click(timeout=5000)
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        await page.wait_for_timeout(800)
        return await self.describe()

    async def point_at(self, text: str) -> None:
        target = await self._find(text, clickable=False)
        await target.evaluate(_HIGHLIGHT_SCRIPT)

    async def scroll(self, direction: str) -> dict:
        page = self._require_page()
        script = {
            "down": "window.scrollBy({ top: innerHeight * 0.8, behavior: 'smooth' })",
            "up": "window.scrollBy({ top: -innerHeight * 0.8, behavior: 'smooth' })",
            "top": "window.scrollTo({ top: 0, behavior: 'smooth' })",
            "bottom": "window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })",
        }[direction]
        await page.evaluate(script)
        await page.wait_for_timeout(700)
        return await page.evaluate(
            "() => ({ at_top: scrollY < 10, at_bottom: innerHeight + scrollY >= document.body.scrollHeight - 10,"
            " visible_text: [...document.querySelectorAll('h1,h2,h3,p,li')]"
            ".filter(e => { const r = e.getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight && r.height > 0; })"
            ".map(e => e.innerText.trim()).filter(Boolean).join('\\n').slice(0, 1500) })"
        )

    async def type_text(self, field: str, text: str, submit: bool) -> dict:
        page = self._require_page()
        box = await self._find_field(field)
        if box is not None:
            await box.click(timeout=5000)
        elif not await page.evaluate(_FOCUS_IS_EDITABLE, _EDITABLE):
            raise LookupError("There is no field to type into on this page.")
        # Type key by key, like a person: works in rich and math editors where filling a
        # value doesn't, and people can watch the text appear.
        await page.keyboard.press("Control+A")
        await page.keyboard.type(text, delay=25)
        if submit:
            await page.keyboard.press("Enter")
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
            except Exception:
                pass
        await page.wait_for_timeout(800)
        return await self.describe()

    async def press_key(self, key: str) -> dict:
        page = self._require_page()
        if key not in _KEYS:
            raise ValueError(f"Unknown key {key!r}; use one of: {', '.join(sorted(_KEYS))}")
        await page.keyboard.press(" " if key == "Space" else key)
        await page.wait_for_timeout(500)
        return await self.describe()

    async def desmos(self, items: list[dict], mode: str, clear: bool, bounds: dict | None = None) -> list[dict]:
        """Add expressions to a Desmos calculator through its own JavaScript API (`Calc`):
        instant, and Desmos reports which expressions it can't understand and why."""
        url = _DESMOS_URLS[mode]
        page = self._page if self.showing else None
        if page is None or not page.url.startswith(url):
            await self.open(url)
            page = self._page
        await page.wait_for_function("() => window.Calc && Calc.setExpressions", timeout=20000)
        if clear:
            await page.evaluate("() => Calc.setBlank()")
        start = self._desmos_count
        self._desmos_count += len(items)
        expressions = [
            {"id": f"mike{start + i}", "latex": item["latex"], **({"color": item["color"]} if item.get("color") else {})}
            for i, item in enumerate(items)
        ]
        await page.evaluate("(expressions) => Calc.setExpressions(expressions)", expressions)
        if not bounds and mode == "3d":
            bounds = _fit_bounds(item["latex"] for item in items)
        if bounds:
            try:
                await page.evaluate("(bounds) => Calc.setMathBounds(bounds)", bounds)
            except Exception as exc:
                log.debug("Desmos rejected the bounds %s: %s", bounds, exc)
        await page.wait_for_timeout(1000)  # let Desmos analyse them
        # Report each expression's result, and take the broken ones off the screen again.
        return await page.evaluate(
            """(expressions) => expressions.map(({ id, latex }) => {
                 const analysis = Calc.expressionAnalysis[id];
                 if (analysis && analysis.isError) {
                   Calc.removeExpression({ id });
                   return { latex, result: "rejected (removed): " + analysis.errorMessage };
                 }
                 return { latex, result: "shown" };
               })""",
            expressions,
        )

    async def ridebly_pricing(self, business: str, vehicles: int | None, users: int | None, language: str) -> dict:
        """Open ridebly.com's pricing page and set its price calculator, click by click so people
        can watch, then highlight the monthly price. Returns what the calculator shows."""
        await self.open(_PRICING_URLS[language])
        page = self._page
        section = page.locator(_PRICING_SECTIONS[business])
        await section.evaluate("el => el.scrollIntoView({ behavior: 'smooth', block: 'start' })")
        await page.wait_for_timeout(800)
        # Transport plans are one scale: plan N = N vehicles and 2N users (the vehicle and user
        # buttons move together). The plan that fits both numbers is the bigger of the two needs.
        # Car rental is simply priced per vehicle.
        plan = max(vehicles or 1, -(-(users or 0) // 2) if business == "transport" else 0, 1)
        # The vehicle +/- buttons are the first pair of icon-only buttons in the calculator.
        steppers = section.locator("button[aria-label]").filter(has_not_text=re.compile(r"\S"))
        reached = await self._step_to(steppers.nth(0), steppers.nth(1), plan)
        await section.evaluate(_HIGHLIGHT_PRICE_SCRIPT)
        result = {"calculator": " ".join((await section.inner_text()).split())[:700]}
        if reached < plan:
            result["note"] = (
                f"The calculator stops at {reached} vehicles; {plan} needs a custom plan from the Ridebly team."
            )
        return result

    async def _step_to(self, decrease, increase, target: int) -> int:
        """Click a calculator's +/- until it shows the target (or can't go further); returns the
        count it ends on."""

        async def count() -> int:
            match = _NUMBER.search(await increase.evaluate("b => b.parentElement.innerText"))
            return int(float(match.group())) if match else 0

        current = await count()
        for _ in range(150):
            if current == target:
                break
            button = increase if current < target else decrease
            if not await button.is_enabled():  # the calculator's limit
                break
            await button.click()
            await self._page.wait_for_timeout(70)  # quick enough for big fleets, slow enough to watch
            after = await count()
            if after == current:
                break
            current = after
        return current

    def current_url(self) -> str | None:
        """The address of the page in the content tab, if one is open."""
        return self._page.url if self._page is not None and not self._page.is_closed() else None

    async def screenshot_jpeg(self) -> bytes:
        page = self._require_page()
        return await page.screenshot(type="jpeg", quality=70)

    async def back(self) -> dict:
        page = self._require_page()
        await page.go_back(wait_until="domcontentloaded", timeout=15000)
        await page.wait_for_timeout(500)
        return await self.describe()

    async def show_face(self) -> None:
        if self.face is not None:
            await self.face.bring_to_front()
        self.showing = False

    # ---------- helpers ----------

    async def _orb_clicked(self) -> None:
        if self._state in (State.THINKING, State.SPEAKING):
            self._bus.publish(InterruptRequested())
        else:
            await self.show_face()

    async def _content_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            self._page = await self._context.new_page()
        return self._page

    def _require_page(self) -> Page:
        if self._page is None or self._page.is_closed() or not self.showing:
            raise RuntimeError("No web page is on screen. Open one first.")
        return self._page

    async def _show_content(self) -> None:
        await self._page.bring_to_front()
        self.showing = True
        await self._update_orb()

    async def _update_orb(self) -> None:
        if self.showing and self._page is not None and not self._page.is_closed():
            try:
                await self._page.evaluate(
                    "s => { window.__assistantLastState = s; window.__assistantState && window.__assistantState(s); }",
                    self._state.value,
                )
            except Exception:
                pass  # page is navigating; the orb picks the state up when it loads

    async def _dismiss_cookie_banner(self, page: Page) -> None:
        for label in _CONSENT_BUTTONS:
            button = page.get_by_role("button", name=label, exact=True)
            try:
                if await button.count() and await button.first.is_visible():
                    await button.first.click(timeout=2000)
                    log.debug("Closed a cookie banner (%s)", label)
                    await page.wait_for_timeout(500)
                    return
            except Exception:
                continue

    async def _find_field(self, field: str):
        """The field to click before typing: the one the LLM names (by label or placeholder);
        otherwise None if a field already has the cursor (e.g. the new line Desmos opens after
        Enter), else the page's first visible field."""
        page = self._require_page()
        if field:
            named = [page.get_by_label(field), page.get_by_placeholder(field)]
            named += [page.get_by_role(role, name=field) for role in ("textbox", "searchbox", "combobox")]
            for candidate in named:
                visible = candidate.filter(visible=True)
                if await visible.count():
                    return visible.first
        if await page.evaluate(_FOCUS_IS_EDITABLE, _EDITABLE):
            return None
        any_field = page.locator(_EDITABLE).filter(visible=True)
        return any_field.first if await any_field.count() else None

    async def _find(self, text: str, clickable: bool):
        page = self._require_page()
        candidates = [page.get_by_role("link", name=text), page.get_by_role("button", name=text)] if clickable else []
        candidates.append(page.get_by_text(text))
        for candidate in candidates:
            for i in range(min(await candidate.count(), 5)):
                if await candidate.nth(i).is_visible():
                    return candidate.nth(i)
        raise LookupError(f'Nothing on the page matches "{text}". Use read_page to see what is there.')

    def _on_new_page(self, page: Page) -> None:
        asyncio.create_task(self._adopt(page))

    async def _adopt(self, page: Page) -> None:
        # Sites that open links in a new tab: show the link in our content tab instead.
        # (Tabs we open ourselves have no opener and are left alone.)
        try:
            if await page.opener() is None:
                return
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            url = page.url
            await page.close()
            if url.startswith("http"):
                await self.open(url)
        except Exception:
            log.debug("Couldn't take over a new tab", exc_info=True)
