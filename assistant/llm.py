"""Conversation with Claude or Ollama, streamed sentence by sentence so speech can start early."""

from __future__ import annotations

import base64
import logging
import os
import re
import time
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import anthropic
import ollama

from .config import Settings
from .language import LANGUAGE_NAMES

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are {name}, a friendly assistant that people talk to out loud. You appear as an \
animated face on a screen, hear people through a microphone and answer through a speaker.

Everything you write is read aloud by a text-to-speech voice, so write plain spoken \
sentences: no markdown, lists, headings, emoji, links or code. Keep replies short, \
usually one to three sentences, and offer to say more rather than giving long explanations.

You speak Greek and English. Answer in the language the person used, and write Greek in \
Greek script. The words you receive come from speech recognition and may be misheard; if \
a request seems garbled, make your best guess or briefly ask the person to repeat.

You represent Ridebly, and people come to you to learn about it. Speech recognition often \
mangles that name (into things like "Ride Bly", "VainoPi" or a Greek spelling), so a \
similar-sounding word probably means Ridebly. Explain it simply, in terms of what it does \
for the person's business, and offer to show it. Only state what the information below \
says; if you don't know something, say so and suggest booking a demo with the Ridebly team.

What you know about Ridebly:

{knowledge}"""

CAMERA_PROMPT = """

You can see through a camera with the look tool. Use it whenever answering needs your \
eyes: the person asks if you can see them, what they look like, what they're wearing or \
holding, or about the room. Look again each time rather than relying on an earlier \
picture, since people move and change places. Describe what you see naturally and kindly. Never try to work \
out who someone is from their appearance, and don't comment on sensitive traits such as \
health, ethnicity or age guesses unless asked."""

WEB_PROMPT = """

You also control a web browser on the same screen as your face. When the person wants to \
see something online, show it: open a well-known address directly with open_website, or \
use search_web and then open the best result (don't type into Google's own search page: it \
blocks automated browsers). For "where is it" questions open Google Maps \
(https://www.google.com/maps/search/ followed by the place). Then explain what is on screen in a few \
short spoken sentences, scrolling, clicking or using point_at so they can follow along. \
Before each action say one short sentence (like "Let me pull that up."), so nobody waits \
in silence. Never announce that you will show, open or start something without calling the \
tool for it in the same reply; otherwise nothing happens on screen. Read and summarise pages in the person's language even when the page is in \
another language. Close cookie or pop-up banners if they cover the page. The page stays on \
screen while you talk about it; call show_face when the person is done or asks to see you.

This screen is in a public place: only show content that suits a business setting and \
politely decline anything adult, hateful, violent or illegal. Never log in to accounts, \
type personal or payment details, or download anything."""

APP_PROMPT = """

You can show the Ridebly app itself, live. When a prepared demo fits what the person wants to \
see, use start_demo; otherwise open_ridebly_app and walk through the screens yourself, \
explaining each one in a sentence or two as you click. {rules}"""

APP_RULES_TEST = """The app on this screen is a test server: you may create example bookings, \
customers or vehicles to show how things work, using made-up names and details (never a real \
person's details or payment data)."""

APP_RULES_LIVE = """This is a real business account: look around and explain, but never \
create, change or delete anything."""

_MAX_TOOL_ROUNDS = 10


@dataclass(frozen=True)
class ToolResult:
    text: str
    image_jpeg: bytes | None = None


@dataclass(frozen=True)
class Tool:
    """A function the LLM can call. `parameters` is a JSON schema for its arguments."""

    name: str
    description: str
    parameters: dict
    run: Callable[[dict], Awaitable[ToolResult]]


# A sentence ends at . ! ? ; (the Greek question mark) or … followed by whitespace, or at a newline.
_SENTENCE_END = re.compile(r"[.!?;;…]+[\"'»)\]]*\s+|\n+")
_MARKDOWN = re.compile(r"[*_#`>]+|^\s*[-•]\s+", re.MULTILINE)


class SentenceSplitter:
    """Accumulates streamed text and emits complete sentences."""

    def __init__(self, min_chars: int = 12) -> None:
        self._buffer = ""
        self._min_chars = min_chars  # merge very short fragments ("Hi.") into the next sentence

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences, start = [], 0
        for match in _SENTENCE_END.finditer(self._buffer):
            candidate = _clean(self._buffer[start : match.end()])
            if len(candidate) >= self._min_chars:
                sentences.append(candidate)
                start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str:
        rest, self._buffer = _clean(self._buffer), ""
        return rest


# Anything but Latin and Greek letters, digits, punctuation and common symbols. Models
# occasionally slip a word from another script into Greek (e.g. Thai), which the voice can't say.
_OTHER_SCRIPTS = re.compile("[^\u0000-ɏͰ-Ͽἀ-῿ -⁯₠-⃏℀-⅏]")


def _clean(text: str) -> str:
    # NFC first, so accents written as separate combining marks join their letters.
    text = unicodedata.normalize("NFC", _MARKDOWN.sub("", text))
    return " ".join(_OTHER_SCRIPTS.sub("", text).split())


class Conversation:
    """Conversation history, tools and sentence streaming; subclasses talk to a specific LLM."""

    def __init__(self, settings: Settings, tools: list[Tool]) -> None:
        self._settings = settings
        self._tools = {tool.name: tool for tool in tools}
        # Every .md file in the knowledge folder, e.g. one about Ridebly and one about the team.
        files = sorted(settings.knowledge_dir.glob("*.md")) if settings.knowledge_dir.is_dir() else []
        knowledge = "\n\n".join(f.read_text(encoding="utf-8").strip() for f in files) or (
            "Nothing yet: say you don't have the details and suggest booking a demo with the Ridebly team."
        )
        self._system = SYSTEM_PROMPT.format(name=settings.assistant_name, knowledge=knowledge)
        if "look" in self._tools:
            self._system += CAMERA_PROMPT
        if "open_website" in self._tools:
            self._system += WEB_PROMPT
        if "open_ridebly_app" in self._tools:
            rules = APP_RULES_TEST if settings.ridebly_app_test_server else APP_RULES_LIVE
            self._system += APP_PROMPT.format(rules=rules)
        self._messages: list[dict] = []
        self._turn_start = 0  # index of this turn's user message
        self._last_turn = 0.0
        self._language = "en"
        # Set by _stream() when the reply is complete; None means the model declined.
        self._reply_content: object = None

    async def reply(self, text: str, language: str) -> AsyncIterator[str]:
        """Sends the user's words (spoken in `language`) and yields the reply one sentence at a time."""
        self._language = language
        if self._messages and time.monotonic() - self._last_turn > self._settings.conversation_timeout_s:
            log.info("Conversation idle for a while; starting a fresh one")
            self._messages = []

        self._turn_start = len(self._messages)
        self._messages.append({"role": "user", "content": text})
        self._reply_content = None
        splitter = SentenceSplitter()
        previous = None
        try:
            async for delta in self._stream():
                for sentence in splitter.feed(delta):
                    # Models sometimes repeat their "one moment" line after a tool call.
                    if sentence != previous:
                        yield sentence
                    previous = sentence
        except BaseException:
            # Failed or interrupted: drop the unanswered turn so the history stays valid.
            del self._messages[self._turn_start :]
            raise

        if self._reply_content is None:
            del self._messages[self._turn_start :]
            return
        if rest := splitter.flush():
            yield rest
        self._messages.append({"role": "assistant", "content": self._reply_content})
        self._last_turn = time.monotonic()

    def _stream(self) -> AsyncIterator[str]:
        """Yields text deltas of the reply to self._messages and sets self._reply_content.

        Tool calls are handled inside: their messages are appended to self._messages, and a
        newline is yielded after each round so text before a tool call is spoken on its own.
        """
        raise NotImplementedError

    async def _run_tool(self, name: str, arguments: object) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"There is no tool called {name}.")
        if not isinstance(arguments, dict):
            return ToolResult("The tool arguments were not valid JSON; try again.")
        log.info("Tool call: %s %s", name, arguments or "")
        try:
            return await tool.run(arguments)
        except Exception as exc:
            log.warning("Tool %s failed: %s", name, str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
            log.debug("Tool failure details", exc_info=True)
            return ToolResult(f"The tool failed: {exc}")


class ClaudeConversation(Conversation):
    def __init__(self, settings: Settings, tools: list[Tool]) -> None:
        super().__init__(settings, tools)
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            log.warning("ANTHROPIC_API_KEY is not set (add it to .env) - Claude requests will fail")
        self._client = anthropic.AsyncAnthropic()
        self._tool_specs = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters, "eager_input_streaming": True}
            for t in tools
        ]

    async def _stream(self) -> AsyncIterator[str]:
        s = self._settings
        for round_ in range(_MAX_TOOL_ROUNDS):
            # On the last round tools are off, so the turn always ends with something said.
            last = round_ == _MAX_TOOL_ROUNDS - 1
            try:
                async with self._client.beta.messages.stream(
                    model=s.llm_model,
                    max_tokens=s.llm_max_tokens,
                    system=self._system,
                    messages=self._messages,
                    output_config={"effort": s.llm_effort},
                    cache_control={"type": "ephemeral"},
                    # If a safety classifier declines, the API retries on Anthropic's
                    # recommended fallback model inside the same request.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **({"tools": self._tool_specs} if self._tool_specs else {}),
                    **({"tool_choice": {"type": "none"}} if last and self._tool_specs else {}),
                ) as stream:
                    async for delta in stream.text_stream:
                        yield delta
                    final = await stream.get_final_message()
            except ValueError:
                # A streamed tool input that wasn't valid JSON: ask again (bounded by the loop).
                log.warning("Claude sent an unreadable tool call; retrying")
                continue

            if final.stop_reason == "refusal":
                log.warning("Claude declined to answer (%s)", final.stop_details)
                return
            tool_uses = [block for block in final.content if block.type == "tool_use"]
            if final.stop_reason != "tool_use":
                if not tool_uses:
                    # Keep the full content (not just the text) so thinking blocks stay valid next turn.
                    self._reply_content = final.content
                return  # otherwise a tool call was cut off by max_tokens: the turn is dropped

            self._messages.append({"role": "assistant", "content": final.content})
            yield "\n"
            results = []
            for block in tool_uses:
                result = await self._run_tool(block.name, block.input)
                content: list[dict] = [{"type": "text", "text": result.text}]
                if result.image_jpeg:
                    data = base64.standard_b64encode(result.image_jpeg).decode()
                    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
            self._messages.append({"role": "user", "content": results})


class OllamaConversation(Conversation):
    """Local models, or Ollama cloud models (names ending in cloud; needs `ollama signin`)."""

    def __init__(self, settings: Settings, tools: list[Tool]) -> None:
        super().__init__(settings, tools)
        self._client = ollama.AsyncClient(host=settings.ollama_host)
        self._tool_specs = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]

    async def _stream(self) -> AsyncIterator[str]:
        s = self._settings
        # Smaller models drift into the wrong language, so name it on every turn. (Claude
        # doesn't need this, and changing its system prompt would break its prompt cache.)
        language = LANGUAGE_NAMES[self._language]
        system = {"role": "system", "content": f"{self._system}\n\nThe person is speaking {language}. Reply in {language}."}
        text = ""
        for round_ in range(_MAX_TOOL_ROUNDS):
            # On the last round tools are off, so the turn always ends with something said.
            last = round_ == _MAX_TOOL_ROUNDS - 1
            parts: list[str] = []
            calls = []
            async for chunk in await self._client.chat(
                model=s.ollama_model,
                messages=[system, *self._history()],
                tools=None if last else (self._tool_specs or None),
                think=s.ollama_think,
                stream=True,
            ):
                if delta := chunk.message.content:
                    parts.append(delta)
                    yield delta
                calls += chunk.message.tool_calls or []
            text = "".join(parts)
            if not calls:
                break

            self._messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [{"function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls],
            })
            yield "\n"
            images = []
            for call in calls:
                result = await self._run_tool(call.function.name, call.function.arguments)
                self._messages.append({"role": "tool", "tool_name": call.function.name, "content": result.text})
                if result.image_jpeg:
                    images.append(result.image_jpeg)
            if images:
                # Images inside tool messages garble some models' output; a user message works everywhere.
                self._messages.append({"role": "user", "content": "(camera picture)", "images": images})
        self._reply_content = text

    def _history(self) -> list[dict]:
        """The messages, with pictures from earlier turns left out so they aren't re-sent every time."""
        return [
            {**m, "images": None, "content": "(camera picture, no longer shown)"}
            if m.get("images") and i < self._turn_start
            else m
            for i, m in enumerate(self._messages)
        ]


def create_conversation(settings: Settings, tools: list[Tool]) -> Conversation:
    if settings.llm_provider == "ollama":
        log.info("LLM: Ollama %s", settings.ollama_model)
        return OllamaConversation(settings, tools)
    log.info("LLM: Claude %s", settings.llm_model)
    return ClaudeConversation(settings, tools)
