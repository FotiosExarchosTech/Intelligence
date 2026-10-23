"""The assistant's state machine: idle -> listening -> thinking -> speaking -> idle."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .audio_in import Microphone, SpeechDiscarded, SpeechStarted, Utterance
from .audio_out import Speaker
from .config import Settings
from .events import (
    AssistantSaid,
    AudioLevel,
    EventBus,
    InterruptRequested,
    LanguageChanged,
    LanguageRequested,
    State,
    StateChanged,
    UserSaid,
)
from .language import LANGUAGE_NAMES, text_language
from .llm import Tool, ToolResult, create_conversation
from .stt import create_speech_to_text
from .tts import TextToSpeech

log = logging.getLogger(__name__)

# After speaking, keep the mic closed a little longer so the room echo dies away.
_ECHO_TAIL_S = 0.3

_TROUBLE = {
    "en": "Sorry, I can't think straight right now. Please try again in a moment.",
    "el": "Συγγνώμη, δεν μπορώ να σκεφτώ αυτή τη στιγμή. Δοκίμασε ξανά σε λίγο.",
}
_NO_ANSWER = {
    "en": "Sorry, I can't help with that.",
    "el": "Συγγνώμη, δεν μπορώ να βοηθήσω με αυτό.",
}


async def _read_line(prompt: str) -> str:
    """input() without blocking the event loop or keeping the program alive when it quits."""
    loop = asyncio.get_running_loop()
    line: asyncio.Future[str] = loop.create_future()

    def read() -> None:
        try:
            text = input(prompt)
        except EOFError:
            text = ""
        loop.call_soon_threadsafe(lambda: line.done() or line.set_result(text))

    threading.Thread(target=read, daemon=True).start()
    return await line


def _save_recording(folder: Path, utterance: Utterance, text: str, language: str) -> None:
    """Keep what was said and what was understood, for checking speech recognition."""
    folder.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S")
    with wave.open(str(folder / f"{name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.clip(utterance.audio, -1, 1) * 32767).astype(np.int16).tobytes())
    (folder / f"{name}.txt").write_text(f"{language}\t{text}\n", encoding="utf-8")


class Assistant:
    def __init__(self, settings: Settings, bus: EventBus, tools: list[Tool], record_dir: Path | None = None) -> None:
        self._bus = bus
        self._record_dir = record_dir
        self._narration: asyncio.Queue | None = None  # the current answer's sentence queue
        self.state = State.LOADING
        self._turn: asyncio.Task | None = None
        log.info("Loading models...")
        self._mic = Microphone(settings)
        # Speech recognition and the voices load side by side to shorten startup.
        with ThreadPoolExecutor(max_workers=2) as pool:
            stt = pool.submit(create_speech_to_text, settings)
            tts = pool.submit(TextToSpeech, settings)
            self._stt, self._tts = stt.result(), tts.result()
        # "el" / "en": listen and answer only in that language; "auto": per sentence.
        self.language = settings.language
        self._stt.language = settings.language
        self._llm = create_conversation(settings, [*tools, self._language_tool()])
        self._speaker = Speaker(settings, on_level=lambda level: bus.publish(AudioLevel(level)))

    def _set_state(self, state: State) -> None:
        if state != self.state:
            self.state = state
            self._bus.publish(StateChanged(state))

    @property
    def _busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    def interrupt(self) -> None:
        """Stop the current answer (thinking, speaking or a browser action) right away."""
        if self._busy:
            log.info("Interrupted")
            self._speaker.stop()
            self._turn.cancel()

    def set_language(self, language: str) -> None:
        """The language switch: from now on listen and answer only in this language."""
        if language not in ("el", "en", "auto") or language == self.language:
            return
        self.language = language
        self._stt.language = language
        log.info("Language switched to %s", language)
        self._bus.publish(LanguageChanged(language))

    def _language_tool(self) -> Tool:
        async def set_language(args: dict) -> ToolResult:
            language = args["language"]
            self.set_language(language)
            name = LANGUAGE_NAMES[language]
            return ToolResult(f"The language switch is now set to {name}: from now on you hear and speak only {name}. "
                              f"Continue in {name}.")

        return Tool(
            "set_language",
            "Switch the conversation language when the person asks for it (\"can we speak English?\", "
            "\"μίλα ελληνικά\"). After switching you only hear and speak that language until it is switched again.",
            {"type": "object", "properties": {"language": {"type": "string", "enum": ["el", "en"]}},
             "required": ["language"], "additionalProperties": False},
            set_language,
        )

    async def _listen_for_requests(self) -> None:
        """The stop button and the language switch."""
        events = self._bus.subscribe()
        self._bus.publish(LanguageChanged(self.language))  # show the switch's starting position
        while True:
            match await events.get():
                case InterruptRequested():
                    self.interrupt()
                case LanguageRequested(language=language):
                    self.set_language(language)

    async def run(self) -> None:
        """Listen on the microphone forever."""
        self._mic.start()
        self._set_state(State.IDLE)
        log.info("Ready - start talking.")
        interrupts = asyncio.create_task(self._listen_for_requests())
        try:
            async for event in self._mic.events():
                if self._busy:
                    continue  # mic is paused while we answer; interruptions come in step 5
                match event:
                    case SpeechStarted():
                        self._set_state(State.LISTENING)
                    case SpeechDiscarded():
                        self._set_state(State.IDLE)
                    case Utterance():
                        # Run the turn as its own task so the mic loop never blocks on it,
                        # and so the stop button can cancel it.
                        self._turn = asyncio.create_task(self._voice_turn(event))
        finally:
            interrupts.cancel()
            if self._turn:
                self._turn.cancel()
            self._mic.stop()
            self._speaker.close()

    async def run_text(self) -> None:
        """Type instead of talking (for testing the LLM + voice without the microphone)."""
        self._set_state(State.IDLE)
        log.info("Ready - type a message (empty line to quit).")
        interrupts = asyncio.create_task(self._listen_for_requests())
        try:
            while text := (await _read_line("> ")).strip():
                language = self.language if self.language != "auto" else text_language(text, fallback="en")
                self._bus.publish(UserSaid(text, language))
                self._turn = asyncio.create_task(self._respond(text, language))
                await asyncio.wait([self._turn])  # doesn't raise when the turn is interrupted
                if not self._turn.cancelled() and self._turn.exception():
                    log.error("Turn failed", exc_info=self._turn.exception())
                self._set_state(State.IDLE)
        finally:
            interrupts.cancel()
            self._speaker.close()

    async def _voice_turn(self, utterance: Utterance) -> None:
        self._mic.paused = True
        self._set_state(State.THINKING)
        try:
            started = time.perf_counter()
            text, language = await self._stt.transcribe(utterance.audio)
            log.debug("STT %.2fs for %.1fs of audio", time.perf_counter() - started, utterance.seconds)
            if self._record_dir:
                await asyncio.to_thread(_save_recording, self._record_dir, utterance, text, language)
            if not text:
                return
            self._bus.publish(UserSaid(text, language))
            await self._respond(text, language)
        except Exception:
            log.exception("Turn failed")
        finally:
            await asyncio.sleep(_ECHO_TAIL_S)
            self._set_state(State.IDLE)
            self._mic.paused = False

    async def _respond(self, text: str, language: str) -> None:
        """LLM -> Piper -> speaker as a 3-stage pipeline: sentence 1 plays while 2 is generated."""
        self._set_state(State.THINKING)
        # Items: (sentence, language, future set once it has been spoken, or None).
        sentences: asyncio.Queue[tuple | None] = asyncio.Queue()
        audio: asyncio.Queue[tuple | None] = asyncio.Queue(maxsize=2)
        started = time.perf_counter()

        async def think() -> None:
            said_something = False
            try:
                async for sentence in self._llm.reply(text, language):
                    if not said_something:
                        log.debug("First sentence from the LLM after %.2fs", time.perf_counter() - started)
                    await sentences.put((sentence, language, None))
                    said_something = True
                if not said_something:
                    await sentences.put((_NO_ANSWER[language], language, None))
            except Exception as exc:
                log.error("LLM request failed: %s", exc)
                if not said_something:
                    await sentences.put((_TROUBLE[language], language, None))
            finally:
                await sentences.put(None)

        async def synthesize() -> None:
            while (item := await sentences.get()) is not None:
                sentence, sentence_language, done = item
                samples, sample_rate = await self._tts.synthesize(sentence, sentence_language)
                await audio.put((sentence, samples, sample_rate, done))
            await audio.put(None)

        async def speak() -> None:
            first = True
            item = await audio.get()
            while item is not None:
                sentence, samples, sample_rate, done = item
                if first:
                    log.debug("First audio after %.2fs", time.perf_counter() - started)
                    first = False
                self._set_state(State.SPEAKING)
                self._bus.publish(AssistantSaid(sentence))  # now, so captions match the voice
                try:
                    await self._speaker.play(samples, sample_rate)
                except asyncio.CancelledError:
                    self._speaker.stop()
                    raise
                if done is not None and not done.done():
                    done.set_result(None)
                try:
                    item = await asyncio.wait_for(audio.get(), timeout=0.5)
                except TimeoutError:
                    # Nothing more to say for now but still working (a page loading, waiting for
                    # approval...): look like it, instead of a closed mouth stuck on "speaking".
                    self._set_state(State.THINKING)
                    item = await audio.get()

        self._narration = sentences
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(think())
                tasks.create_task(synthesize())
                tasks.create_task(speak())
        finally:
            self._narration = None

    async def narrate(self, text: str, language: str) -> None:
        """Say a line in the middle of an answer (e.g. a demo step) and return once it has been
        spoken, so what is shown stays in step with what is said. Lines queue up behind
        whatever the assistant was already saying."""
        queue = self._narration
        if queue is None:
            return
        done = asyncio.get_running_loop().create_future()
        await queue.put((text, language, done))
        await done
