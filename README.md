# Intelligence – a talking, seeing AI assistant for Ridebly

**Intelligence** is a desktop AI assistant (called *Anna* by default) that appears as an
animated face on a screen. It sees the person in front of it through a camera, listens,
answers out loud in **Greek and English**, and can take over the screen to show web pages,
give narrated demos of [Ridebly](https://www.ridebly.com/) and walk customers through its
pricing.

It was built to stand in for a person at a demo stand or office: a customer walks up, talks
to the face, and Anna explains what Ridebly is, shows prices for their fleet, tours the
website, or (with approval) gives a live demo of the Ridebly app.

It runs on Windows today; the same code is meant to run on a Raspberry Pi 5 later.

Created by **Fotis Exarchos (Φώτης Έξαρχος)**, [Flarmio](https://www.ridebly.com/about-us),
Ridebly and VisionArrays teams.

---

## Contents

- [What it can do](#what-it-can-do)
- [Technology used, and what for](#technology-used-and-what-for)
- [How it works](#how-it-works)
  - [The big picture](#the-big-picture)
  - [Startup](#startup)
  - [One conversation turn, step by step](#one-conversation-turn-step-by-step)
  - [The face](#the-face)
  - [Vision](#vision)
  - [The browser and the tools](#the-browser-and-the-tools)
  - [Ridebly: knowledge, pricing, demos and approval](#ridebly-knowledge-pricing-demos-and-approval)
  - [Interrupting](#interrupting)
  - [Pronunciation](#pronunciation)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Running](#running)
- [Configuration](#configuration)
- [Writing demos](#writing-demos)
- [What data leaves the computer](#what-data-leaves-the-computer)
- [Limitations and next steps](#limitations-and-next-steps)

---

## What it can do

- **Talk** in Greek and English. It detects which language is spoken, answers in the same
  language, and switches voices sentence by sentence.
- **Show an animated face** whose eyes follow the person in front of the camera and whose
  mouth moves with its voice. The face shows whether it is idle, listening, thinking or
  speaking.
- **See**: when asked "can you see me?" or "what am I holding?", it takes a picture and
  describes it.
- **Browse the web for people**: search, open pages, scroll, click, type, highlight what it
  is talking about, open maps, draw graphs on Desmos (including 3D shapes), then return to
  its face.
- **Explain Ridebly** from a knowledge file, **show Ridebly's prices** on the live pricing page
  for the customer's fleet size, **run narrated demos** (e.g. a tour of ridebly.com), and give
  **live demos of the Ridebly app**, which need the creator's approval (a login).
- **Be interrupted** at any time with an on-screen stop button.

## Technology used, and what for

| Technology | Used for |
|---|---|
| **Python 3.12**, `asyncio` | The whole program. Everything runs concurrently on one event loop, so a slow answer never freezes the face or the camera. |
| **uv** | Python version and dependency management (`pyproject.toml`, `uv.lock`). |
| **FastAPI** + **Uvicorn** + **WebSockets** | A small local web server (port 8765) that serves the face page and pushes live events to it (state, mouth loudness, gaze, captions) and receives the stop button. |
| **HTML / CSS / JavaScript (canvas)** | The animated face, drawn on a `<canvas>`, with captions and a stop button. |
| **Playwright** + **Chromium** | One fullscreen (kiosk) browser window with two tabs: the face and the web pages the assistant shows. It also drives those pages (clicks, typing, scrolling, highlighting). |
| **Chrome DevTools Protocol** (via Playwright) | Clearing the browser's file cache at startup, so an updated face page is always used. |
| **OpenCV** | Reading frames from the webcam. |
| **MediaPipe Face Detector** (BlazeFace short-range) | Finding faces in camera frames, so the eyes can follow the closest person. |
| **sounddevice** (PortAudio) | Microphone input and speaker output. |
| **Silero VAD** (ONNX, run with **onnxruntime**) | Voice activity detection: noticing when someone starts and stops talking. The model is the one bundled with faster-whisper, so no PyTorch is needed. |
| **Groq API – Whisper large-v3** | Speech-to-text in the cloud (default). Very accurate Greek and English, ~0.5 s per sentence, free tier. |
| **NVIDIA Canary 1B v2** via **onnx-asr** (int8) | Local speech-to-text, used automatically when Groq is not configured, offline or over its limit. |
| **faster-whisper** (CTranslate2) – Whisper *tiny* | Fast local language identification (Greek or English) for the local recognizer. Whisper *small* can also be used as a full local recognizer. |
| **NVIDIA Parakeet v3** via onnx-asr | Optional, fastest local recognizer (only reliable for English-only use). |
| **Ollama** – **Gemma 4 31B (cloud)** | The default "brain" (LLM): conversation, image understanding (camera and screenshots) and tool calling. Runs on Ollama's servers on the free plan. |
| **Anthropic Claude API** (`anthropic` SDK, Claude Opus 5.5) | Optional "brain" (`LLM_PROVIDER=claude`), with server-side fallback if a request is declined. |
| **Piper TTS** | Local text-to-speech voices: `en_US-lessac-medium` (English) and `el_GR-rapunzelina-medium` (Greek). |
| **espeak-ng** phonemes (inside Piper) | Used to respell English words in Greek letters so the Greek voice pronounces them properly. |
| **DuckDuckGo** | Web search (Google blocks automated browsers). |
| **Google Maps**, **Desmos API** (`Calc`) | Showing places on a map; drawing 2D/3D graphs through Desmos's own JavaScript API. |
| **PyYAML** | Reading the scripted demo files in `demos/`. |
| **python-dotenv** | Loading all settings from `.env`. |

## How it works

### The big picture

```text
                          ┌──────────────────────── event bus (events.py) ────────────────────────┐
                          │  StateChanged · UserSaid · AssistantSaid · AudioLevel · Gaze · Interrupt │
                          └───────▲───────────────▲──────────────────▲───────────────────▲─────────┘
                                  │               │                  │                   │
 microphone ─► VAD ─► speech-to-text ─► LLM (+ tools) ─► sentences ─► text-to-speech ─► speakers
 audio_in.py   vad.py   stt.py          llm.py           app.py       tts.py, pronounce  audio_out.py
                                          │  tools.py
                                          ├─ camera picture (vision.py)
                                          ├─ web browser (browser.py) ──► content tab
                                          ├─ Ridebly pricing, app, demos (browser.py, demos.py)
                                          │
 camera ─► MediaPipe ─► gaze ─────────────┘           face server (face/server.py) ─► face tab
 vision.py                                             FastAPI + websocket            (canvas)
```

- `app.py` holds the **state machine**: `loading → idle → listening → thinking → speaking → idle`.
- Every module talks to the others through a small **publish/subscribe event bus**
  (`events.py`) instead of calling each other: the face server, the browser's orb and the
  console all just listen to events.
- Anything that blocks (speech recognition, voice synthesis, audio output, the camera) runs in
  a worker thread, so the event loop – and with it the face – stays responsive.

### Startup

1. The face server starts and Chromium opens fullscreen on the face page. The face has its
   **eyes closed** while loading.
2. The browser's file cache is cleared (cookies and logins are kept in the browser profile in
   `data/`).
3. The camera thread starts tracking faces.
4. The speech recognizer and the voices load side by side in a worker thread.
5. The microphone opens; the state becomes `idle` and the **eyes open**.

### One conversation turn, step by step

**1. Hearing that someone speaks** (`audio_in.py`, `vad.py`)
- The microphone is read at 16 kHz in 32 ms frames. If the device can't do 16 kHz, it is
  captured at its own rate and resampled.
- Each frame goes through **Silero VAD**, which gives a speech probability. Speech starts at
  probability ≥ 0.5 and ends after 0.6 s below 0.35 (hysteresis, so pauses between words don't
  end the turn).
- 300 ms of audio from *before* speech started is kept, so first syllables aren't cut off.
  Most of the trailing silence is dropped (200 ms is kept). Sounds shorter than 0.25 s are
  ignored; turns are capped at 30 s.
- While the assistant answers, the microphone is **paused** (plus 0.3 s for room echo), so it
  doesn't hear itself through the speakers.
- If the microphone is almost silent for the first 3 s after starting, a warning suggests
  choosing another device.

**2. Understanding the words** (`stt.py`)
- **Default – Groq, Whisper large-v3**: the audio is sent as WAV; Whisper detects the language
  itself. If it detects anything other than Greek or English, it is asked again in the
  conversation's language. One- or two-word replies ("ναι", "ok") keep the conversation's
  language. Segments that Whisper itself rates as probably-not-speech, and its well-known
  hallucinations ("Υπότιτλοι AUTHORWAVE", "Thanks for watching"), are dropped.
- **Fallback – NVIDIA Canary, locally**: Canary has to be told the language, and told the
  wrong one it *translates* instead of transcribing. So each sentence is transcribed **both as
  Greek and as English at the same time**, and the reading Canary is clearly more confident in
  (average token log-probability) wins – translating costs it confidence. Near-ties are decided
  by Whisper *tiny*'s language guess. The NVIDIA vocabulary has no final sigma, so `<unk>` at
  the end of a Greek word is turned back into `ς`.
- The result is the text plus its language (`el` or `en`), which travels with the turn.

**3. Thinking** (`llm.py`)
- The LLM gets a **system prompt** made of: who it is and how to speak (short, plain spoken
  sentences, no markdown, Greek or English), **everything in `knowledge/*.md`** (Ridebly, the
  team), and instructions for each group of tools it has (camera, web, Ridebly app). For
  Ollama models the detected language is repeated each turn ("Reply in Greek").
- The conversation history is kept and forgotten after 5 minutes of silence. Pictures from
  earlier turns aren't re-sent.
- **Tool calling loop**: the LLM may call tools (look, open a page, click, start a demo…).
  Each tool result is added to the conversation and the LLM continues, up to 10 rounds; on
  the last round tools are switched off so the turn always ends with something said. Text
  written before a tool call ("Let me pull that up.") is spoken while the tool runs.
- With **Claude**, the full response content (including thinking blocks) is kept in history,
  and tool results can carry images directly. With **Ollama**, camera pictures are attached
  in a user message (some models' templates garble images inside tool messages).

**4. Speaking** (`app.py`, `tts.py`, `pronounce.py`, `audio_out.py`)
- The LLM's reply is **streamed** and cut into **sentences** as it arrives. Markdown, letters
  from other alphabets (a model occasionally slips one in) and repeated sentences are removed.
- Three stages run at once: the LLM writes sentence 3 while Piper synthesizes sentence 2 and
  the speaker plays sentence 1 – the first words are heard about a second after the LLM starts.
- For each sentence the **voice** is chosen: the conversation's language whenever the sentence
  contains any of its letters (Greek sentences are full of English product terms).
- The text is rewritten for **pronunciation** (see [Pronunciation](#pronunciation)), then
  synthesized by **Piper**.
- The speaker plays it in 30 ms chunks, measuring the loudness of each chunk for the mouth.
  Loudness is reported delayed by the speaker's buffer latency, so lips and voice line up.
- The caption for a sentence is published the moment it starts playing.
- If nothing is left to say for 0.5 s but the turn isn't over (a page loading, waiting for
  approval), the face switches to *thinking*.

**5. Back to idle** – the microphone reopens and the face waits for the next person.

### The face

`assistant/face/static/` – plain HTML, CSS and JavaScript, no build step.

- Two glowing eyes and a mouth on a `<canvas>`. Every property (gaze, eye size, openness,
  mouth, colour) **eases toward a target** each frame, so changes are always smooth.
- **States**: *loading* – eyes closed; *idle* – blinking, looking at the person or around;
  *listening* – green, eyes a little bigger; *thinking* – keeps eye contact, glances up to
  the side now and then, mouth becomes a small sideways "hmm"; *speaking* – mouth opens with
  the voice's loudness.
- **Captions** under the face: what the person said (italic) while thinking, then each
  sentence the assistant speaks.
- A **language switch** (ΕΛ | EN) in the top-right corner. While it is set, the assistant only
  listens for and speaks that language: speech recognition is told the language, and replies use
  it. People can also ask ("can we speak English?"), and the assistant flips the switch itself
  (`set_language` tool). `LANGUAGE` in `.env` sets the starting position (`auto` = detect per
  sentence until someone chooses).
- A **stop button** appears while the assistant is thinking or speaking (Escape works too).
  The mouse pointer only shows while the mouse moves.
- It reconnects to the server automatically (eyes close while disconnected). The server sends
  `Cache-Control: no-store` so updated face files are always used.

### Vision

`vision.py` – a thread reads the camera ~15 times a second and runs MediaPipe face detection.
The **largest face** (the closest person) is mapped to a gaze direction, and the eyes follow
it. When nobody has been seen for 1.5 s, the eyes wander again. The latest frame is kept in
memory only; when the LLM calls `look`, that one frame is encoded as JPEG and given to the LLM.

### The browser and the tools

`browser.py` controls one Chromium window with two tabs: the **face** and a **content tab**.
The LLM gets these tools (`tools.py`):

| Tool | What it does |
|---|---|
| `look` | Takes a camera picture so the LLM can see the person. |
| `search_web` | DuckDuckGo search, results shown on screen and returned to the LLM. |
| `open_website` | Opens a page; returns its title, text, links, buttons and input fields. |
| `read_page` | Reads the page on screen again. |
| `click` | Clicks a link or button by its visible text. |
| `scroll` | Scrolls and returns the text now visible. |
| `point_at` | Scrolls to some text and outlines it, to show exactly what's being discussed. |
| `type_text` | Types key by key (works in search boxes, rich editors and math editors). |
| `press_key` | Presses Enter, Escape, arrows, etc. |
| `desmos` | Adds many expressions to Desmos at once through its JavaScript API, reports the ones Desmos rejects (and removes them) and zooms 3D drawings to fit. |
| `look_at_screen` | A screenshot of the page, e.g. to check a drawing. |
| `go_back` | Previous page. |
| `show_face` | Hides the page and shows the face again. |
| `set_language` | Flips the language switch when the person asks for another language. |
| `show_ridebly_pricing` | Sets the ridebly.com price calculator for the customer's fleet and highlights the price. |
| `open_ridebly_app` | Opens the Ridebly app; tells the LLM when approval is needed. |
| `wait_for_approval` | Waits until someone logs in to the app (that's the approval). |
| `start_demo` | Runs a scripted, narrated demo from `demos/`. |

Around the tools:
- A small **orb** with the assistant's eyes is added to every web page. It glows by state, is a
  stop button while the assistant talks, and brings the face back when clicked otherwise.
- **Cookie banners** are dismissed automatically (rejecting where the site allows it).
- Pages that try to open **new tabs** are shown in the content tab instead.
- **Downloads are blocked**, only `http(s)` addresses open, and the prompt limits what may be
  shown in public (business-appropriate content, no logins, payments or personal data).
- After 90 s without conversation, the face comes back automatically.

### Ridebly: knowledge, pricing, demos and approval

- **Knowledge** – `knowledge/ridebly.md` (what Ridebly is, plans, pages) and
  `knowledge/about-us.md` (who made the assistant, Flarmio) are part of the system prompt.
  The assistant is told to say only what these files say and to suggest booking a demo
  otherwise.
- **Pricing** – `show_ridebly_pricing` opens ridebly.com/en/pricing or /el/pricing and clicks
  the calculator's + / − buttons while the customer watches. Transport plans come in steps of
  N vehicles with 2N users, so it picks the smallest plan covering both numbers; car rental is
  priced per vehicle. The highlighted price is read back from the page, so quoted prices always
  match the live site; above the calculator's limit it says a custom plan is needed.
- **Scripted demos** – YAML files in `demos/` (see [Writing demos](#writing-demos)). The
  runner (`demos.py`) performs each step while the assistant says that step's line in the
  customer's language. Lines go through the normal voice pipeline (after anything already
  being said), and the next step starts once both the action and the line are done.
- **Live app demo and approval** – a live demo of the Ridebly app needs the creator's
  approval, which is given by **logging in on the screen**. When asked, the assistant opens the
  app (`RIDEBLY_APP_URL`); if the login page appears, it tells the customer that approval is
  needed and waits (`wait_for_approval`, up to 5 minutes) until someone logs in, then carries
  on. The assistant never logs in itself and stores no password; the login lives in the browser
  profile. On a test server (`RIDEBLY_APP_TEST_SERVER=true`) it may create example bookings
  with made-up details; otherwise it only looks and explains.

### Interrupting

The face's stop button, the Escape key, or the orb on a web page send an `InterruptRequested`
event. The assistant cancels the running turn: speech stops within a few tens of
milliseconds, any browser action or demo in progress stops, and it is ready to listen again.

### Pronunciation

`pronounce.py` rewrites each sentence for the voice that will speak it (captions keep the
original text):
- **Fixed spellings** for words the voices get wrong, e.g. *Ridebly* → «Ράιντμπλι» /
  "Ridebli", *booking engine* → «μπούκινγκ έντζιν», *rent a car* → «ρέντακαρ».
- **Any other English word in a Greek sentence** is respelled in Greek letters from its
  English pronunciation (espeak-ng phonemes mapped to Greek spelling, with the stress accent),
  because the Greek voice would otherwise read it with sounds it never learned.
- **Greek words in an English sentence** are written in Latin letters (the English voice would
  spell them out letter by letter).
- **Prices and times** are written as people say them: «€19,99» → «19 ευρώ και 99 λεπτά»,
  "09:00-23:00" → «9 έως 23» / "9 am to 11 pm".

## Project layout

```text
assistant/
  __main__.py      Entry point: starts everything, command-line options, --login
  app.py           The assistant: state machine, conversation turns, speaking pipeline, interrupts
  events.py        Event bus and event types
  config.py        All settings, read from .env
  audio_in.py      Microphone capture and turning the stream into utterances
  audio_devices.py Picking microphones/speakers by name
  vad.py           Silero voice activity detection on onnxruntime
  stt.py           Speech-to-text: Groq, Canary, Parakeet, Whisper, language detection
  language.py      Greek/English helpers
  llm.py           Conversation with Ollama or Claude, tools loop, system prompt
  tools.py         The tools the LLM can use (camera, web, Ridebly, demos)
  browser.py       Playwright browser: face tab, content tab, page actions, orb, pricing
  demos.py         Loading and running scripted demos
  tts.py           Piper voices
  pronounce.py     Pronunciation rewriting for the voices
  audio_out.py     Speaker output and mouth loudness
  vision.py        Camera and face tracking
  face/server.py   FastAPI server for the face page and its websocket
  face/static/     The face page (HTML, CSS, JavaScript)
knowledge/         What the assistant knows (Markdown, loaded into the system prompt)
demos/             Scripted demos (YAML)
.env.example       Every setting, with explanations
```

Not in the repository (created locally): `.env` (your settings and keys), `models/`
(downloaded models), `data/` (browser profile with logins, optional recordings).

## Setup

Windows 10/11 with a microphone, speakers and (optionally) a webcam.

1. Install [uv](https://docs.astral.sh/uv/) (`pip install uv`). uv installs its own Python 3.12.
2. Install the dependencies and the browser:

   ```sh
   uv sync
   uv run playwright install chromium
   ```

3. Download the voices and the face detection model:

   ```sh
   uv run python -m piper.download_voices --download-dir models/piper en_US-lessac-medium el_GR-rapunzelina-medium
   curl -L -o models/mediapipe/blaze_face_short_range.tflite --create-dirs https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
   ```

   The local speech models (Canary ~1 GB, Whisper tiny) download themselves on first run.
4. Copy `.env.example` to `.env` and fill in:
   - **LLM**: install [Ollama](https://ollama.com), run `ollama signin` and
     `ollama pull gemma4:31b-cloud` (free plan). Or set `LLM_PROVIDER=claude` and
     `ANTHROPIC_API_KEY` (paid API, from platform.claude.com).
   - **Speech recognition**: a free API key from [console.groq.com](https://console.groq.com)
     in `GROQ_API_KEY` (strongly recommended for Greek). Without it, the local recognizer is used.
   - **Microphone**: `uv run assistant --list-devices`; set `AUDIO_INPUT_DEVICE` to part of the
     microphone's name (e.g. `Microphone Array`) if the default isn't right.
5. Optional, for live app demos: `uv run assistant --login`, log in to the Ridebly app in the
   window that opens, close it.

## Running

```sh
uv run assistant              # fullscreen face, camera, microphone
uv run assistant -v           # also print state changes, language decisions and timings
uv run assistant --text       # type instead of speaking (for testing)
uv run assistant --record     # save each utterance and what was understood to data/recordings
uv run assistant --no-browser # no window; open http://127.0.0.1:8765 yourself
uv run assistant --login      # log in to the Ridebly app once (approval for live demos)
uv run assistant --list-devices
```

**Alt+F4** closes the window and the assistant. Set `FACE_FULLSCREEN=false` for a normal
window while developing.

## Configuration

All settings live in `.env`; `.env.example` lists every one with explanations. The main ones:

| Setting | Default | Meaning |
|---|---|---|
| `ASSISTANT_NAME` | Anna | The assistant's name. |
| `LLM_PROVIDER` | ollama | `ollama` or `claude`. |
| `OLLAMA_MODEL` | gemma4:31b-cloud | Ollama model; must support tools (and vision for the camera). |
| `LLM_MODEL`, `LLM_EFFORT` | claude-opus-5-5, low | Claude model and effort. |
| `STT_ENGINE` | groq | `groq`, or local `canary`, `parakeet`, `whisper`. |
| `STT_FALLBACK` | canary | Local recognizer when Groq can't be used. |
| `LANGUAGE` | auto | Starting position of the language switch: `el`, `en`, or `auto` (detect per sentence). |
| `AUDIO_INPUT_DEVICE`, `AUDIO_OUTPUT_DEVICE` | system default | Number or part of the device name. |
| `VAD_MIN_SILENCE_MS` | 600 | Silence that ends a turn. |
| `FACE_FULLSCREEN`, `FACE_CAPTIONS` | true, true | Kiosk window; subtitles. |
| `CAMERA_ENABLED`, `CAMERA_MIRROR`, `GAZE_GAIN` | true, false, 1.3 | Face tracking. |
| `BROWSE_IDLE_RETURN_S` | 90 | Back to the face after this much silence. |
| `RIDEBLY_APP_URL` | dev.app.ridebly.com | The app for live demos. |
| `RIDEBLY_APP_TEST_SERVER` | true | May create example data during demos. |
| `RIDEBLY_APPROVAL_WAIT_S` | 300 | How long to wait for approval (a login). |
| `KNOWLEDGE_DIR`, `DEMOS_DIR` | knowledge, demos | Where knowledge files and demos are. |

## Writing demos

Copy `demos/_template.yaml` to `demos/<name>.yaml` (files starting with `_` are ignored).
A demo has a `title`, a `description` (the LLM uses both to decide when to run it) and
`steps`. Each step does at most one thing and has an optional line to say; texts can be one
string or `{en: ..., el: ...}`:

```yaml
title: Ridebly website tour
description: How Ridebly works, the booking engine and the pricing page.
steps:
  - open: https://www.ridebly.com/how-it-works      # ridebly.com opens in the customer's language
    say:
      en: Let me show you how Ridebly works.
      el: Ας δούμε πώς λειτουργεί το Ridebly.
  - point_at: {en: STEP 1, el: ΒΗΜΑ 1}               # scroll to it and outline it
    say: {en: First, the customer books online., el: Πρώτα, ο πελάτης κάνει κράτηση online.}
```

Step types: `open`, `open_app` (a path in the Ridebly app; needs approval), `click`,
`point_at`, `scroll`, `type`, `press`, `pricing`, `wait`, `show_face`.

## What data leaves the computer

- **Speech audio** of each utterance goes to Groq (if `GROQ_API_KEY` is set); otherwise speech
  recognition is local.
- **Conversation text** goes to the LLM provider (Ollama's cloud for `-cloud` models, or
  Anthropic for Claude).
- **One camera picture** goes to the LLM only when it calls `look` (e.g. "can you see me?").
  Face tracking itself runs locally and frames are never stored.
- **Screenshots** of web pages go to the LLM only when it calls `look_at_screen`.
- Voices, voice activity detection and face detection run locally. `--record` stores
  recordings locally only.

## Limitations and next steps

- The local Piper voices sound synthetic, especially in Greek, and the Greek voice is female.
  Cloud voices (e.g. Azure neural voices) would sound far more natural.
- The microphone is closed while the assistant speaks; interrupting is done with the stop
  button, not by talking over it yet.
- The LLM occasionally announces an action without performing it; the prompt forbids this and
  `-v` shows every tool call (`Tool call: ...`).
- Live demos inside the Ridebly app are done by the LLM exploring the app; scripted app demos
  are still to be written.
- Running on a Raspberry Pi 5 is planned but not yet tested.
