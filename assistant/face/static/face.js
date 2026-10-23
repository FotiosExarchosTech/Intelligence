// Animated face: glowing eyes + mouth on a canvas, driven by websocket messages from the
// assistant ({type: "state" | "level" | "gaze" | "say" | "config"}).
//
// Every visual property moves smoothly toward a target value each frame, so state changes
// never jump.

const canvas = document.getElementById("face");
const ctx = canvas.getContext("2d");
const caption = document.getElementById("caption");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

const COLORS = {
  idle: [90, 209, 255],      // calm cyan
  listening: [110, 240, 160], // green: "I hear you"
  thinking: [90, 209, 255],
  speaking: [90, 209, 255],
  demo: [90, 209, 255],
};

// What the assistant tells us.
const input = { state: "idle", level: 0, gaze: null, connected: false, captions: true };

// What is currently drawn (eased toward the targets).
const face = {
  color: [...COLORS.idle],
  gazeX: 0, gazeY: 0,
  mouth: 0,      // 0 = closed (smile), 1 = wide open
  scale: 1,      // eyes grow a little while listening
  squint: 0,     // eyes narrow a little while thinking
  sleep: 1,      // 1 = eyes shut (not connected to the assistant)
  ponder: 0,     // 1 = mouth is a small sideways "hmm" line
};

// Exponential easing that behaves the same at any frame rate.
const approach = (current, target, rate, dt) => target + (current - target) * Math.exp(-rate * dt);

// ---------- canvas sizing (sharp on high-DPI screens) ----------

let width = 0, height = 0;
function resize() {
  const dpr = devicePixelRatio || 1;
  width = innerWidth;
  height = innerHeight;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
addEventListener("resize", resize);
resize();

// ---------- blinking ----------

let nextBlink = performance.now() + 1500;
let blinkStart = -1;
const BLINK_MS = 140;

function blinkAmount(now) {
  if (blinkStart < 0 && now >= nextBlink) blinkStart = now;
  if (blinkStart < 0) return 0;
  const p = (now - blinkStart) / BLINK_MS; // 0..1 closing, 1..2 opening
  if (p >= 2) {
    blinkStart = -1;
    const doubleBlink = Math.random() < 0.2;
    nextBlink = now + (doubleBlink ? 120 : 2200 + Math.random() * 4000);
    return 0;
  }
  return p < 1 ? p : 2 - p;
}

// ---------- idle eye movement (when nobody is tracked) ----------

let wander = { x: 0, y: 0 };
let nextWander = 0;

function wanderGaze(now) {
  if (now >= nextWander) {
    const range = reducedMotion.matches ? 0.15 : 0.55;
    wander = Math.random() < 0.4
      ? { x: 0, y: 0 } // look straight ahead fairly often
      : { x: (Math.random() * 2 - 1) * range, y: (Math.random() * 2 - 1) * range * 0.5 };
    nextWander = now + 1500 + Math.random() * 3500;
  }
  return wander;
}

// ---------- pondering (thinking): glance up to the side now and then, then back ----------

let pondering = false;
let glanceUntil = 0;
let nextGlance = 0;
let glanceSide = 1;

function ponderGaze(now, gaze) {
  if (!pondering) {
    pondering = true;
    nextGlance = now + 300; // first "hmm, let me think" glance right away
  }
  if (now >= nextGlance) {
    glanceSide = Math.random() < 0.5 ? -1 : 1;
    glanceUntil = now + 900;
    nextGlance = glanceUntil + (reducedMotion.matches ? 5000 : 1800 + Math.random() * 1700);
  }
  return now < glanceUntil ? { x: 0.85 * glanceSide, y: -0.95 } : gaze;
}

// ---------- animation loop ----------

let last = performance.now();

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  const state = input.state;

  let gaze = input.gaze ?? wanderGaze(now);
  if (state === "thinking") gaze = ponderGaze(now, gaze);
  else pondering = false;
  face.gazeX = approach(face.gazeX, gaze.x, 9, dt);
  face.gazeY = approach(face.gazeY, gaze.y, 9, dt);

  const color = COLORS[state] ?? COLORS.idle;
  face.color = face.color.map((c, i) => approach(c, color[i], 5, dt));
  face.scale = approach(face.scale, state === "listening" ? 1.1 : 1, 7, dt);
  face.squint = approach(face.squint, state === "thinking" ? 0.12 : 0, 6, dt);
  face.sleep = approach(face.sleep, input.connected && state !== "loading" ? 0 : 1, 3, dt);
  face.ponder = approach(face.ponder, state === "thinking" ? 1 : 0, 8, dt);

  // Mouth opens fast and closes a bit slower, like real speech.
  const mouthTarget = state === "speaking" ? input.level : 0;
  face.mouth = approach(face.mouth, mouthTarget, mouthTarget > face.mouth ? 28 : 12, dt);

  draw(now);
  requestAnimationFrame(frame);
}

function draw(now) {
  ctx.clearRect(0, 0, width, height);

  const unit = Math.min(width, height);
  const cx = width / 2;
  const cy = height / 2 - unit * 0.04;
  const [r, g, b] = face.color.map(Math.round);
  const color = `rgb(${r} ${g} ${b})`;

  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = unit * 0.05;

  // Eyes: two rounded bars that move together with the gaze.
  const lookX = face.gazeX * unit * 0.12;
  const lookY = face.gazeY * unit * 0.08;
  const eyeW = unit * 0.15 * face.scale;
  const openness = Math.max(0.07, (1 - blinkAmount(now)) * (1 - face.squint) * (1 - face.sleep));
  const eyeH = unit * 0.25 * face.scale * openness;
  const eyeGap = unit * 0.19;
  const eyeY = cy - unit * 0.07 + lookY;
  for (const side of [-1, 1]) {
    const x = cx + side * eyeGap + lookX;
    ctx.beginPath();
    ctx.roundRect(x - eyeW / 2, eyeY - eyeH / 2, eyeW, eyeH, Math.min(eyeW, eyeH) / 2.2);
    ctx.fill();
  }

  // Mouth: a gentle smile when closed (turning into a short, slightly crooked "hmm" line
  // while thinking), a rounded opening when speaking.
  const p = face.ponder;
  const mouthX = cx + lookX * 0.5 + unit * 0.035 * p;
  const mouthY = cy + unit * 0.2 + lookY * 0.4;
  const mouthW = unit * (0.15 + 0.05 * face.mouth) * (1 - 0.45 * p);
  if (face.mouth < 0.03) {
    const tilt = unit * 0.012 * p;
    ctx.lineWidth = unit * 0.016;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(mouthX - mouthW / 2, mouthY + tilt);
    ctx.quadraticCurveTo(mouthX, mouthY + unit * 0.035 * (1 - face.sleep) * (1 - p), mouthX + mouthW / 2, mouthY - tilt);
    ctx.stroke();
  } else {
    const mouthH = unit * (0.016 + 0.13 * face.mouth);
    ctx.beginPath();
    ctx.roundRect(mouthX - mouthW / 2, mouthY - mouthH / 2, mouthW, mouthH, Math.min(mouthW, mouthH) / 2);
    ctx.fill();
  }
}

requestAnimationFrame(frame);

// ---------- captions ----------

let captionTimer = 0;

function showCaption(text, who) {
  if (!input.captions) return;
  clearTimeout(captionTimer);
  caption.textContent = text;
  caption.classList.toggle("user", who === "user");
  caption.classList.add("visible");
}

function hideCaption(delayMs) {
  clearTimeout(captionTimer);
  captionTimer = setTimeout(() => caption.classList.remove("visible"), delayMs);
}

// ---------- stop button (and Escape key) ----------

const stopButton = document.getElementById("stop");
let socket = null;

function updateBusy() {
  const busy = input.connected && (input.state === "thinking" || input.state === "speaking");
  document.body.classList.toggle("busy", busy);
}

function interrupt() {
  if (socket?.readyState === WebSocket.OPEN && document.body.classList.contains("busy")) {
    socket.send(JSON.stringify({ type: "interrupt" }));
  }
}

stopButton.addEventListener("click", interrupt);
addEventListener("keydown", (event) => {
  if (event.key === "Escape") interrupt();
});

// ---------- language switch ----------

const languageButtons = [...document.querySelectorAll("#language button")];

function showLanguage(language) {
  for (const button of languageButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.language === language));
  }
}

for (const button of languageButtons) {
  button.addEventListener("click", () => {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "language", language: button.dataset.language }));
    }
  });
}

// The mouse pointer shows while the mouse moves and hides after a few quiet seconds.
let cursorTimer = 0;
addEventListener("pointermove", () => {
  document.body.classList.remove("hide-cursor");
  clearTimeout(cursorTimer);
  cursorTimer = setTimeout(() => document.body.classList.add("hide-cursor"), 3000);
});
document.body.classList.add("hide-cursor");

// ---------- connection to the assistant ----------

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  socket = ws;

  ws.addEventListener("open", () => { input.connected = true; });

  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
      case "config":
        input.captions = msg.captions;
        break;
      case "state":
        input.state = msg.state;
        if (msg.state === "idle") hideCaption(2500);
        updateBusy();
        break;
      case "level":
        input.level = msg.level;
        break;
      case "gaze":
        input.gaze = msg.x === null ? null : { x: msg.x, y: msg.y };
        break;
      case "say":
        showCaption(msg.text, msg.who);
        break;
      case "language":
        showLanguage(msg.language);
        break;
    }
  });

  // The assistant restarted or stopped: close the eyes and keep retrying.
  ws.addEventListener("close", () => {
    input.connected = false;
    input.state = "idle";
    input.level = 0;
    hideCaption(0);
    updateBusy();
    setTimeout(connect, 1000);
  });
}

connect();
