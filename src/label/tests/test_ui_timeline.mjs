#!/usr/bin/env node

import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class ClassList {
  constructor() { this.values = new Set(); }
  add(value) { this.values.add(value); }
  remove(value) { this.values.delete(value); }
  contains(value) { return this.values.has(value); }
}

class Element {
  constructor(id) {
    this.id = id;
    this.tagName = "DIV";
    this.style = {};
    this.dataset = {};
    this.classList = new ClassList();
    this.attributes = new Map();
    this.listeners = new Map();
    this.value = "";
    this.textContent = "";
    this.innerHTML = "";
    this.clientWidth = 1000;
    this.currentTime = 0;
    this.paused = true;
    this.playCalls = 0;
    this.capturedPointer = null;
  }

  addEventListener(type, callback) { this.listeners.set(type, callback); }
  emit(type, event = {}) {
    event.type = type;
    event.target ??= this;
    this.listeners.get(type)?.(event);
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name); }
  getBoundingClientRect() { return { left: 100, width: this.clientWidth }; }
  setPointerCapture(id) { this.capturedPointer = id; }
  hasPointerCapture(id) { return this.capturedPointer === id; }
  releasePointerCapture(id) { if (this.capturedPointer === id) this.capturedPointer = null; }
  focus() {}
  pause() { this.paused = true; }
  play() { this.paused = false; this.playCalls += 1; return Promise.resolve(); }
}

const html = fs.readFileSync(new URL("../ui/index.html", import.meta.url), "utf8");
const script = html.split("<script>", 2)[1].split("</script>", 1)[0]
  .replace(/\nloadFamilies\(\);\s*$/, "");
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(match => match[1]);
const elements = new Map(ids.map(id => [id, new Element(id)]));
elements.get("v").tagName = "VIDEO";

const context = vm.createContext({
  console,
  document: { querySelector: selector => elements.get(selector.slice(1)) },
  addEventListener() {},
  confirm: () => true,
  prompt: () => null,
  fetch: async () => { throw new Error("unexpected fetch in timeline unit test"); },
  setTimeout,
  clearTimeout,
});
context.window = context;
vm.runInContext(script, context, { filename: "src/label/ui/index.html" });
vm.runInContext("S.total = 101; S.fps = 25", context);

const track = elements.get("track");
const video = elements.get("v");
const pointer = (pointerId, clientX) => ({
  button: 0,
  pointerId,
  clientX,
  preventDefault() {},
});

// 1000 px 对应 0..100 帧。拖动必须量化到整数帧，并覆盖完整首尾边界。
video.paused = false;
track.emit("pointerdown", pointer(7, 350)); // 25%
assert.equal(video.currentTime, 1, "pointerdown should seek to frame 25");
assert.equal(video.paused, true, "scrubbing should pause playback");
assert.equal(track.classList.contains("scrubbing"), true);
assert.equal(track.capturedPointer, 7);

track.emit("pointermove", pointer(7, 855)); // 75.5% -> frame 76
assert.equal(video.currentTime, 76 / 25);
assert.equal(track.getAttribute("aria-valuenow"), "76");

track.emit("pointerup", pointer(7, 1200)); // right edge -> final frame
assert.equal(video.currentTime, 4);
assert.equal(elements.get("frame").textContent, 100);
assert.equal(track.getAttribute("aria-valuemax"), "100");
assert.equal(track.getAttribute("aria-valuenow"), "100");
assert.equal(track.classList.contains("scrubbing"), false);
assert.equal(track.capturedPointer, null);
assert.equal(video.paused, false, "playback state should be restored after scrubbing");
assert.equal(video.playCalls, 1);

// 原本暂停的视频在点击/拖动后必须继续暂停；越过左边界要夹到第 0 帧。
video.paused = true;
track.emit("pointerdown", pointer(8, -500));
track.emit("pointerup", pointer(8, -500));
assert.equal(video.currentTime, 0);
assert.equal(video.paused, true);
assert.equal(video.playCalls, 1);

console.log("timeline drag: PASS (frame quantization, bounds, pointer capture, playback restore)");
