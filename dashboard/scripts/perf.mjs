// Performance check of the 3D twin next to the live video, with the laptop's real GPU (not swiftshader).
// node dashboard/scripts/perf.mjs [url] [--video=seconds]
// Prints: the WebGL renderer, twin frames per second on screen / scrolled away / tab hidden, and with the video
// playing: twin fps and the video's dropped frames (getVideoPlaybackQuality inside the YouTube frame).
import { chromium } from "playwright";

const args = process.argv.slice(2);
const flags = Object.fromEntries(args.filter((a) => a.startsWith("--")).map((a) => a.slice(2).split("=")));
const url = args.find((a) => !a.startsWith("--")) ?? "http://localhost:8000/?perf";
const browser = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11", "--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on("pageerror", (e) => console.log("pageerror:", e.message));
await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(4000);
console.log("renderer:", await page.evaluate(() => {
  const gl = document.createElement("canvas").getContext("webgl2");
  const ext = gl?.getExtension("WEBGL_debug_renderer_info");
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : "unknown";
}));
const fps = async (ms = 5000) => {
  const a = await page.evaluate(() => window.__twinFrames ?? 0);
  await page.waitForTimeout(ms);
  const b = await page.evaluate(() => window.__twinFrames ?? 0);
  return ((b - a) * 1000) / ms;
};
console.log("twin fps on screen:", (await fps()).toFixed(1));
await page.setViewportSize({ width: 1440, height: 420 }); // short window: the twin can scroll fully out of view
await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
await page.waitForTimeout(800);
const off = await page.evaluate(() => document.querySelector(".twin-canvas").getBoundingClientRect().bottom < 0);
console.log(`twin fps scrolled away (${off ? "fully off screen" : "STILL ON SCREEN"}):`, (await fps()).toFixed(1));
await page.setViewportSize({ width: 1440, height: 900 });
await page.evaluate(() => window.scrollTo(0, 0));
await page.waitForTimeout(800);
await page.evaluate(() => { // what a hidden tab reports (headless pages stay "visible")
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
  document.dispatchEvent(new Event("visibilitychange"));
});
await page.waitForTimeout(500);
console.log("twin fps tab hidden:", (await fps()).toFixed(1));
await page.evaluate(() => {
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "visible" });
  document.dispatchEvent(new Event("visibilitychange"));
});
await page.waitForTimeout(500);
console.log("twin fps visible again:", (await fps(3000)).toFixed(1));

const secs = +(flags.video ?? 40);
if (secs > 0) {
  await page.click("button.yt");
  await page.waitForTimeout(8000);
  const yt = page.frames().find((f) => f.url().includes("youtube"));
  const q = () => yt?.evaluate(() => {
    const v = document.querySelector("video");
    if (!v) return null;
    const p = v.getVideoPlaybackQuality();
    return { total: p.totalVideoFrames, dropped: p.droppedVideoFrames, playing: !v.paused, w: v.videoWidth, h: v.videoHeight };
  }).catch(() => null);
  const q0 = await q();
  const f = await fps(secs * 1000);
  const q1 = await q();
  console.log("with video: twin fps", f.toFixed(1));
  if (q0 && q1) {
    const tot = q1.total - q0.total, drop = q1.dropped - q0.dropped;
    console.log(`video ${q1.w}x${q1.h} playing=${q1.playing}: ${tot} frames in ${secs} s (${(tot / secs).toFixed(1)} fps), dropped ${drop} (${((100 * drop) / Math.max(tot, 1)).toFixed(2)} %)`);
  } else console.log("video: could not read the player (", q0, q1, ")");
}
await browser.close();
