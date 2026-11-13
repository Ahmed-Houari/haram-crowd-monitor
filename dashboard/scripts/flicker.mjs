// Flicker check of the 3D twin on the laptop's real GPU (D3D11), with the collector running.
// node dashboard/scripts/flicker.mjs [--dist=folder] [--secs=120] [--url=http://localhost:8000/]
// --dist serves the page's own files from another build folder (the API and WebSocket stay live), to compare builds.
// Every 0.5 s: is the twin canvas blank (nearly no pixels differ from its background)? Counts WebGL context losses,
// canvas resizes and canvas elements being replaced, and prints console warnings about WebGL.
import { readFileSync, existsSync } from "node:fs";
import { extname, join } from "node:path";
import { chromium } from "playwright";

const flags = Object.fromEntries(process.argv.slice(2).filter((a) => a.startsWith("--")).map((a) => a.slice(2).split("=")));
const url = flags.url ?? "http://localhost:8000/";
const secs = +(flags.secs ?? 120);
const types = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".woff2": "font/woff2", ".woff": "font/woff",
  ".geojson": "application/geo+json", ".png": "image/png", ".svg": "image/svg+xml" };
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1.25 });
if (flags.dist) {
  await page.route((u) => !u.pathname.startsWith("/api/") && u.host.startsWith("localhost"), (route) => {
    const p = new URL(route.request().url()).pathname;
    const f = join(flags.dist, p === "/" ? "index.html" : p);
    if (!existsSync(f)) return route.continue();
    route.fulfill({ body: readFileSync(f), contentType: types[extname(f)] ?? "application/octet-stream" });
  });
}
const warn = new Map();
page.on("console", (m) => {
  const t = m.text();
  if (/webgl|context|gpu|too many/i.test(t)) warn.set(t.slice(0, 160), (warn.get(t.slice(0, 160)) ?? 0) + 1);
});
await page.addInitScript(() => {
  window.__ev = { lost: 0, restored: 0, resized: 0, replaced: 0 };
  const orig = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (...a) {
    const ctx = orig.apply(this, a);
    if (ctx && !this.__w && /webgl/.test(String(a[0]))) {
      this.__w = true;
      this.addEventListener("webglcontextlost", () => window.__ev.lost++);
      this.addEventListener("webglcontextrestored", () => window.__ev.restored++);
      let last = `${this.width}x${this.height}`;
      new ResizeObserver(() => { const s = `${this.width}x${this.height}`; if (s !== last) { window.__ev.resized++; last = s; } }).observe(this);
    }
    return ctx;
  };
  addEventListener("DOMContentLoaded", () => {
    new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => n.nodeName === "CANVAS" && n.closest?.(".twin-canvas") && window.__ev.replaced++)))
      .observe(document.body, { childList: true, subtree: true });
  });
});
await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(6000);
const base = await page.evaluate(() => ({ ...window.__ev }));
// blank = the canvas shows no model: share of bright pixels (the building, the floor) in a downscaled grab
const isBlank = async () => {
  const buf = await page.locator(".twin-canvas canvas").first().screenshot({ timeout: 5000 }).catch(() => null);
  if (!buf) return true;
  const share = await page.evaluate(async (b64) => {
    const img = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
    const c = new OffscreenCanvas(160, 160 * img.height / img.width), x = c.getContext("2d");
    x.drawImage(img, 0, 0, c.width, c.height);
    const d = x.getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 240) n++;
    return n / (d.length / 4);
  }, buf.toString("base64"));
  return share < 0.03;
};
if ("lose" in flags) { // what a GPU short of memory does: lose the twin's context, give it back after 1 s
  const ok = await page.evaluate(() => {
    const c = document.querySelector(".twin-canvas canvas");
    const gl = c.getContext("webgl2") || c.getContext("webgl");
    const ext = gl && gl.getExtension("WEBGL_lose_context");
    if (!ext) return false;
    ext.loseContext();
    setTimeout(() => ext.restoreContext(), 1000);
    return true;
  });
  const marks = [];
  for (const t of [500, 1500, 3000, 6000, 10000]) {
    await page.waitForTimeout(t - (marks.length ? [500, 1500, 3000, 6000, 10000][marks.length - 1] : 0));
    marks.push(`${t / 1000} s: ${(await isBlank()) ? "BLANK" : "drawn"}`);
  }
  console.log(`forced context loss (${ok ? "done" : "no extension"}): ${marks.join(", ")}`);
}
if ("scroll" in flags) { // scroll the twin out of view and back, ten times
  let after = 0;
  for (let i = 0; i < 10; i++) {
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(700);
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(250);
    if (await isBlank()) after++;
  }
  console.log(`scroll away and back x10: blank 0.25 s after coming back: ${after}`);
}
if ("video" in flags) { // the live broadcast playing next to the twin (hardware video decode on the same GPU)
  await page.click("button.yt").catch(() => console.log("no video button"));
  await page.waitForTimeout(5000);
}
let blank = 0, n = 0;
const t0 = Date.now();
while (Date.now() - t0 < secs * 1000) {
  n++;
  if (await isBlank()) blank++;
  await page.waitForTimeout(500);
}
const ev = await page.evaluate(() => window.__ev);
console.log(`build: ${flags.dist ?? "served"}; ${n} samples in ${secs} s; blank samples: ${blank}`);
console.log(`after load: context lost ${ev.lost - base.lost}, restored ${ev.restored - base.restored}, canvas resized ${ev.resized - base.resized}, twin canvas replaced ${ev.replaced - base.replaced}`);
console.log(`during load: lost ${base.lost}, resized ${base.resized}, replaced ${base.replaced}`);
for (const [t, c] of warn) console.log(`console x${c}: ${t}`);
await browser.close();
