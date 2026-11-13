// Screenshot the dashboard with the installed Chrome (swiftshader WebGL, no browser download).
// node dashboard/scripts/shot.mjs <url> <png out> [width] [height] [--el=selector] [--reduce] [--wait=ms] [--clip=x,y,w,h] [--scale=n] [--viewport-only] [--timeout=ms]
// Prints page errors and console errors so a broken build is visible in the terminal.
import { chromium } from "playwright";

const args = process.argv.slice(2);
const flags = Object.fromEntries(args.filter((a) => a.startsWith("--")).map((a) => a.slice(2).split("=")));
const [url, out, w = "1440", h = "900"] = args.filter((a) => !a.startsWith("--"));
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({
  viewport: { width: +w, height: +h }, deviceScaleFactor: +(flags.scale ?? 1),
  reducedMotion: "reduce" in flags ? "reduce" : "no-preference",
});
page.on("pageerror", (e) => console.log("pageerror:", e.message));
page.on("console", (m) => m.type() === "error" && console.log("console:", m.text()));
await page.goto(url, { waitUntil: "networkidle", timeout: 60000 }).catch((e) => console.log("goto:", e.message));
await page.waitForTimeout(+(flags.wait ?? 6000));
const clip = flags.clip?.split(",").map(Number);
if (flags.el) { // one element only, e.g. --el=.twin
  await page.locator(flags.el).first().screenshot({ path: out, timeout: +(flags.timeout ?? 30000) });
  await browser.close();
  console.log("saved", out);
  process.exit(0);
}
await page.screenshot({ path: out, timeout: +(flags.timeout ?? 30000), fullPage: !clip && !("viewport-only" in flags), ...(clip ? { clip: { x: clip[0], y: clip[1], width: clip[2], height: clip[3] } } : {}) });
await browser.close();
console.log("saved", out);
