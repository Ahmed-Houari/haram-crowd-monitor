// node shot.mjs <html file> <png out> [width] [height] : screenshot with the installed Chrome, waits for the map
import { chromium } from "playwright";
const [, , file, out, w = "1440", h = "1080"] = process.argv;
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.log("pageerror:", e.message));
await page.goto("file:///" + file.replace(/\\/g, "/"), { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(6000);
await page.screenshot({ path: out, fullPage: true });
await browser.close();
console.log("saved", out);
