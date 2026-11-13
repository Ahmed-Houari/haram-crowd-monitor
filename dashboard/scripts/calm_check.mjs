// Checks of the main dashboard (2026-11-01 redesign) with the installed Chrome: the main screen, an (i) bubble open,
// the language switch, "Details" open and closed, and the old dashboard at ?ui=old. Saves screenshots and prints what
// it finds (page errors too).
// node dashboard/scripts/calm_check.mjs <base url> <out dir>
import { chromium } from "playwright";

const [base = "http://localhost:5174", out = "docs/design/calm"] = process.argv.slice(2);
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
await page.goto(`${base}/?lang=ar`, { waitUntil: "networkidle", timeout: 60000 }).catch((e) => console.log("goto:", e.message));
await page.waitForTimeout(9000);
const main = await page.evaluate(() => ({
  dir: document.documentElement.dir,
  word: document.querySelector(".now .word")?.textContent,
  people: document.querySelector(".now .people")?.textContent,
  state: document.querySelector(".state")?.textContent,
  details: document.querySelector("#calm-details") != null,
  m2OnMain: document.querySelector(".calm main")?.textContent?.includes("m²"),
  floor: [...document.querySelectorAll("img, canvas")].length,
}));
console.log("main", JSON.stringify(main));
await page.screenshot({ path: `${out}/final-ar-desktop.png` });
await page.locator(".now .info-btn").first().click();
await page.waitForTimeout(400);
console.log("info open", await page.locator(".info-pop").count(), await page.locator(".info-pop").first().textContent());
await page.screenshot({ path: `${out}/final-ar-info.png`, clip: { x: 1000, y: 60, width: 600, height: 330 } });
await page.keyboard.press("Escape");
console.log("info after Escape", await page.locator(".info-pop").count());
await page.getByRole("button", { name: "Switch to English" }).click();
await page.waitForTimeout(800);
console.log("en", JSON.stringify(await page.evaluate(() => ({ dir: document.documentElement.dir, word: document.querySelector(".now .word")?.textContent }))));
await page.getByRole("button", { name: "Details" }).click();
await page.waitForTimeout(6000);
console.log("details open", await page.locator("#calm-details").count(), "camera labels", await page.locator(".label3d:visible").count());
await page.screenshot({ path: `${out}/final-en-details.png`, fullPage: true });
await page.getByRole("button", { name: "Details" }).click();
console.log("details closed", await page.locator("#calm-details").count());
await page.goto(`${base}/?ui=old`, { waitUntil: "networkidle", timeout: 60000 }).catch(() => {});
await page.waitForTimeout(4000);
console.log("old dashboard", await page.locator("header.top .brand").count(), await page.locator(".metric, .hero").count());
await browser.close();
console.log("errors", JSON.stringify(errors));
