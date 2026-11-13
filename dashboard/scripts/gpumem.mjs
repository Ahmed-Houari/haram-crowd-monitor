// GPU memory of the dashboard in a real-GPU Chrome (D3D11), next to the collector, from Windows' GPU counters.
// node dashboard/scripts/gpumem.mjs [url] [--wait=ms]
// Prints, after the page settles: dedicated VRAM of this Chrome's GPU process (the dashboard tab's WebGL, video
// decode excluded: the video is not started), the crowd collector's (CUDA) and the adapter's total.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const args = process.argv.slice(2);
const flags = Object.fromEntries(args.filter((a) => a.startsWith("--")).map((a) => a.slice(2).split("=")));
const url = args.find((a) => !a.startsWith("--")) ?? "http://localhost:8000/";
const ps = (cmd) => execFileSync("powershell.exe", ["-NoProfile", "-Command", cmd], { encoding: "utf8" });
// dedicated usage per pid, MB (counter instances look like pid_1234_luid_..._phys_0)
const perPid = () => {
  const out = ps("(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | ForEach-Object { $_.InstanceName + ' ' + $_.CookedValue }");
  const m = {};
  for (const line of out.trim().split(/\r?\n/)) {
    const [inst, v] = line.trim().split(" ");
    const pid = +(/pid_(\d+)_/.exec(inst)?.[1] ?? 0);
    if (pid) m[pid] = (m[pid] ?? 0) + +v / 2 ** 20;
  }
  return m;
};
const adapterMB = () => +ps("((Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum).Sum").trim() / 2 ** 20;

const before = perPid();
const browser = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1.25 });
let lost = 0;
await page.exposeFunction("__lost", () => void lost++);
await page.addInitScript(() => {
  const orig = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (...a) {
    const ctx = orig.apply(this, a);
    if (ctx && !this.__watched) {
      this.__watched = true;
      this.addEventListener("webglcontextlost", () => window.__lost());
    }
    return ctx;
  };
});
await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(+(flags.wait ?? 20000));
const after = perPid();
// this Chrome's processes: the ones that appeared with it
const fresh = Object.keys(after).map(Number).filter((p) => !(p in before));
const chromeMB = fresh.reduce((s, p) => s + after[p], 0);
let crowdMB = null;
try {
  const pid = +readFileSync("data/.pid_crowd", "utf8").trim();
  const kids = ps(`Get-CimInstance Win32_Process -Filter "ParentProcessId=${pid}" | ForEach-Object { $_.ProcessId }`).trim().split(/\s+/).map(Number);
  crowdMB = [pid, ...kids].reduce((s, p) => s + (after[p] ?? 0), 0);
} catch { /* no collector */ }
console.log(`dashboard Chrome (GPU process): ${chromeMB.toFixed(0)} MB dedicated`);
console.log(`crowd collector (CUDA): ${crowdMB == null ? "?" : crowdMB.toFixed(0)} MB dedicated`);
console.log(`adapter total: ${adapterMB().toFixed(0)} MB dedicated (with this Chrome open)`);
console.log(`WebGL contexts lost during the run: ${lost}`);
await browser.close();
await new Promise((r) => setTimeout(r, 3000));
console.log(`adapter total after closing it: ${adapterMB().toFixed(0)} MB dedicated`);
