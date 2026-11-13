// Sample crowd data for ?demo=1 (a week of history: levels + usual-for-this-time) and ?demo=new (first day:
// calibrating). Only crowd numbers are faked; status, roads and prayer times stay real. The page shows a
// "Sample data" banner whenever this is on.
import type { Alerts, CrowdBucket, CrowdNow, Forecast, Level, PrayerEvent, Usual } from "./api";

export type DemoMode = "week" | "new" | null;

export const DEMO: DemoMode = (() => {
  const q = new URLSearchParams(location.search).get("demo");
  return q === "new" ? "new" : q != null ? "week" : null;
})();

const SCALE: Record<string, number> = { cam0_mataf_low: 1, cam5_roof: 1.32 };
const LAG_MIN: Record<string, number> = { cam0_mataf_low: 3, cam5_roof: 11 };
const VISIBLE: Record<string, number> = { cam0_mataf_low: 1, cam5_roof: 0.64 }; // roof camera zoomed in: part of the zone
const AREA = 66; // zone area in units of 100 x 100 px: people in the whole zone = density x AREA

/** Density in a camera zone: a night baseline plus a swell around each prayer, with slow wobble. */
function sampleAt(t: number, prayers: PrayerEvent[], scale: number): number {
  return peopleAt(t, prayers, scale) / AREA;
}

function peopleAt(t: number, prayers: PrayerEvent[], scale: number): number {
  let v = 1020;
  for (const p of prayers) {
    if (p.name === "sunrise") continue;
    const dt = (t - Date.parse(p.utc)) / 60000;
    v += 640 * Math.exp(-((dt + 20) ** 2) / (2 * 42 ** 2));
  }
  v += Math.sin(t / 7.3e5) * 55 + Math.sin(t / 2.9e6) * 85;
  return Math.max(150, v * scale);
}

const minute = (t: number) => Math.floor(t / 60000) * 60000;
const bursty = (t: number) => Math.floor(t / 60000) % 6 < 2; // camera on air 2 min out of 6

function level(v: number, scale: number): Level {
  const base = { pcts: [40, 80] as [number, number], history_days: 28, need_days: 3, need_buckets: 36 };
  if (DEMO === "new") return { state: "calibrating", days: 0.6, buckets: 9, ...base };
  const low = (1010 * scale) / AREA, high = (1480 * scale) / AREA;
  return { state: v < low ? "quiet" : v >= high ? "crowded" : "busy", days: 7.4, buckets: 820, low, high,
    rank: 0.62, ...base };
}

function usual(v: number): Usual {
  if (DEMO === "new") return { state: "waiting", days: 0.6, need_days: 7 };
  const u = v / 1.12;
  return { state: "ok", days: 7.4, need_days: 7, usual: u, pct: 12, days_at_time: 6 };
}

export function demoNow(prayers: PrayerEvent[], now = Date.now()): CrowdNow[] {
  return Object.keys(SCALE).map((id) => {
    const s = SCALE[id];
    const last = minute(now - LAG_MIN[id] * 60000);
    const spark: [string, number][] = [];
    for (let t = last - 120 * 60000; t <= last; t += 60000)
      if (bursty(t) || t === last) spark.push([new Date(t).toISOString(), sampleAt(t, prayers, s)]);
    const v = sampleAt(last, prayers, s);
    const ref = sampleAt(last - 30 * 60000, prayers, s);
    const vis = VISIBLE[id];
    return {
      view_id: id, zone: "mataf_ground", source: "youtube", minute_utc: new Date(last).toISOString(),
      density: v, visible: vis, count: vis >= 1 ? Math.round(v * AREA) : null, count_visible: Math.round(v * AREA * vis),
      density_5min: v,
      trend: { ref_utc: new Date(last - 30 * 60000).toISOString(), ref_density: ref, pct: (100 * (v - ref)) / ref },
      spark, level: level(v, s), usual: usual(v),
    };
  });
}

export function demoSeries(prayers: PrayerEvent[], hours: number, now = Date.now()): CrowdBucket[] {
  const out: CrowdBucket[] = [];
  const end = minute(now - 3 * 60000);
  for (let t = end - hours * 3600000; t <= end; t += 5 * 60000) {
    if (Math.floor(t / 300000) % 7 === 3) continue; // a few empty buckets: the camera was off air
    for (const id of Object.keys(SCALE))
      out.push({ bucket_utc: new Date(t).toISOString(), view_id: id, zone: "mataf_ground", source: "youtube",
        density: sampleAt(t, prayers, SCALE[id]), visible: VISIBLE[id], count: null, n_frames: 6 });
  }
  return out;
}

/** Sample forecast: the same curve 15 / 30 / 60 min ahead, crowded threshold = the sample "high" level. */
export function demoForecast(prayers: PrayerEvent[], now = Date.now()): Forecast {
  return {
    available: true, trained_utc: new Date(now - 6 * 3600000).toISOString(),
    cameras: Object.keys(SCALE).map((id) => {
      const s = SCALE[id], issued = minute(now - LAG_MIN[id] * 60000), thr = (1480 * s) / AREA;
      const predictions = [15, 30, 60].map((h) => {
        const d = sampleAt(issued + h * 60000, prayers, s) * 1.04;
        return { h, utc: new Date(issued + h * 60000).toISOString(), density: d };
      });
      return { view_id: id, zone: "mataf_ground", issued_utc: new Date(issued).toISOString(), threshold_density: thr,
        predictions, crowded_within_30: predictions.some((p) => p.h <= 30 && p.density >= thr) };
    }),
  };
}

/** Sample alerts: one open forecast alert, a closed crowd alert and a closed road alert. */
export function demoAlerts(now = Date.now()): Alerts {
  const iso = (minAgo: number) => new Date(minute(now - minAgo * 60000)).toISOString();
  const at = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Riyadh", hour: "2-digit", minute: "2-digit" })
    .format(new Date(now + 24 * 60000));
  return {
    cooldown_min: 30,
    rules: {
      crowd_now: { ready: true, note: "a camera's latest reading is in its top 20 %" },
      crowd_forecast: { ready: true, note: "the forecast reaches crowded within 30 min" },
      road: { ready: true, note: "over 25 % of road length within 1 km congested on 2 polls in a row" },
    },
    items: [
      { id: "f1", kind: "crowd_forecast", key: "cam0_mataf_low", severity: "watch", place: "Low camera zone",
        message: `Low camera zone expected crowded by ${at}: forecast density 23.3 per 100x100 px (crowded from 22.4)`,
        start_utc: iso(6), last_on_utc: iso(1), end_utc: null, value: 23.3, peak: 23.3, threshold: 22.4, reopened: 0, active: true },
      { id: "c1", kind: "crowd_now", key: "cam5_roof", severity: "high", place: "Roof camera zone",
        message: "Roof camera zone was crowded: density 30.5 people per 100x100 px (crowded from 29.6 for this camera)",
        start_utc: iso(130), last_on_utc: iso(102), end_utc: iso(92), value: 29.7, peak: 30.5, threshold: 29.6, reopened: 1, active: false },
      { id: "r1", kind: "road", key: "r1km", severity: "watch", place: "Roads within 1 km",
        message: "Roads within 1 km: 31 % of road length congested (over 25 % for 10 min)",
        start_utc: iso(250), last_on_utc: iso(215), end_utc: iso(205), value: 31, peak: 34, threshold: 25, reopened: 0, active: false },
    ],
  };
}
