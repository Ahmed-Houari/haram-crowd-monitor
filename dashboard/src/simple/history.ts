// What past days' readings say (until a forecast model exists): the usual crowd at a prayer, and the quietest hours.
// Input: the Mataf's 5-minute people per m^2 ("all" ring, /api/mataf/series). Prayer times move ~1 min a day, so a
// past day's prayer is taken at today's Makkah clock time.
import type { MatafBucket, PrayerTime } from "../api";

const MAKKAH_MS = 3 * 3600_000; // Makkah is UTC+3 all year
const DAY = 86400_000;
const minuteOfDay = (t: number) => Math.floor(((t + MAKKAH_MS) % DAY) / 60000);
const dayOf = (t: number) => Math.floor((t + MAKKAH_MS) / DAY);

export interface Point { t: number; v: number }

export function points(series: MatafBucket[]): Point[] {
  return series.filter((b) => b.ring === "all" && b.people_m2 != null).map((b) => ({ t: Date.parse(b.bucket_utc), v: b.people_m2! }));
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN;
};

/** Usual people per m^2 at this prayer: per past day, the mean from 10 min before to 20 min after its time; the median
 * over days. null with fewer than 2 days. */
export function usualAtPrayer(pts: Point[], prayerUtc: string, now: number): { v: number; days: number } | null {
  const m0 = minuteOfDay(Date.parse(prayerUtc));
  const today = dayOf(now);
  const byDay = new Map<number, number[]>();
  for (const p of pts) {
    const d = dayOf(p.t);
    if (d >= today) continue;
    let dm = minuteOfDay(p.t) - m0;
    if (dm > 720) dm -= 1440;
    if (dm < -720) dm += 1440;
    if (dm >= -10 && dm <= 20) byDay.set(d, [...(byDay.get(d) ?? []), p.v]);
  }
  const days = [...byDay.values()].filter((v) => v.length >= 2).map((v) => v.reduce((a, b) => a + b, 0) / v.length);
  return days.length >= 2 ? { v: median(days), days: days.length } : null;
}

/** The hours are rated from these days on only (user 2026-11-03): the speed sensor raised the levels from 1 Nov. */
export const BEST_SINCE = Date.parse("2026-11-01T00:00:00+03:00");

export interface Window { start: number; v: number }

/** A window that overlaps a prayer is skipped (user 2026-11-03): the floor holds the prayer rows for about half an hour
 * from the adhan (1-3 Nov: 1.2-1.3 people/m² then, against ~2 before and after), and tawaf stops. */
const PRAYER_MIN = 30;
const FIVE = ["fajr", "dhuhr", "asr", "maghrib", "isha"];
const SLOT = 30; // minutes: windows start on the hour or the half hour (user 2026-11-03)
const SLOTS = 1440 / SLOT, WIN = 120 / SLOT;

/** The lightest two-hour windows from the median of each half hour over past days since BEST_SINCE (a half hour needs
 * at least 2 days of readings), the window the mean of its four, outside the prayers (today's adhan times, also used for tomorrow: they move about a minute a day):
 * the one still ahead today (Makkah; null when none is left) and tomorrow's, which is the lightest of the whole day.
 * null without enough history. */
export function lightestWindows(pts: Point[], now: number, times: Record<string, PrayerTime>):
    { today: Window | null; tomorrow: Window } | null {
  const today = dayOf(now);
  const perSlot: number[][] = Array.from({ length: SLOTS }, () => []);
  const daysPerSlot: Set<number>[] = Array.from({ length: SLOTS }, () => new Set());
  for (const p of pts) {
    if (p.t < BEST_SINCE || dayOf(p.t) >= today) continue;
    const s = Math.floor(minuteOfDay(p.t) / SLOT);
    perSlot[s].push(p.v);
    daysPerSlot[s].add(dayOf(p.t));
  }
  const usual = perSlot.map((v, s) => (daysPerSlot[s].size >= 2 ? median(v) : NaN));
  const adhan = FIVE.flatMap((n) => (times[n] ? [minuteOfDay(Date.parse(times[n].utc))] : []));
  const prayerIn = (s: number) => adhan.some((m) => m < s * SLOT + 120 && m + PRAYER_MIN > s * SLOT);
  const pick = (from: number, day: number): Window | null => {
    let best: { s: number; v: number } | null = null;
    for (let s = from; s + WIN <= SLOTS; s++) {
      if (prayerIn(s)) continue;
      const v = usual.slice(s, s + WIN).reduce((a, b) => a + b, 0) / WIN;
      if (Number.isFinite(v) && (!best || v < best.v)) best = { s, v };
    }
    return best && { start: day * DAY - MAKKAH_MS + best.s * SLOT * 60_000, v: best.v };
  };
  const tomorrow = pick(0, today + 1);
  return tomorrow && { today: pick(Math.floor(minuteOfDay(now) / SLOT) + 1, today), tomorrow };
}
