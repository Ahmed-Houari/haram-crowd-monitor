// The next prayer with its countdown and the best time to go, as on /simple (user 2026-11-03): the same words and
// the same estimate from past days' readings (simple/history.ts). Its own one-second clock, so the countdown does not
// redraw the rest of the dashboard. Display only: the readings are unchanged.
import { useEffect, useMemo, useState } from "react";
import { CROWD_LEVELS, crowdLevel, getJSON, type CrowdLevel, type MatafBucket, type PrayerToday } from "../api";
import { clock, countdown, LEVEL_WORD, PRAYER, T, type Lang } from "../simple/i18n";
import { points, usualAtPrayer } from "../simple/history";
import BestTime from "../simple/BestTime";

const PRAYERS = ["fajr", "dhuhr", "asr", "maghrib", "isha"];
const levelRgb = (l: CrowdLevel) => `rgb(${CROWD_LEVELS.find((x) => x.name === l)!.rgb.join(",")})`;

function usePoll<V>(path: string, every: number): V | null {
  const [v, setV] = useState<V | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => getJSON<V>(path).then((d) => alive && setV(d)).catch(() => {});
    load();
    const id = window.setInterval(load, every);
    return () => { alive = false; window.clearInterval(id); };
  }, [path, every]);
  return v;
}

export default function PrayerSoon({ lang }: { lang: Lang }) {
  const t = T[lang];
  const [now, setNow] = useState(Date.now());
  const prayer = usePoll<PrayerToday>("/api/prayer/today", 60_000);
  const series = usePoll<MatafBucket[]>("/api/mataf/series?hours=336", 30 * 60_000);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const pts = useMemo(() => points(series ?? []), [series]);
  const next = useMemo(() => {
    const times = prayer?.times;
    if (!times) return null;
    const ahead = PRAYERS.map((n) => ({ name: n, utc: times[n]?.utc })).filter((p) => p.utc && Date.parse(p.utc) > now);
    if (ahead.length) return ahead[0] as { name: string; utc: string };
    // after Isha: tomorrow's Fajr, at today's clock time (it moves about a minute a day)
    return times.fajr ? { name: "fajr", utc: new Date(Date.parse(times.fajr.utc) + 86400_000).toISOString() } : null;
  }, [prayer, now]);
  const tenMin = Math.floor(now / 600_000);
  const expected = useMemo(() => (next ? usualAtPrayer(pts, next.utc, now) : null), [pts, next?.utc, tenMin]);

  return (
    <section className="soon">
      <div className="cell">
        <h3 className="ttl small">{t.nextPrayer}</h3>
        {next ? (
          <>
            <p className="pline"><span className="pname">{PRAYER[lang][next.name]}</span><span className="ptime">{clock(lang, next.utc)}</span></p>
            <p className="count">{t.inTime} <b>{countdown(lang, Date.parse(next.utc) - now)}</b></p>
            {expected ? (
              <p className="lvl"><i style={{ background: levelRgb(crowdLevel(expected.v)) }} />{t.expected}: {LEVEL_WORD[lang][crowdLevel(expected.v)]}</p>
            ) : <p className="sub">{t.notEnough}</p>}
          </>
        ) : <p className="sub">{prayer ? t.notEnough : t.loading}</p>}
      </div>
      <div className="cell">
        <h3 className="ttl small">{t.bestTitle}</h3>
        <BestTime pts={series ? pts : null} times={prayer?.times} now={now} lang={lang} />
      </div>
    </section>
  );
}
