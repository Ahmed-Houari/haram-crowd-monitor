// The visitor page (/simple, 2026-10-30): one answer about the Mataf now, the next prayer, the quietest hours, the
// roads, and how fresh it is. Arabic first, English on request. No numbers or units for crowd: words and colours.
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { CROWD_LEVELS, crowdLevel, getJSON, type CrowdLevel, type MatafBucket, type MatafNow, type PrayerToday } from "../api";
import MatafRing from "./MatafRing";
import LevelKey from "./LevelKey";
import type { RoadsData } from "./SimpleMap";
import { clock, countdown, LEVEL_WORD, PRAYER, relative, T, type Lang } from "./i18n";
import { points, usualAtPrayer } from "./history";
import BestTime from "./BestTime";
import ForecastLine from "./ForecastLine";

const SimpleMap = lazy(() => import("./SimpleMap"));

const STALE_MIN = 15; // older than this is "not live" (user, 2026-10-30)
const ROADS_STALE_MIN = 30;
const HARAM = { lat: 21.4225, lon: 39.8262 };
const DEMO = new URLSearchParams(location.search).has("demo"); // labelled sample answer (dashboard's ?demo convention)
const PRAYERS = ["fajr", "dhuhr", "asr", "maghrib", "isha"];

const levelRgb = (l: CrowdLevel) => `rgb(${CROWD_LEVELS.find((x) => x.name === l)!.rgb.join(",")})`;

function storedLang(): Lang {
  try {
    return localStorage.getItem("simple-lang") === "en" ? "en" : "ar";
  } catch {
    return "ar";
  }
}

type Roads = RoadsData & { poll_utc: string | null };

function usePoll<T>(path: string, every: number): T | null {
  const [v, setV] = useState<T | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => getJSON<T>(path).then((d) => alive && setV(d)).catch(() => {});
    load();
    const id = setInterval(load, every);
    return () => { alive = false; clearInterval(id); };
  }, [path, every]);
  return v;
}

export default function SimpleApp() {
  const [lang, setLang] = useState<Lang>(storedLang);
  const [now, setNow] = useState(Date.now());
  const t = T[lang];
  const mataf = usePoll<MatafNow>("/api/mataf/now", 60_000);
  const prayer = usePoll<PrayerToday>("/api/prayer/today", 60_000);
  const series = usePoll<MatafBucket[]>("/api/mataf/series?hours=336", 30 * 60_000);
  const roads = usePoll<Roads>("/api/traffic/segments", 60_000);
  const meta = usePoll<{ haram?: { lat: number; lon: number } }>("/api/meta", 60 * 60_000);

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    document.title = t.title;
    try { localStorage.setItem("simple-lang", lang); } catch { /* private mode: language just isn't remembered */ }
  }, [lang, t.title]);

  // the answer: the Mataf's 5-minute people per m^2 while fresh
  const reading = DEMO ? { v: 1.6, at: new Date(now - 2 * 60_000).toISOString() }
    : mataf ? { v: mataf.density_5min ?? mataf.people_m2, at: mataf.minute_utc } : null;
  const live = reading != null && (now - Date.parse(reading.at)) / 60000 <= STALE_MIN;
  const level = reading ? crowdLevel(reading.v) : null;

  const pts = useMemo(() => points(series ?? []), [series]);
  const next = useMemo(() => {
    const times = prayer?.times;
    if (!times) return null;
    const ahead = PRAYERS.map((n) => ({ name: n, utc: times[n]?.utc })).filter((p) => p.utc && Date.parse(p.utc) > now);
    if (ahead.length) return ahead[0] as { name: string; utc: string };
    // after Isha: tomorrow's Fajr, at today's clock time (it moves about a minute a day)
    return times.fajr ? { name: "fajr", utc: new Date(Date.parse(times.fajr.utc) + 86400_000).toISOString() } : null;
  }, [prayer, now]);
  const expected = useMemo(() => (next ? usualAtPrayer(pts, next.utc, now) : null), [pts, next?.utc, Math.floor(now / 600_000)]);
  const roadsOld = roads?.poll_utc ? (now - Date.parse(roads.poll_utc)) / 60000 > ROADS_STALE_MIN : roads != null;

  const word = level ? LEVEL_WORD[lang][level] : "";
  return (
    <div className="page">
      {DEMO && <p className="sample">{t.sample}</p>}
      <header className="bar">
        <h1>{t.title}</h1>
        <button type="button" className="lang" lang={lang === "ar" ? "en" : "ar"} aria-label={t.switchLabel}
          onClick={() => setLang(lang === "ar" ? "en" : "ar")}>{t.switchTo}</button>
      </header>

      <section className="answer" aria-live="polite">
        <MatafRing colour={live && level ? levelRgb(level) : null} label={live ? `${t.title}: ${word}` : t.noLive} />
        {reading == null ? (
          <p className="word wait">{mataf === null ? t.loading : t.never}</p>
        ) : live ? (
          <>
            <p className={`word lv-${level}`}>{word}</p>
            {!DEMO && <ForecastLine lang={lang} />}
          </>
        ) : (
          <>
            <p className="word off">{t.noLive}</p>
            <p className="sub">{t.lastReading(clock(lang, reading.at))}</p>
          </>
        )}
        <LevelKey lang={lang} estimate={false} />
      </section>

      {next && (
        <section className="row">
          <h2>{t.nextPrayer}</h2>
          <div className="line">
            <span className="big">{PRAYER[lang][next.name]}</span>
            <span className="time">{clock(lang, next.utc)}</span>
          </div>
          <div className="line">
            <span className="count">{t.inTime} <b className="num">{countdown(lang, Date.parse(next.utc) - now)}</b></span>
            {expected ? (
              <span className="chip"><i style={{ background: levelRgb(crowdLevel(expected.v)) }} />{t.expected}: {LEVEL_WORD[lang][crowdLevel(expected.v)]}</span>
            ) : (
              <span className="sub">{t.notEnough}</span>
            )}
          </div>
        </section>
      )}

      <section className="row">
        <h2>{t.bestTitle}</h2>
        <BestTime pts={series ? pts : null} times={prayer?.times} now={now} lang={lang} />
      </section>

      <section className="row roads">
        <h2>{t.roads}</h2>
        <Suspense fallback={<div className="map" />}>
          <SimpleMap center={meta?.haram ?? HARAM} roads={roads} lang={lang} label={t.roads} />
        </Suspense>
        <ul className="legend">
          <li><i className="free" />{t.free}</li>
          <li><i className="slow" />{t.slow}</li>
          <li><i className="jam" />{t.jam}</li>
        </ul>
        {roadsOld && <p className="sub">{t.roadsOld}</p>}
      </section>

      <footer>
        <p className="updated">
          <span className={`dot${live ? " on" : ""}`} aria-hidden="true" />
          {reading ? t.updated(relative(lang, reading.at, now)) : t.never}
        </p>
        <p className="privacy">{t.privacy}</p>
      </footer>
    </div>
  );
}
