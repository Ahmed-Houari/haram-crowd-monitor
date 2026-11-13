// The main dashboard since 2026-11-01 (user redesign, option A of two): the main screen holds three things and one
// small trend: the crowd now (a word and colour plus about how many people), the Mataf twin in flat level colours, the
// live video. A side column (the answer, the trend in hourly blocks, the next prayer and the quietest time, the video)
// sits next to the twin, which carries a small colour key. The road map under them (user 2026-11-03). Everything
// technical (camera chips, legends, people per m², the AI view, the prayer horizon, road numbers, alerts) sits inside
// "Details", closed by default. Arabic first, English on
// request. Display only: the readings are the same as before. The previous
// engineer dashboard (App.tsx) stays at /?ui=old.
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { ChevronDown } from "lucide-react";
import {
  connectLive, crowdLevel, fmtM2, getJSON, minutesSince, viewLabel,
  type CrowdBucket, type CrowdNow, type MatafBucket, type MatafNow, type Meta, type PrayerEvent, type Snapshot,
  type TrafficPoint, type View,
} from "../api";
import { useMedia, useNow, useReducedMotion } from "../hooks";
import { clock, LEVEL_WORD, relative, T, type Lang } from "../simple/i18n";
import type { TwinState } from "../components/Twin";
import CameraChips from "../components/CameraChips";
import LevelScale from "../components/LevelScale";
import SeePanes, { SbaVideo, YouTubeEmbed } from "../components/SeePanes";
import Horizon from "../components/Horizon";
import RoadPanel from "../components/RoadPanel";
import AlertsPanel from "../components/AlertsPanel";
import Info from "./Info";
import HourBlocks from "./HourBlocks";
import PrayerSoon from "./PrayerSoon";
import LevelKey from "../simple/LevelKey";
import ForecastLine from "../simple/ForecastLine";
import { approxPeople, fmtNum, RING_AREA_M2, W } from "./words";
import "./calm.css";

const Twin = lazy(() => import("../components/Twin"));
const RoadMap = lazy(() => import("../components/RoadMap"));

const LIVE_MIN = 15; // older than this is "not live", never the current level (user, 2026-10-30)
const CAM_STALE_MIN = 2;
const REFRESH_MS = 5 * 60_000;
const HARAM = { lat: 21.4225, lon: 39.8262 };
const QUERY = new URLSearchParams(location.search);

function storedLang(): Lang {
  const q = QUERY.get("lang"); // ?lang=en|ar (screenshots, links)
  if (q === "en" || q === "ar") return q;
  try {
    return localStorage.getItem("calm-lang") === "en" ? "en" : "ar";
  } catch {
    return "ar";
  }
}

export default function CalmApp() {
  const [lang, setLang] = useState<Lang>(storedLang);
  const [details, setDetails] = useState(() => QUERY.has("details")); // ?details: open (screenshots)
  const w = W[lang];
  const reduce = useReducedMotion();
  const mobile = useMedia("(max-width: 760px)");
  const now = useNow(15_000);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [liveNow, setLiveNow] = useState<CrowdNow[]>([]);
  const [mataf, setMataf] = useState<MatafNow | null>(null);
  const [series, setSeries] = useState<MatafBucket[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [views, setViews] = useState<View[]>([]);
  const [prayers, setPrayers] = useState<PrayerEvent[]>([]);
  const [traffic, setTraffic] = useState<TrafficPoint[]>([]);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => connectLive(
    (s) => { setSnap(s); if (s.crowd_now) setLiveNow(s.crowd_now); if (s.mataf_now !== undefined) setMataf(s.mataf_now); },
    (n, m) => { setLiveNow(n); if (m !== undefined) setMataf(m); },
    setConnected,
  ), []);
  useEffect(() => {
    getJSON<Meta>("/api/meta").then(setMeta).catch(console.error);
    const load = () => {
      getJSON<View[]>("/api/views").then(setViews).catch(console.error);
      getJSON<MatafBucket[]>("/api/mataf/series?hours=20").then(setSeries).catch(() => setSeries([]));
      getJSON<PrayerEvent[]>("/api/prayer/window?back=24&ahead=4").then(setPrayers).catch(console.error);
      getJSON<TrafficPoint[]>("/api/traffic/series?hours=24").then(setTraffic).catch(console.error);
    };
    load();
    const t = window.setInterval(load, REFRESH_MS);
    return () => window.clearInterval(t);
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    document.title = lang === "ar" ? "المطاف الآن · مرصد الحشود" : "The Mataf now · Haram Crowd Monitor";
    try { localStorage.setItem("calm-lang", lang); } catch { /* private mode: not remembered */ }
  }, [lang]);

  // the answer: the 5-minute reading while it is live
  const value = mataf ? mataf.density_5min ?? mataf.people_m2 : null;
  const age = mataf ? minutesSince(mataf.minute_utc, now) : Infinity;
  const live = value != null && age <= LIVE_MIN;
  const level = value != null ? crowdLevel(value) : null;
  const people = value != null ? approxPeople(value) : null;
  const state = !connected ? "off" : live ? "on" : "stale";

  const viewById = useMemo(() => new Map(views.map((v) => [v.id, v])), [views]);
  const twinCams = useMemo<TwinState[]>(() => views.filter((v) => v.twin).map((v) => {
    const c = liveNow.find((x) => x.view_id === v.id);
    const t = c?.minute_utc ?? mataf?.by_camera?.[v.id]?.minute_utc;
    const st = !v.has_reference ? "waiting" : !t ? "idle" : minutesSince(t, now) > CAM_STALE_MIN ? "stale" : "live";
    return { id: v.id, ...v.twin!, quality: v.quality, state: st, k: c?.level.rank ?? (c ? 0.55 : 0), selected: picked === v.id };
  }), [views, liveNow, now, mataf, picked]);
  const cameraM2 = useMemo(() => {
    const out: Record<string, { v: number; at: string }> = {};
    for (const [id, c] of Object.entries(mataf?.by_camera ?? {})) out[id] = { v: c.people_m2, at: c.minute_utc };
    for (const c of liveNow) if (c.people_m2 != null && c.people_m2_utc) out[c.view_id] ??= { v: c.people_m2, at: c.people_m2_utc };
    return out;
  }, [mataf, liveNow]);
  const aiView = picked ? viewById.get(picked) : viewById.get(snap?.on_air?.view_id ?? "") ?? views.find((v) => v.has_heatmap);

  return (
    <div className="calm">
      <header className="cbar">
        <div className="cbrand">
          <svg viewBox="0 0 40 40" aria-hidden="true">
            <g fill="none" stroke="#c8a45c" strokeWidth="1.5">
              <rect x="9" y="9" width="22" height="22" /><rect x="9" y="9" width="22" height="22" transform="rotate(45 20 20)" />
              <circle cx="20" cy="20" r="6.5" />
            </g>
            <rect x="17" y="17" width="6" height="6" fill="#c8a45c" />
          </svg>
          <h1>{w.brand}</h1>
        </div>
        <p className={`state ${state}`} role="status">
          <i aria-hidden="true" />
          <b>{state === "on" ? w.live : state === "stale" ? w.notLive : w.offline}</b>
          {mataf && <span>{w.updated(relative(lang, mataf.minute_utc, now))}</span>}
        </p>
        <div className="cctl">
          <button type="button" className="ctl lang" lang={lang === "ar" ? "en" : "ar"} aria-label={w.switchLabel}
            onClick={() => setLang(lang === "ar" ? "en" : "ar")}>{w.switchTo}</button>
          <button type="button" className={`ctl det${details ? " open" : ""}`} aria-expanded={details} aria-controls="calm-details"
            onClick={() => setDetails(!details)}>
            {w.details}<ChevronDown size={16} strokeWidth={1.8} aria-hidden="true" />
          </button>
        </div>
      </header>

      <main className="cmain">
        <div className="col side">
          <section className="now" aria-live="polite">
            <h2 className="ttl">{w.crowdNow}<Info lines={w.i.crowd} label={w.info} close={w.close} /></h2>
            {value == null ? (
              <p className="word wait">{mataf === null && !snap ? w.loading : w.noReading}</p>
            ) : live ? (
              <>
                <p className={`word lv-${level}`}>{LEVEL_WORD[lang][level!]}</p>
                <Meter level={level!} />
                <p className="people"><span className="n">{w.about} {fmtNum(lang, people!)}</span> {w.people}</p>
                <ForecastLine lang={lang} />
              </>
            ) : (
              <>
                <p className="word off">{w.notLive}</p>
                <p className="people dim">{w.lastAt(clock(lang, mataf!.minute_utc))} · {LEVEL_WORD[lang][level!]}</p>
              </>
            )}
          </section>
          <section className="trend">
            <h3 className="ttl small">{w.trend}<Info lines={w.i.trend} label={w.info} close={w.close} /></h3>
            <HourBlocks series={series} now={now} lang={lang} />
          </section>
          <PrayerSoon lang={lang} />
          <section className="video">
            <h3 className="ttl small">{w.video}<Info lines={w.i.video} label={w.info} close={w.close} /></h3>
            <LiveVideo youtubeId={meta?.youtube_id ?? null} sbaUrl={meta?.sba_hls ?? null} youtubePaused={(meta?.youtube_paused_s ?? 0) > 0}
              label={w.play} liveWord={w.live} />
          </section>
        </div>
        <section className="stage">
          <h3 className="ttl small on-stage">{w.twin}<Info lines={w.i.twin} label={w.info} close={w.close} /></h3>
          <Suspense fallback={<div className="twin-canvas"><div className="twin-fallback">{w.loading}</div></div>}>
            <Twin cams={twinCams} floor={snap?.floor ?? null} reduce={reduce} mobile={mobile} floorStyle="flat" labels={details}
              pad={mobile ? { t: 40, r: 6, b: 56, l: 6 } : { t: 40, r: 18, b: 18, l: 18 }} />
          </Suspense>
          <LevelKey lang={lang} />
        </section>
      </main>

      <section className="croads">
        <h3 className="ttl small">{w.roads}<Info lines={w.i.roads} label={w.info} close={w.close} /></h3>
        <div className="mapwrap calm-map">
          <Suspense fallback={null}>
            <RoadMap center={meta?.haram ?? HARAM} reloadKey={snap?.traffic.poll_utc ?? null} crowdM2={live ? value : null} />
          </Suspense>
        </div>
        <ul className="roadkey">
          <li><i className="free" />{T[lang].free}</li>
          <li><i className="slow" />{T[lang].slow}</li>
          <li><i className="jam" />{T[lang].jam}</li>
          <li><i className="closed" />{w.closed}</li>
        </ul>
      </section>

      {details && (
        <section id="calm-details" className="details">
          <div className="drow">
            <div className="dblock">
              <h3>{w.d.reading}</h3>
              {mataf ? (
                <p className="m2">
                  <bdi><b className="num">{fmtM2(mataf.people_m2)}</b> {w.d.perM2}</bdi> · {w.d.fiveMin} <bdi className="num">{fmtM2(mataf.density_5min)}</bdi>
                  {" "}· {w.d.inner} <bdi className="num">{mataf.rings.inner != null ? fmtM2(mataf.rings.inner) : "–"}</bdi>
                  {" "}· {w.d.outer} <bdi className="num">{mataf.rings.outer != null ? fmtM2(mataf.rings.outer) : "–"}</bdi>
                  {mataf.far ? <> · {w.d.far} <bdi className="num">{fmtM2(mataf.far.people_m2)}</bdi> ({mataf.far.confidence})</> : null}
                  {mataf.model ? <> · <bdi>{mataf.model}</bdi></> : null}
                  {" "}· <bdi>{mataf.cameras.map((c) => viewLabel(viewById.get(c), c)).join(", ")}</bdi>
                </p>
              ) : <p className="m2">{w.noReading}</p>}
              <p className="m2 mute">{w.d.peopleRule(fmtNum(lang, RING_AREA_M2))}</p>
              <div dir="ltr" lang="en"><LevelScale /></div>
              <ul className="twin-key">
                <li><i className="k-flat" />{w.d.keyFlat}</li>
                <li><i className="k-hatch" />{w.d.keyHatch}</li>
                <li><i className="k-none" />{w.d.keyNone}</li>
              </ul>
            </div>
            <div className="dblock" dir="ltr" lang="en">
              <h3 dir={lang === "ar" ? "rtl" : "ltr"} lang={lang}>{w.d.cameras}</h3>
              <CameraChips views={views} crowd={liveNow} m2={cameraM2} heroId={picked} onPick={setPicked} now={now} staleMin={CAM_STALE_MIN} />
            </div>
          </div>
          <div className="drow two" dir="ltr" lang="en">
            <SeePanes only="ai" youtubeId={null} view={aiView} heatTs={null} sample={false} now={now} onAir={snap?.on_air?.view_id ?? null} />
            <Horizon prayers={prayers} series={series.filter((b) => b.ring === "all").map<CrowdBucket>((b) => ({ bucket_utc: b.bucket_utc,
              view_id: "mataf", zone: "all", source: "youtube", density: b.people_m2, count: null, n_frames: b.n_frames }))}
              cameraLabel="Mataf" now={now} mataf />
          </div>
          <div className="drow one" dir="ltr" lang="en">
            <RoadPanel latest={snap?.traffic ?? null} series={traffic} prayers={prayers}>
              <AlertsPanel alerts={snap?.alerts ?? null} sample={false} />
            </RoadPanel>
          </div>
        </section>
      )}
      <footer className="cfoot">{w.privacy}</footer>
    </div>
  );
}

/** Where the level sits on the four: four bars in the level colours, the current one solid and taller. No numbers. */
function Meter({ level }: { level: string }) {
  return (
    <div className="meter" aria-hidden="true">
      {["quiet", "busy", "crowded", "packed"].map((l) => <i key={l} className={`${l}${l === level ? " on" : ""}`} />)}
    </div>
  );
}

/** The live broadcast, loaded on a click: YouTube (no cookies), SBA's stream when YouTube is paused or blocked. */
function LiveVideo({ youtubeId, sbaUrl, youtubePaused, label, liveWord }: { youtubeId: string | null; sbaUrl: string | null;
  youtubePaused: boolean; label: string; liveWord: string }) {
  const [play, setPlay] = useState(false);
  const [useSba, setUseSba] = useState(false);
  const sba = !!sbaUrl && (useSba || youtubePaused || !youtubeId);
  return (
    <div className="frame cframe">
      {play && sba ? <SbaVideo url={sbaUrl!} />
        : play && youtubeId ? <YouTubeEmbed id={youtubeId} onBlocked={sbaUrl ? () => setUseSba(true) : undefined} />
        : (
          <button type="button" className="yt" onClick={() => setPlay(true)} aria-label={label}>
            <span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" /></svg></span>
          </button>
        )}
      {!play && <span className="badge">{liveWord}</span>}
    </div>
  );
}
