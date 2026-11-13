import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  ago,
  connectLive,
  crowdColour,
  crowdLevel,
  fmtM2,
  getJSON,
  minutesSince,
  viewLabel,
  type CrowdBucket,
  type CrowdNow,
  type MatafBucket,
  type MatafNow,
  type Meta,
  type PrayerEvent,
  type Snapshot,
  type TrafficPoint,
  type View,
} from "./api";
import { DEMO, demoAlerts, demoForecast, demoNow, demoSeries } from "./demo";
import { useMedia, useNow, useReducedMotion } from "./hooks";
import Header from "./components/Header";
import LevelScale from "./components/LevelScale";
import HeroNumber, { MatafHero } from "./components/HeroNumber";
import CameraChips from "./components/CameraChips";
import SeePanes from "./components/SeePanes";
import Horizon from "./components/Horizon";
import RoadPanel from "./components/RoadPanel";
import AlertsPanel from "./components/AlertsPanel";
import type { TwinState } from "./components/Twin";

const Twin = lazy(() => import("./components/Twin"));
const RoadMap = lazy(() => import("./components/RoadMap"));

const REFRESH_MS = 5 * 60 * 1000;
const STALE_MIN = 2; // an older reading is shown as "at the last reading", not "now" (2026-10-26: readings every 1-2 s)
const MAP_MIN = 20; // the road map still colours the Haram with a reading this old, its age shown
const HARAM = { lat: 21.4225, lon: 39.8262 };
const AI_VIEW = new URLSearchParams(location.search).get("ai"); // ?ai=<view id>: the AI view of one camera (review)

export default function App() {
  const reduce = useReducedMotion();
  const mobile = useMedia("(max-width: 760px)");
  const now = useNow(15000);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [liveNow, setLiveNow] = useState<CrowdNow[]>([]);
  const [mataf, setMataf] = useState<MatafNow | null>(null);
  const [matafSeries, setMatafSeries] = useState<MatafBucket[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [views, setViews] = useState<View[]>([]);
  const [series, setSeries] = useState<CrowdBucket[]>([]);
  const [traffic, setTraffic] = useState<TrafficPoint[]>([]);
  const [prayers, setPrayers] = useState<PrayerEvent[]>([]);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(
    () =>
      connectLive(
        (s) => {
          setSnap(s);
          if (s.crowd_now) setLiveNow(s.crowd_now);
          if (s.mataf_now !== undefined) setMataf(s.mataf_now);
        },
        (now, m) => {
          setLiveNow(now);
          if (m !== undefined) setMataf(m);
        },
        setConnected,
      ),
    [],
  );

  useEffect(() => {
    getJSON<Meta>("/api/meta").then(setMeta).catch(console.error);
    const load = () => {
      getJSON<View[]>("/api/views").then(setViews).catch(console.error);
      getJSON<PrayerEvent[]>("/api/prayer/window?back=24&ahead=4").then(setPrayers).catch(console.error);
      getJSON<TrafficPoint[]>("/api/traffic/series?hours=24").then(setTraffic).catch(console.error);
      if (!DEMO) getJSON<CrowdBucket[]>("/api/crowd/series?hours=20").then(setSeries).catch(console.error);
      // the view-independent series; an API from before 2026-10-24 has no such route: keep the per-camera view
      if (!DEMO) getJSON<MatafBucket[]>("/api/mataf/series?hours=20").then(setMatafSeries).catch(() => setMatafSeries([]));
    };
    load();
    const t = window.setInterval(load, REFRESH_MS);
    return () => window.clearInterval(t);
  }, []);

  // ?demo: sample crowd numbers; a new "reading" each minute so the tween and pulse can be seen
  const demoMinute = Math.floor(now / 60000);
  const crowdNow = useMemo(
    () => (DEMO ? (prayers.length ? demoNow(prayers, demoMinute * 60000 + 30000) : []) : liveNow),
    [liveNow, prayers, demoMinute],
  );
  const crowdSeries = useMemo(() => (DEMO ? demoSeries(prayers, 20) : series), [series, prayers]);
  // forecast + alerts: live from the API; ?demo shows labelled samples (live alerts appear once the rules are ready)
  const forecast = useMemo(() => (DEMO ? (prayers.length ? demoForecast(prayers, demoMinute * 60000 + 30000) : null)
    : snap?.forecast ?? null), [snap, prayers, demoMinute]);
  const alerts = useMemo(() => (DEMO ? demoAlerts(demoMinute * 60000) : snap?.alerts ?? null), [snap, demoMinute]);

  const viewById = useMemo(() => new Map(views.map((v) => [v.id, v])), [views]);
  // hero = a fresh reading from a good camera first, then the most recent one; or the camera the user picked
  const hero = useMemo(() => {
    const p = picked && crowdNow.find((c) => c.view_id === picked);
    if (p) return p;
    const score = (c: CrowdNow) =>
      (minutesSince(c.minute_utc, now) > STALE_MIN ? 2e6 : 0) + (viewById.get(c.view_id)?.quality === "ok" ? 0 : 1e6)
      + minutesSince(c.minute_utc, now);
    return [...crowdNow].sort((a, b) => score(a) - score(b))[0] ?? null;
  }, [crowdNow, picked, viewById, now]);
  const heroView = hero ? viewById.get(hero.view_id) : undefined;
  // the Mataf (people per m², all cameras) leads unless a camera was picked; when old it says "at the last reading"
  const showMataf = !DEMO && !picked && mataf != null;
  const matafAge = showMataf ? minutesSince(mataf.minute_utc, now) : Infinity;
  const matafFresh = showMataf && matafAge <= MAP_MIN; // the map colours only a recent one (its age shown when > 1 min)
  const matafBuckets = useMemo<CrowdBucket[]>(
    () => matafSeries.filter((b) => b.ring === "all").map((b) => ({ bucket_utc: b.bucket_utc, view_id: "mataf", zone: "all",
      source: "youtube", density: b.people_m2, count: null, n_frames: b.n_frames })),
    [matafSeries],
  );

  const twinCams = useMemo<TwinState[]>(
    () =>
      views.filter((v) => v.twin).map((v) => {
        const c = crowdNow.find((x) => x.view_id === v.id);
        // a camera only the m2 metric uses (no per-camera zone) is live by its own people per m²
        const own = mataf?.by_camera?.[v.id];
        const t = c?.minute_utc ?? own?.minute_utc;
        const state = !v.has_reference ? "waiting" : !t ? "idle" : minutesSince(t, now) > STALE_MIN ? "stale" : "live";
        return { id: v.id, ...v.twin!, quality: v.quality, state, k: c?.level.rank ?? (c ? 0.55 : 0), selected: hero?.view_id === v.id };
      }),
    [views, crowdNow, now, hero, mataf],
  );

  const heroSeries = useMemo(
    () => (hero ? crowdSeries.filter((b) => b.view_id === hero.view_id && b.zone === hero.zone && b.source === hero.source) : []),
    [crowdSeries, hero],
  );
  // "What the AI sees": the picked camera, else the camera with the newest heatmap (so it changes as often as any
  // camera is counted); ?demo shows the sample density map of the low camera (docs/design/mockups/heatmap-cam0.png)
  const heatTimes = snap?.heatmaps ?? {};
  const freshest = Object.entries(heatTimes).filter(([, t]) => t).sort((a, b) => b[1]!.localeCompare(a[1]!))[0]?.[0];
  // in sync with the live video: the camera on air when its counted frame is under a minute old
  const onAir = snap?.on_air?.view_id ?? null;
  const onAirFresh = onAir != null && heatTimes[onAir] != null && minutesSince(heatTimes[onAir]!, now) < 1 ? onAir : null;
  // the current shot has no known camera (live shot service, method c): the AI view shows it, labelled approximate
  const shot = snap?.shot ?? null;
  const shotNow = shot != null && shot.age_s != null && shot.age_s <= 40 && shot.method === "approximate" && !picked && !AI_VIEW;
  const heatmapView = DEMO ? "cam0_mataf_low" : shotNow ? "shot"
    : AI_VIEW ?? picked ?? onAirFresh ?? freshest ?? hero?.view_id ?? views.find((v) => v.has_heatmap)?.id ?? null;
  // each camera's own people per m² (view-independent metric), for the chips and the AI view
  const cameraM2 = useMemo(() => {
    const out: Record<string, { v: number; at: string }> = {};
    for (const [id, c] of Object.entries(mataf?.by_camera ?? {})) out[id] = { v: c.people_m2, at: c.minute_utc };
    for (const c of crowdNow) if (c.people_m2 != null && c.people_m2_utc) out[c.view_id] ??= { v: c.people_m2, at: c.people_m2_utc };
    return out;
  }, [mataf, crowdNow]);

  return (
    <>
      {DEMO && (
        <div className="demo-bar" role="note">
          <b>Sample data</b> Crowd numbers on this page are generated for a design preview{DEMO === "new" ? " (first day, still calibrating)" : " (as after a week of data)"}. Status, roads and prayer times are real.
        </div>
      )}
      <Header status={snap?.status ?? null} connected={connected} prayer={snap?.prayer ?? null} now={now} />
      <main>
        <section className="hero">
          <div className="twin">
            <Suspense fallback={<div className="twin-canvas"><div className="twin-fallback">Loading the Mataf twin…</div></div>}>
              <Twin cams={twinCams} floor={DEMO ? null : snap?.floor ?? null} reduce={reduce} mobile={mobile} avoid=".metric" />
            </Suspense>
            {showMataf ? <MatafHero mataf={mataf} views={views} now={now} reduce={reduce} staleMin={STALE_MIN} />
              : <HeroNumber hero={hero} view={heroView} m2={hero ? cameraM2[hero.view_id] ?? null : null} status={snap?.status ?? null}
                now={now} reduce={reduce} staleMin={STALE_MIN} />}
            <div className="tag3d"><span className="ar" lang="ar">المطاف</span>Mataf twin · schematic</div>
            <CameraChips views={views} crowd={crowdNow} m2={cameraM2} heroId={hero?.view_id ?? null} onPick={setPicked} now={now} staleMin={STALE_MIN} />
            <div className="scale" aria-hidden="true">
              Crowd level on the Mataf floor
              <LevelScale />
              <small><i />all cameras, newest view of each spot</small>
              <small><i className="avg" />fading: older, up to {snap?.floor?.max_age_min ?? 10} min</small>
              <small><i className="none" />grey: no camera saw it in {snap?.floor?.max_age_min ?? 10} min</small>
              <small><i className="hidden" />light hatch: estimated (behind the Kaaba or under the broadcast text: from the floor beside it at the same distance)</small>
              <small>during tawaf, the walking parts of each ring show that ring's reading from the walking speed</small>
              <small><i className="line" />inner / outer / far Mataf: 1–10 m, 10–25 m (headline), 25 m to the arcade (approximate; drawn squeezed)</small>
            </div>
          </div>
          <SeePanes youtubeId={meta?.youtube_id ?? null} sbaUrl={meta?.sba_hls ?? null} youtubePaused={(meta?.youtube_paused_s ?? 0) > 0} view={heatmapView ? viewById.get(heatmapView) : undefined}
            shot={heatmapView === "shot" ? shot : null}
            heatTs={DEMO || !heatmapView ? null : heatmapView === "shot" ? shot?.ts ?? null : heatTimes[heatmapView] ?? null} sample={DEMO != null} now={now}
            onAir={DEMO ? undefined : onAir} onAirLabel={onAir ? viewLabel(viewById.get(onAir), onAir).toLowerCase() : null} />
        </section>
        {/* people per m² whenever the Mataf series exists, also while a camera is picked */}
        {!DEMO && (showMataf || matafBuckets.length > 0) ? (
          <Horizon prayers={prayers} series={matafBuckets} cameraLabel="Mataf" now={now} mataf
            forecast={forecast?.available ? forecast.cameras?.find((c) => c.view_id === "mataf" && c.zone === "all") ?? null : null} />
        ) : (
          <Horizon prayers={prayers} series={heroSeries} cameraLabel={hero ? viewLabel(heroView, hero.view_id) : "low camera"} now={now}
            forecast={forecast?.available && hero ? forecast.cameras?.find((c) => c.view_id === hero.view_id) ?? null : null} />
        )}
        <section className="lower">
          <div className="mapwrap">
            <Suspense fallback={null}>
              <RoadMap center={meta?.haram ?? HARAM} reloadKey={snap?.traffic.poll_utc ?? null} crowdM2={matafFresh ? mataf.people_m2 : null} />
            </Suspense>
            <div className="overlay maptitle">
              <b>Roads around the Haram</b><span className="ar" lang="ar">الحركة المرورية</span>
            </div>
            <div className="overlay legend" aria-hidden="true">
              <span><i style={{ background: "#3fa56a" }} />Clear · <bdi lang="ar">سالك</bdi></span>
              <span><i style={{ background: "#f28a1e" }} />Slow · <bdi lang="ar">بطيء</bdi></span>
              <span><i style={{ background: "#e8384f" }} />Traffic jam · <bdi lang="ar">زحمة</bdi></span>
              <span><i className="dash" />tunnel, no data</span>
              <span><i className="haram" style={matafFresh ? { background: crowdColour(mataf.people_m2) } : undefined} />Haram: {matafFresh ? <>{fmtM2(mataf.people_m2)} people/m² · {crowdLevel(mataf.people_m2)}{matafAge > 1 ? ` (${ago(mataf.minute_utc, now)})` : ""}</> : "no recent reading"}</span>
              <span><i className="plaza" />pedestrian zone</span>
              <span className="mute">rings 1 / 3 km</span>
            </div>
          </div>
          <RoadPanel latest={snap?.traffic ?? null} series={traffic} prayers={prayers}>
            <AlertsPanel alerts={alerts} sample={DEMO != null} />
          </RoadPanel>
        </section>
      </main>
      <footer className="foot">
        <span>Counting only: no faces, no tracking. Frames are processed on the edge PC and deleted; only numbers and a small density heatmap leave it.</span>
        <span>Times: Makkah (UTC+3){DEMO ? " · crowd values on this page are samples" : ""}</span>
      </footer>
    </>
  );
}
