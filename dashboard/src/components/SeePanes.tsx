import { useEffect, useRef, useState } from "react";
import LevelScale from "./LevelScale";
import { ago, fmtInt, fmtM2, FULL_VISIBLE, getJSON, M2_HELP, makkahTime, minutesSince, viewLabel, type AiView, type Shot, type View } from "../api";

interface Props {
  youtubeId: string | null;
  /** SBA's HLS stream of the same broadcast: played when YouTube is paused or its embed does not start */
  sbaUrl?: string | null;
  youtubePaused?: boolean;
  view: View | undefined;
  /** time of the camera's newest heatmap: the AI view reloads when it changes */
  heatTs: string | null;
  sample: boolean;
  now: number;
  /** the camera on air now (null: another shot; undefined: unknown, e.g. the sample page) */
  onAir?: string | null;
  onAirLabel?: string | null;
  /** the current shot when it has no known camera: the AI view shows it, labelled approximate */
  shot?: Shot | null;
  /** only one of the two panes (calm dashboard 2026-11-01: the AI view inside "Details") */
  only?: "ai";
}

const clamp = (pct: number) => Math.min(80, Math.max(20, pct)); // keep the count label inside the frame
const OLD_MIN = 15; // older than this: the AI view says so plainly
const REFRESH_MS = 60_000; // also reload every minute (the camera's people per m² changes between heatmaps)
// the YouTube embed must report "playing" within this long, else it is taken as blocked (e.g. "Sign in to confirm
// you're not a bot") and SBA's stream plays instead
const YT_START_MS = 15_000;

/** "What the camera sees" (YouTube, loaded on click; SBA's stream when YouTube is blocked or paused) next to "What the
 * AI sees" (density over building outlines). */
export default function SeePanes({ youtubeId, sbaUrl, youtubePaused, view, heatTs, sample, now, onAir, onAirLabel, shot, only }: Props) {
  const [play, setPlay] = useState(false);
  const [useSba, setUseSba] = useState(false);
  const [why, setWhy] = useState<string | null>(null);
  const sba = !!sbaUrl && (useSba || youtubePaused || !youtubeId);
  const [ai, setAi] = useState<AiView | null>(null);
  const [tick, setTick] = useState(0);
  const label = shot ? "current shot" : view ? viewLabel(view, view.id).toLowerCase() : "";
  const viewId = shot ? "shot" : view?.id ?? null;

  useEffect(() => {
    const t = window.setInterval(() => setTick((n) => n + 1), REFRESH_MS);
    return () => window.clearInterval(t);
  }, []);
  useEffect(() => {
    if (!viewId) return setAi(null);
    let gone = false;
    getJSON<AiView>(`/api/aiview/${viewId}${sample ? "?sample=1" : ""}`)
      .then((a) => !gone && setAi(a))
      .catch(() => !gone && setAi(null));
    return () => void (gone = true);
  }, [viewId, heatTs, sample, tick]);

  return (
    <div className="see">
      {only !== "ai" && <div className="pane">
        <h3>What the camera sees <span className="ar" lang="ar">البث المباشر</span></h3>
        <div className="frame">
          {play && sba ? (
            <SbaVideo url={sbaUrl!} />
          ) : youtubeId && play ? (
            <YouTubeEmbed id={youtubeId} onBlocked={sbaUrl ? () => {
              setUseSba(true);
              setWhy("YouTube did not start (it may be asking to sign in to confirm you're not a bot): playing SBA's stream of the same broadcast instead.");
            } : undefined} />
          ) : youtubeId || sbaUrl ? (
            <button type="button" className="yt" onClick={() => setPlay(true)} aria-label="Play the live broadcast">
              <span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z" /></svg></span>
            </button>
          ) : (
            <div className="empty"><div><b>No stream configured</b>STREAM_URL is missing in .env.</div></div>
          )}
          {!play && <span className="badge">LIVE</span>}
          {!play && <span className="stamp">Saudi Quran TV</span>}
        </div>
        <p>
          {why ?? (sba ? "Official 24/7 broadcast from Masjid al-Haram, from SBA's own stream (720p) when you press play."
            : "Official 24/7 broadcast from Masjid al-Haram. Loads from YouTube (no cookies) when you press play.")}
          {sbaUrl && youtubeId && !youtubePaused && (
            <> <button type="button" className="src-switch" onClick={() => { setUseSba(!sba); setWhy(null); }}>
              {sba ? "Try YouTube" : "Use SBA's stream"}
            </button></>
          )}
        </p>
      </div>}
      <div className="pane">
        <h3>What the AI sees <span className="ar" lang="ar">خريطة الكثافة</span></h3>
        {ai?.ts && ai.approximate ? <p className="sync live"><i />Current shot · no known camera · approximate reading</p>
          : ai?.ts && onAir !== undefined && <SyncNote ai={ai} label={label} onAir={onAir} onAirLabel={onAirLabel ?? null} now={now} />}
        {ai ? <AiFrame ai={ai} label={label} color={view?.twin?.color ?? "#ff6f52"} now={now} /> : (
          <div className="frame">
            <div className="empty">
              <div><b>No counted frame yet</b>The density map appears with the first reading from a counting camera.</div>
            </div>
          </div>
        )}
        {ai && (
          <div className={`ai-scale${!ai.approximate && (ai.absolute || ai.people_m2 != null) ? " levels" : ""}`} aria-hidden="true">
            {!ai.approximate && (ai.absolute || ai.people_m2 != null) ? <LevelScale unit={false} />
              : <><span>fewer</span><i /><span /><i /><span>more people</span></>}
          </div>
        )}
        <p>
          {ai?.approximate
            ? "The shot on air has no known camera: colour shows where the people are, relative to this frame. The number is approximate (floor area from the size of the heads, about 2x either way) and is not placed on the Mataf."
            : ai?.absolute
            ? "People per m² on the Mataf floor, on the same scale as the twin; the floor is drawn faintly, the dark block is the Kaaba. The thin line is 25 m from the Kaaba's walls: inside it the headline rings, outside it the far Mataf out to the arcade (approximate)."
            : ai?.people_m2 != null
            ? "Colour on the same people-per-m² scale as the Mataf twin: the zone's average takes this camera's reading, denser spots go warmer (approximate)."
            : "Colour shows where the people are, relative to this frame (this camera has no people-per-m² reading yet)."}{" "}
          {ai?.approximate ? "No video pixels." : "Lines are the building, traced from the camera's reference view. No video pixels."}
        </p>
      </div>
    </div>
  );
}

/** The YouTube embed; onBlocked when it has not reported "playing" within YT_START_MS (the bot check page never does). */
export function YouTubeEmbed({ id, onBlocked }: { id: string; onBlocked?: () => void }) {
  const ref = useRef<HTMLIFrameElement>(null);
  useEffect(() => {
    let playing = false;
    const onMsg = (e: MessageEvent) => {
      if (!/^https:\/\/www\.youtube(-nocookie)?\.com$/.test(e.origin) || typeof e.data !== "string") return;
      try {
        const d = JSON.parse(e.data);
        if ((d.event === "onStateChange" && d.info === 1) || d.info?.playerState === 1) playing = true;
      } catch { /* not the player's message */ }
    };
    window.addEventListener("message", onMsg);
    // the player sends its state only after this "listening" handshake (IFrame API protocol)
    const hello = window.setInterval(() => ref.current?.contentWindow?.postMessage(
      JSON.stringify({ event: "listening", id: 1, channel: "widget" }), "*"), 1000);
    const t = window.setTimeout(() => !playing && onBlocked?.(), YT_START_MS);
    return () => { window.removeEventListener("message", onMsg); window.clearInterval(hello); window.clearTimeout(t); };
  }, [id, onBlocked === undefined]);
  const origin = encodeURIComponent(window.location.origin);
  return <iframe ref={ref} src={`https://www.youtube-nocookie.com/embed/${id}?autoplay=1&mute=1&rel=0&enablejsapi=1&origin=${origin}`}
    title="Makkah live broadcast" allow="autoplay; encrypted-media; picture-in-picture" allowFullScreen />;
}

/** SBA's HLS stream in a muted video (hls.js, loaded on demand; Safari plays HLS natively). */
export function SbaVideo({ url }: { url: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    let hls: { destroy(): void } | null = null;
    let gone = false;
    if (v.canPlayType("application/vnd.apple.mpegurl")) {
      v.src = url;
    } else {
      import("hls.js").then(({ default: Hls }) => {
        if (gone) return;
        if (!Hls.isSupported()) return setErr(true);
        const h = new Hls({ liveSyncDurationCount: 3 });
        h.on(Hls.Events.ERROR, (_e, d) => d.fatal && setErr(true));
        h.loadSource(url);
        h.attachMedia(v);
        hls = h;
      }).catch(() => setErr(true));
    }
    return () => { gone = true; hls?.destroy(); };
  }, [url]);
  return err ? <div className="empty"><div><b>SBA's stream did not load</b>Try again in a minute.</div></div>
    : <video ref={ref} muted autoPlay playsInline controls title="Makkah live broadcast (SBA)" />;
}

/**
 * Whether the AI view shows what the live video shows: the same camera, counted within the last minute. Otherwise
 * it says what it is showing instead.
 */
/** How the frame's camera was found (live shot service), in words. */
function methodLabel(method: string, how: string): string {
  const rows = how.includes("+rows") ? " · from the prayer rows" : "";
  if (method === "registered") return `position by image registration${rows}`;
  if (method === "calibrated") return `calibrated camera${rows}`;
  return rows.slice(3);
}

function SyncNote({ ai, label, onAir, onAirLabel, now }: { ai: AiView; label: string; onAir: string | null;
  onAirLabel: string | null; now: number }) {
  const age = minutesSince(ai.ts!, now);
  const n = Math.max(1, Math.round(age));
  if (onAir === ai.view_id && age < 1) return <p className="sync live"><i />In sync with the live video</p>;
  if (onAir === ai.view_id)
    return <p className="sync">Live video shows the {onAirLabel}; its last counted frame is {n} min old.</p>;
  return <p className="sync">Live video is on another camera – this is the last reading from the {label}, {n} min ago.</p>;
}

/** The heat images cross-fade: the new one fades in over the previous one, which then goes. */
function useLayers(url: string): string[] {
  const [layers, setLayers] = useState<string[]>([url]);
  const last = useRef(url);
  useEffect(() => {
    if (url === last.current) return;
    last.current = url;
    setLayers((l) => [l[l.length - 1], url]);
    const t = window.setTimeout(() => setLayers([url]), 900);
    return () => window.clearTimeout(t);
  }, [url]);
  return layers;
}

function AiFrame({ ai, label, color, now }: { ai: AiView; label: string; color: string; now: number }) {
  const [w, h] = ai.size;
  const box = `0 0 ${w} ${h}`;
  const layers = useLayers(ai.heat_url);
  const old = ai.ts != null && minutesSince(ai.ts, now) > OLD_MIN;
  const fresh = ai.ts != null && minutesSince(ai.ts, now) < 2;
  return (
    <div className="frame ai">
      {ai.bg_url && <img className="ai-bg" src={ai.bg_url} alt="" aria-hidden="true" />}
      <svg className="ai-layer" viewBox={box} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        {ai.outline && <path d={ai.outline} className="arch" />}
      </svg>
      {layers.map((u) => (
        <img key={u} className="ai-heat" src={u} alt={u === ai.heat_url ? `Where the people are in the last counted frame, ${label}` : ""} />
      ))}
      <svg className="ai-layer" viewBox={box} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        {Object.entries(ai.zones).map(([name, p]) => (
          <polygon key={name} points={p.map((q) => q.join(",")).join(" ")} className="zone" style={{ stroke: color }} />
        ))}
      </svg>
      <div className="zone-count" title={ai.people_m2 != null ? M2_HELP : undefined}
        style={{ left: `${clamp((100 * ai.label[0]) / w)}%`, top: `${clamp((100 * ai.label[1]) / h)}%` }}>
        {ai.people_m2 != null ? (
          <>
            <b className="num">{fmtM2(ai.people_m2)}</b>
            <span>people per m²</span>
            {ai.approximate ? <span>current shot, approximate</span>
              : ai.people_m2_ring && <span>1–25 m from the Kaaba</span>}
            {!ai.approximate && ai.people_m2_far != null && (
              <span className="far">far Mataf {fmtM2(ai.people_m2_far)} · {ai.far_confidence ?? "approximate"}</span>
            )}
            {!ai.approximate && ai.method && <span className="how">{methodLabel(ai.method, ai.how ?? "")}</span>}
          </>
        ) : <span className="plain">no people-per-m² reading from this camera yet</span>}
        <span>{ai.count != null && (ai.visible ?? 1) >= FULL_VISIBLE ? `${fmtInt(ai.count)} people in zone`
          : ai.visible != null ? `${Math.round(100 * ai.visible)} % of zone in view` : ""}</span>
      </div>
      <span key={ai.heat_url} className={`stamp${fresh ? " fresh" : ""}${old ? " old" : ""}`}>
        {label}{ai.ts ? ` · ${makkahTime(ai.ts)} · ${old ? `no new counted frame for ${ago(ai.ts, now).replace(" ago", "")}` : ago(ai.ts, now)}` : ""}
        {ai.sample ? " · sample" : ""}
      </span>
    </div>
  );
}
