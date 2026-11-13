import { useEffect, useRef, useState } from "react";
import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";
import { ago, CROWD_LEVELS, crowdLevel, DENSITY_HELP, DENSITY_UNIT, fmtDensity, fmtInt, fmtM2, FULL_VISIBLE, LEVELS_HELP, M2_HELP,
  M2_UNIT, makkahTime, minutesSince, viewLabel, type CrowdNow, type Level, type MatafNow, type Status, type Usual,
  type View } from "../api";
import { useTween, useWidth } from "../hooks";

interface Props {
  hero: CrowdNow | null;
  view: View | undefined;
  /** this camera's own people per m² (view-independent metric), null while it has none */
  m2: { v: number; at: string } | null;
  status: Status | null;
  now: number;
  reduce: boolean;
  staleMin: number;
}

const LEVEL_WORD = { quiet: "Quiet", busy: "Busy", crowded: "Crowded", packed: "Packed" } as const;

/**
 * The biggest thing on the page: crowd density in the visible part of one camera's counted zone (people per 100 x 100 px
 * of floor in its reference view), with level, trend, last 2 h and usual; people in the whole zone only when it is fully
 * visible.
 */
export default function HeroNumber(props: Props) {
  const { hero } = props;
  // one unit on the page (2026-10-26): a camera shows its own people per m², or says plainly that it has none
  if (hero && (props.m2 || hero.visible !== undefined)) return <CameraM2 {...props} hero={hero} />;
  return <LegacyHero {...props} />;
}

/** An API from before the visible-part metric, or no reading at all. */
function LegacyHero({ hero, view, status, now, reduce, staleMin }: Props) {
  const value = hero?.density ?? null;
  const shown = useTween(value, reduce);
  // an API from before the visible-part metric sends count-based trend / spark / levels: show only the density then
  const legacy = hero != null && hero.visible === undefined;
  const pulse = usePulse(hero ? `${hero.view_id}|${hero.minute_utc}` : null);
  const stale = hero != null && minutesSince(hero.minute_utc, now) > staleMin;
  const name = hero ? viewLabel(view, hero.view_id) : "";

  if (!hero || value == null)
    return (
      <div className="metric">
        <div className="lbl">Crowd density in the counted zone <span className="ar" lang="ar">كثافة الحشود</span></div>
        <div className="big num empty" aria-label="no reading">–</div>
        <p className="empty-msg">No counted reading yet.</p>
        <p className="note">
          Readings appear when the broadcast shows one of the counting cameras (low, roof or wide view of the Mataf).
          {status?.running ? ` The collector is running: last frame ${makkahTime(status.last_frame_utc)}.` : ""}
        </p>
      </div>
    );

  return (
    <div className="metric">
      <div className="lbl">
        {stale ? "Crowd density at the last reading" : "Crowd density now"}
        <span className="ar" lang="ar">{stale ? "كثافة الحشود" : "كثافة الحشود الآن"}</span>
      </div>
      <div className={`big num${stale ? " stale" : ""}`} title={DENSITY_HELP}>
        <span aria-hidden="true">{fmtDensity(shown ?? value)}</span>
        <span className="sr" aria-live="polite">{fmtDensity(value)} {DENSITY_UNIT}</span>
        {pulse > 0 && !reduce && <span key={pulse} className="ring" aria-hidden="true" />}
      </div>
      <div className="unit" title={DENSITY_HELP}>{DENSITY_UNIT} · visible part of the zone</div>
      <div className="chips-row">
        <LevelBadge level={hero.level} />
        {!legacy && <Trend trend={hero.trend} />}
      </div>
      {!legacy && <Sparkline points={hero.spark} level={hero.level} now={now} />}
      {!legacy && <UsualLine usual={hero.usual} />}
      {!legacy && <Visible hero={hero} />}
      <div className="src">
        <span className={`dot${stale ? "" : " on"}`} />
        <span>
          {name}
          {view?.quality === "low" && <span className="q" style={{ color: "#f1b45a" }}> · lower quality</span>} · last reading{" "}
          {makkahTime(hero.minute_utc)} ({ago(hero.minute_utc, now)}) · {hero.source === "youtube" ? "YouTube 1080p" : hero.source === "sba" ? "SBA 720p" : "SBA fallback 720p"}
        </span>
      </div>
      <p className="note">
        Density of the part of this camera&rsquo;s floor zone that the shot shows, not the whole Mataf; zoomed-in and partly
        visible shots stay comparable. Compare it over time, not between cameras.
        {hero.level.state !== "calibrating" && ` Levels compare with this camera's own last ${levelDays(hero.level)} days.`}
      </p>
    </div>
  );
}

/**
 * One picked camera in people per m² (its own view-independent reading). A camera without one says so in plain words
 * (no pixel unit: 2026-10-26, one unit on the page).
 */
function CameraM2({ hero, view, m2, now, reduce, staleMin }: { hero: CrowdNow; view: View | undefined;
  m2: { v: number; at: string } | null; now: number; reduce: boolean; staleMin: number }) {
  const shown = useTween(m2?.v ?? null, reduce);
  const pulse = usePulse(m2 ? `${hero.view_id}|${m2.at}` : null);
  const name = viewLabel(view, hero.view_id);
  const stale = m2 != null && minutesSince(m2.at, now) > staleMin;
  return (
    <div className="metric">
      <div className="lbl">
        {name}: {stale ? "crowd density at the last reading" : "crowd density now"}
        <span className="ar" lang="ar">كثافة الحشود</span>
      </div>
      {m2 ? (
        <>
          <div className={`big num${stale ? " stale" : ""}`} title={M2_HELP}>
            <span aria-hidden="true">{fmtM2(shown ?? m2.v)}</span>
            <span className="sr" aria-live="polite">{fmtM2(m2.v)} {M2_UNIT}</span>
            {pulse > 0 && !reduce && <span key={pulse} className="ring" aria-hidden="true" />}
          </div>
          <div className="unit" title={M2_HELP}>{M2_UNIT} · what this camera sees of the Mataf</div>
        </>
      ) : (
        <>
          <div className="big num empty" aria-label="no reading in people per square metre">–</div>
          <p className="empty-msg">No people-per-m² reading from this camera in the last 20 minutes.</p>
        </>
      )}
      <div className="src">
        <span className={`dot${m2 && !stale ? " on" : ""}`} />
        <span>
          {name}
          {view?.quality === "low" && <span className="q" style={{ color: "#f1b45a" }}> · lower quality</span>}
          {m2 ? <> · last reading {makkahTime(m2.at)} ({ago(m2.at, now)})</> : <> · last frame {makkahTime(hero.minute_utc)} ({ago(hero.minute_utc, now)})</>}
        </span>
      </div>
      <p className="note">
        {m2 ? "People per square metre of the floor this camera shows, the same unit as the Mataf number. Pick the camera again to go back to all cameras."
          : "It gets one when the broadcast shows it and the camera is recognised. Pick the camera again to go back to all cameras."}
      </p>
    </div>
  );
}

/**
 * The view-independent hero (2026-10-24): people per m² on the Mataf floor, all cameras together, with the same level,
 * trend, last 2 h and usual as the per-camera hero, and the two rings around the Kaaba.
 */
export function MatafHero({ mataf, views, now, reduce, staleMin }: { mataf: MatafNow; views: View[]; now: number;
  reduce: boolean; staleMin: number }) {
  const shown = useTween(mataf.people_m2, reduce);
  const pulse = usePulse(mataf.minute_utc);
  const stale = minutesSince(mataf.minute_utc, now) > staleMin;
  const names = mataf.cameras.map((id) => (id === "kaaba" ? "a camera fitted to the Kaaba"
    : viewLabel(views.find((v) => v.id === id), id).toLowerCase()));
  return (
    <div className="metric">
      <div className="lbl">
        {stale ? "Crowd density on the Mataf at the last reading" : "Crowd density on the Mataf now"}
        <span className="ar" lang="ar">{stale ? "كثافة المطاف" : "كثافة المطاف الآن"}</span>
      </div>
      <div className={`big num${stale ? " stale" : ""}`} title={M2_HELP}>
        <span aria-hidden="true">{fmtM2(shown ?? mataf.people_m2)}</span>
        {minutesSince(mataf.minute_utc, now) > 1 && <span className="age">{ago(mataf.minute_utc, now)}</span>}
        <span className="sr" aria-live="polite">{fmtM2(mataf.people_m2)} {M2_UNIT}</span>
        {pulse > 0 && !reduce && <span key={pulse} className="ring" aria-hidden="true" />}
      </div>
      <div className="unit" title={M2_HELP}>{M2_UNIT} · 1–25 m from the Kaaba · all cameras</div>
      <div className="chips-row">
        <SafetyBadge v={mataf.people_m2} />
        <Trend trend={mataf.trend} fmt={fmtM2} />
      </div>
      <Sparkline points={mataf.spark} level={{ ...mataf.level, low: undefined, high: undefined }} now={now} who="the Mataf"
        marks={CROWD_LEVELS.slice(0, 3).map((l) => l.upTo)} />
      <UsualLine usual={mataf.usual} />
      {mataf.rings.inner != null && mataf.rings.outer != null && (
        <p className="visible">
          <span>near the Kaaba <b>{fmtM2(mataf.rings.inner)}</b> · further out <b>{fmtM2(mataf.rings.outer)}</b>
            {mataf.far && <> · far Mataf <b>{fmtM2(mataf.far.people_m2)}</b></>}</span>
          <span className="mute">· per m², 1–10 m and 10–25 m from its walls{mataf.far
            ? `, far Mataf 25 m out to the arcade (${mataf.far.confidence}, not in the headline)` : ""}{mataf.speed
            ? "; walking parts of the rings from the tawaf's walking speed" : ""}</span>
        </p>
      )}
      <div className="src">
        <span className={`dot${stale ? "" : " on"}`} />
        <span>
          from {names.join(", ") || "the cameras"} · last reading {makkahTime(mataf.minute_utc)} ({ago(mataf.minute_utc, now)})
        </span>
      </div>
      <p className="note">
        People per square metre of floor, measured from every camera whose position is known: the same unit for every
        camera, zoom and framing, by day and by night. Levels: Fruin&rsquo;s walkway level of service (quiet below 0.7,
        busy 0.7–1.1, crowded 1.1–2.2, packed from 2.2 per m²).
      </p>
    </div>
  );
}

/** How much of the zone the reading saw; people in the whole zone only when all of it was in view. */
function Visible({ hero }: { hero: CrowdNow }) {
  const v = hero.visible ?? 1;
  if (v >= FULL_VISIBLE && hero.count != null)
    return (
      <p className="visible full">
        <span><b>{fmtInt(hero.count)} people</b> in the whole zone</span><span className="mute">· zone fully in view</span>
      </p>
    );
  return (
    <p className="visible" title="People in the whole zone are shown only when the shot shows all of it.">
      <i style={{ ["--v" as string]: v }} aria-hidden="true" />
      <span><b>{Math.round(100 * v)} %</b> of the zone in view</span><span className="mute">· whole-zone count only when fully in view</span>
    </p>
  );
}

/** Increments when `key` changes after the first value (a new reading landed). */
function usePulse(key: string | null): number {
  const first = useRef(key);
  const [n, setN] = useState(0);
  useEffect(() => {
    if (key && key !== first.current) setN((v) => v + 1);
    first.current = key;
  }, [key]);
  return n;
}

const levelDays = (l: Level) => Math.max(1, Math.min(l.history_days, Math.floor(l.days)));

function LevelBadge({ level, who = "this camera's", unit = DENSITY_UNIT, fmt = fmtDensity }: { level: Level; who?: string;
  unit?: string; fmt?: (v: number) => string }) {
  if (level.state === "calibrating") {
    const p = Math.round(100 * Math.min(1, level.days / level.need_days, level.buckets / level.need_buckets));
    return (
      <span className="level calibrating" style={{ ["--p" as string]: p }}
        title={`Quiet / busy / crowded will come from ${who} own readings once ${level.need_days} days are collected: the lowest ${level.pcts[0]} % of its 5-min averages = quiet, the top ${100 - level.pcts[1]} % = crowded.`}>
        <i aria-hidden="true" />Calibrating · {level.days.toFixed(1)} of {level.need_days} days
      </span>
    );
  }
  return (
    <span className={`level ${level.state}`}
      title={`Compared with ${who} own 5-min averages over the last ${levelDays(level)} days: quiet below ${fmt(level.low ?? 0)}, crowded from ${fmt(level.high ?? 0)} ${unit} (lowest ${level.pcts[0]} % / top ${100 - level.pcts[1]} %).`}>
      <i aria-hidden="true" />{LEVEL_WORD[level.state]}
    </span>
  );
}

/** The Mataf's crowd-safety level (Fruin's walkway level of service) for its people per m². */
function SafetyBadge({ v }: { v: number }) {
  const lv = crowdLevel(v);
  const i = CROWD_LEVELS.findIndex((l) => l.name === lv);
  const range = i === 0 ? "below 0.7" : i === CROWD_LEVELS.length - 1 ? "from 2.2"
    : `${CROWD_LEVELS[i - 1].upTo.toFixed(1)}–${CROWD_LEVELS[i].upTo.toFixed(1)}`;
  return (
    <span className={`level ${lv}`} title={LEVELS_HELP}>
      <i aria-hidden="true" />{LEVEL_WORD[lv]} · {range} per m²
    </span>
  );
}

function Trend({ trend, fmt = fmtDensity }: { trend: CrowdNow["trend"]; fmt?: (v: number) => string }) {
  if (!trend) return <span className="pill trend none">no reading 30 min earlier</span>;
  const p = trend.pct;
  const flat = Math.abs(p) < 2;
  const Icon = flat ? ArrowRight : p > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <span className={`pill trend${flat ? " flat" : ""}`} title={`5-min average vs ${makkahTime(trend.ref_utc)} (${fmt(trend.ref_density)})`}>
      <Icon aria-hidden="true" strokeWidth={2.2} />
      {flat ? "steady" : `${Math.abs(Math.round(p))}%`} vs 30 min ago
    </span>
  );
}

function UsualLine({ usual }: { usual: Usual | null }) {
  if (!usual) return null;
  if (usual.state === "waiting")
    return <p className="usual wait">Comparison with the usual crowd at this time starts after {usual.need_days} days of data ({usual.days.toFixed(1)} so far).</p>;
  if (usual.state === "no_history_at_time" || usual.pct == null)
    return <p className="usual wait">Not enough earlier readings at this time of day to say what is usual yet.</p>;
  const p = Math.round(usual.pct);
  if (Math.abs(p) < 3)
    return <p className="usual"><b>About usual</b> for this time <span className="mute">· median of {usual.days_at_time} earlier days</span></p>;
  return (
    <p className={`usual ${p > 0 ? "up" : "down"}`}>
      <b>{Math.abs(p)}% {p > 0 ? "above" : "below"} usual</b> for this time <span className="mute">· median of {usual.days_at_time} earlier days</span>
    </p>
  );
}

const SPARK_H = 40;
const GAP_MS = 8 * 60000; // a longer gap = the camera was off air: break the line

/** Last 2 h of per-minute readings; faint band = this camera's "busy" range once calibrated; marks: faint dashed lines
 * at these values (the Mataf: the crowd-safety level boundaries) where they fall inside the plotted range. */
function Sparkline({ points, level, now, who = "this camera", marks }: { points: [string, number][]; level: Level;
  now: number; who?: string; marks?: number[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const width = Math.min(useWidth(ref) || 260, 400);
  const t0 = now - 2 * 3600000;
  const pts = points.map(([t, v]) => [Date.parse(t), v] as const).filter(([t]) => t >= t0 && t <= now + 60000);
  if (!pts.length)
    return <div className="spark" ref={ref}><p className="usual wait" style={{ margin: 0 }}>No readings from {who} in the last 2 h.</p></div>;
  // scale to the readings (span at least 20 % of their mean, so noise is not blown up); the band is clipped
  const vals = pts.map((p) => p[1]);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const grow = Math.max(0, mean * 0.2 - (hi - lo)) / 2;
  lo -= grow + (hi - lo) * 0.1;
  hi += grow + (hi - lo) * 0.1;
  const x = (t: number) => 1 + ((t - t0) / (now - t0)) * (width - 6);
  const yRaw = (v: number) => 3 + (1 - (v - lo) / (hi - lo)) * (SPARK_H - 6);
  const y = (v: number) => Math.max(0, Math.min(SPARK_H, yRaw(v)));
  const runs: (readonly [number, number])[][] = [];
  for (const p of pts) {
    const run = runs[runs.length - 1];
    if (run && p[0] - run[run.length - 1][0] <= GAP_MS) run.push(p);
    else runs.push([p]);
  }
  const last = pts[pts.length - 1];
  return (
    <div className="spark" ref={ref}>
      <svg width={width} height={SPARK_H} role="img" aria-label="Crowd density over the last 2 hours">
        {level.low != null && level.high != null && (
          <>
            <rect x={0} width={width} y={y(level.high)} height={Math.max(1, y(level.low) - y(level.high))} fill="rgba(216,180,95,0.09)" />
          </>
        )}
        {(marks ?? []).filter((m) => yRaw(m) > 1 && yRaw(m) < SPARK_H - 1).map((m) => (
          <line key={m} x1={0} x2={width} y1={y(m)} y2={y(m)} stroke={`rgb(${CROWD_LEVELS.find((l) => m < l.upTo)!.rgb.join(",")})`}
            strokeOpacity={0.45} strokeDasharray="2 3" />
        ))}
        <line x1={0} x2={width} y1={SPARK_H - 0.5} y2={SPARK_H - 0.5} stroke="rgba(236,231,220,0.12)" />
        {runs.map((run, i) =>
          run.length > 1 ? (
            <path key={i} d={run.map(([t, v], k) => `${k ? "L" : "M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`).join("")}
              fill="none" stroke="#7fd0c4" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
          ) : (
            <circle key={i} cx={x(run[0][0])} cy={y(run[0][1])} r={1.7} fill="#7fd0c4" />
          ),
        )}
        <circle cx={x(last[0])} cy={y(last[1])} r={3.2} fill="#c8a45c" />
      </svg>
      <div className="axis" style={{ width }}>
        <span>2 h ago</span>
        {level.low != null && <span><i className="band" />busy range for {who}</span>}
        {marks && marks.some((m) => yRaw(m) > 1 && yRaw(m) < SPARK_H - 1) && <span>dashed: level boundaries</span>}
        <span>now</span>
      </div>
    </div>
  );
}
