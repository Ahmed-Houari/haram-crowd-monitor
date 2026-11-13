import { useRef, type ReactNode } from "react";
import { makkahTime, PRAYER_EN, type PrayerEvent, type TrafficLatest, type TrafficPoint } from "../api";
import { useWidth } from "../hooks";

interface Props {
  children?: ReactNode; // the alerts panel, under the road chart
  latest: TrafficLatest | null;
  series: TrafficPoint[];
  prayers: PrayerEvent[];
}

const area = (l: TrafficLatest | null, a: "r1km" | "r3km") => l?.areas.find((x) => x.area === a && x.road_category === "all");

/** Road KPIs (1 km / 3 km) + 24 h congestion chart, then the alerts panel. */
export default function RoadPanel({ latest, series, prayers, children }: Props) {
  return (
    <div className="roads">
      <div className="kpis">
        {(["r1km", "r3km"] as const).map((a) => {
          const v = area(latest, a);
          return (
            <div className="kpi" key={a}>
              <span>Within {a === "r1km" ? "1" : "3"} km</span>
              <b className="num">{v?.congested_pct != null ? `${Math.round(v.congested_pct)}%` : "–"}</b>
              <em>{v?.mean_level != null ? `congested · speed ${Math.round(v.mean_level * 100)}% of free flow` : "no TomTom poll yet"}</em>
            </div>
          );
        })}
      </div>
      <div className="pane">
        <h3>Road congestion, 24 h <small>% of road length below half of free-flow speed</small></h3>
        <div className="chart-legend">
          <span><i style={{ background: "#e8384f" }} />within 1 km</span>
          <span><i style={{ background: "#f59f00" }} />within 3 km</span>
          {latest?.poll_utc && <span>last poll {makkahTime(latest.poll_utc)}</span>}
        </div>
        <CongestionChart series={series} prayers={prayers} />
      </div>
      {children}
    </div>
  );
}

function CongestionChart({ series, prayers }: { series: TrafficPoint[]; prayers: PrayerEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const H = 140, pl = 32, pr = 6, pt = 26, pb = 18;
  const by = { r1km: [] as [number, number][], r3km: [] as [number, number][] };
  for (const p of series) if (p.congested_pct != null) by[p.area]?.push([Date.parse(p.poll_utc), p.congested_pct]);
  const all = [...by.r1km, ...by.r3km];
  if (!all.length) return <div ref={ref} className="mute" style={{ height: H, display: "grid", placeItems: "center" }}>No road data in the last 24 h.</div>;
  const t1 = Math.max(...all.map((p) => p[0])), t0 = Math.max(t1 - 24 * 3600000, Math.min(...all.map((p) => p[0])));
  const vmax = Math.max(10, ...all.map((p) => p[1])) * 1.1;
  const x = (t: number) => pl + ((t - t0) / (t1 - t0)) * (W - pl - pr);
  const y = (v: number) => pt + (1 - v / vmax) * (H - pt - pb);
  const line = (pts: [number, number][]) =>
    pts.filter(([t]) => t >= t0).map(([t, v], i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ticks = Array.from({ length: 5 }, (_, i) => t0 + (i * (t1 - t0)) / 4);
  return (
    <div ref={ref}>
      {W > 0 && (
        <svg width={W} height={H} className="chart" role="img" aria-label="Road congestion over the last 24 hours">
          {[0, 0.5, 1].map((k) => (
            <g key={k}>
              <line x1={pl} x2={W - pr} y1={y(vmax * k / 1.1)} y2={y(vmax * k / 1.1)} stroke="rgba(236,231,220,.08)" />
              <text x={pl - 6} y={y(vmax * k / 1.1) + 4} fill="#8d9b9c" fontSize="10.5" textAnchor="end">{Math.round(vmax * k / 1.1)}%</text>
            </g>
          ))}
          {placePrayerTicks(prayers, t0, t1, x).map(({ p, X, row }) => (
            <g key={p.utc}>
              <line x1={X} x2={X} y1={row ? pt - 9 : pt - 20} y2={H - pb} stroke="rgba(200,164,92,.3)" strokeDasharray="2 3" />
              <text x={X + 3} y={row ? pt - 1 : pt - 12} fill="#c8a45c" fontSize="10">{PRAYER_EN[p.name]}</text>
            </g>
          ))}
          {ticks.map((t) => (
            <text key={t} x={x(t)} y={H - 4} fill="#8d9b9c" fontSize="10.5" textAnchor={t === t0 ? "start" : t === t1 ? "end" : "middle"}>{makkahTime(t)}</text>
          ))}
          <path d={line(by.r3km)} fill="none" stroke="#f59f00" strokeWidth="1.8" strokeLinejoin="round" />
          <path d={line(by.r1km)} fill="none" stroke="#e8384f" strokeWidth="1.8" strokeLinejoin="round" />
        </svg>
      )}
    </div>
  );
}

/** Prayer ticks for the small chart: a label that would touch the previous one drops to a second row. */
function placePrayerTicks(prayers: PrayerEvent[], t0: number, t1: number, x: (t: number) => number) {
  let end0 = -Infinity;
  return prayers.filter((p) => p.name !== "sunrise").map((p) => ({ p, t: Date.parse(p.utc) }))
    .filter(({ t }) => t > t0 && t < t1).sort((a, b) => a.t - b.t)
    .map(({ p, t }) => {
      const X = x(t), w = (PRAYER_EN[p.name] ?? p.name).length * 5.8 + 8;
      const row = X + 3 < end0 ? 1 : 0;
      if (!row) end0 = X + 3 + w;
      return { p, X, row };
    });
}
