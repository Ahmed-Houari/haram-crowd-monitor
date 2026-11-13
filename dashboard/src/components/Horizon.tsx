import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { CROWD_LEVELS, fmtDensity, fmtM2, makkahTime, PRAYER_AR, PRAYER_EN, type CrowdBucket, type ForecastCam,
  type PrayerEvent } from "../api";
import { useWidth } from "../hooks";

interface Props {
  prayers: PrayerEvent[];
  series: CrowdBucket[]; // 5-min averages of the hero camera
  cameraLabel: string;
  now: number;
  /** the hero camera's forecast, if a model is trained (else the forecast area stays an empty slot) */
  forecast?: ForecastCam | null;
  /** the Mataf series (people per m²) instead of one camera's density */
  mataf?: boolean;
}

const BACK_H = 20, AHEAD_H = 4;
const ROW_H = 36, LABEL_GAP = 10, MAX_ROWS = 3, PLOT_H = 104;
const RUN_GAP_MS = 16 * 60000; // longer gap between 5-min averages = camera off air: break the curve

let ctx: CanvasRenderingContext2D | null = null;
function textWidth(text: string, font: string): number {
  ctx ??= document.createElement("canvas").getContext("2d");
  if (!ctx) return text.length * 7;
  ctx.font = font;
  return ctx.measureText(text).width;
}

interface Placed { p: PrayerEvent; t: number; X: number; row: number; left: boolean; w: number }

/**
 * Prayer labels are staggered into rows so close prayers (Maghrib + Isha are 90 min apart, ~47 px on a
 * phone) never overlap: each label goes into the first row where it clears the previous label by LABEL_GAP.
 * Labels near the right edge flip to the left of their line.
 */
export function placeLabels(events: { p: PrayerEvent; t: number; X: number }[], W: number): Placed[] {
  const rows: [number, number][][] = []; // occupied [start, end] pairs per row
  return events.map(({ p, t, X }) => {
    const en = `${PRAYER_EN[p.name] ?? p.name} ${p.makkah}`;
    const w = Math.max(textWidth(en, "500 12px 'Readex Pro Variable'"), textWidth(PRAYER_AR[p.name] ?? "", "12.5px 'Reem Kufi'")) + 2;
    const left = X + 5 + w > W - 2;
    const [a, b] = left ? [X - 5 - w, X - 5] : [X + 5, X + 5 + w];
    let row = rows.findIndex((occ) => occ.every(([s, e]) => b + LABEL_GAP <= s || a >= e + LABEL_GAP));
    if (row < 0) row = rows.length < MAX_ROWS ? rows.push([]) - 1 : rows.length - 1;
    rows[row].push([a, b]);
    return { p, t, X, row, left, w };
  });
}

/** Rolling 20 h back / 4 h ahead: crowd curve, five prayers (EN + AR), now-line, forecast 15/30/60 min ahead. */
export default function Horizon({ prayers, series, cameraLabel, now, forecast, mataf = false }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const W = useWidth(svgRef);
  const [fontsReady, setFontsReady] = useState(false);
  useEffect(() => void document.fonts?.ready.then(() => setFontsReady(true)), []);

  const t0 = now - BACK_H * 3600000, t1 = now + AHEAD_H * 3600000;
  const x = (t: number) => 4 + ((t - t0) / (t1 - t0)) * (W - 8);

  const placed = useMemo(() => {
    if (!W) return [];
    const ev = prayers.map((p) => ({ p, t: Date.parse(p.utc) })).filter(({ t }) => t >= t0 && t <= t1)
      .sort((a, b) => a.t - b.t).map((e) => ({ ...e, X: 4 + ((e.t - t0) / (t1 - t0)) * (W - 8) }));
    return placeLabels(ev, W);
    // fontsReady: re-measure once the web fonts are in
  }, [prayers, W, t0, t1, fontsReady]); // eslint-disable-line react-hooks/exhaustive-deps

  const rowsUsed = placed.reduce((m, l) => Math.max(m, l.row + 1), 1);
  const top = 46 + (rowsUsed - 1) * ROW_H;
  const base = top + PLOT_H;
  const H = base + 26;

  // phones: the band is wider than the screen; open it at "now"
  useLayoutEffect(() => {
    const box = boxRef.current;
    if (!box || box.scrollWidth <= box.clientWidth) return;
    let left = x(now) - box.clientWidth * 0.62;
    for (const l of placed) { // do not cut a prayer label at the left edge
      const a = l.left ? l.X - 5 - l.w : l.X - 2;
      if (a < left && left < a + l.w + 8) left = a - 6;
    }
    box.scrollLeft = left;
  }, [W, placed]); // eslint-disable-line react-hooks/exhaustive-deps

  const pts = series.filter((b) => b.density != null).map((b) => [Date.parse(b.bucket_utc), b.density as number] as const)
    .filter(([t]) => t >= t0 && t <= now).sort((a, b) => a[0] - b[0]);
  const fpts = (forecast?.predictions ?? []).map((p) => [Date.parse(p.utc), p.density] as const).filter(([t]) => t <= t1);
  const vmax = Math.max(mataf ? 0.5 : 5, ...pts.map((p) => p[1]), ...fpts.map((p) => p[1]),
    mataf ? 0 : forecast?.threshold_density ?? 0) * 1.15;
  const y = (v: number) => base - (v / vmax) * (base - top);
  const runs: (readonly [number, number])[][] = [];
  for (const p of pts) {
    const r = runs[runs.length - 1];
    if (r && p[0] - r[r.length - 1][0] <= RUN_GAP_MS) r.push(p);
    else runs.push([p]);
  }
  const xn = x(now);
  const last = pts[pts.length - 1];
  const forecastText = fpts.length ? "Forecast · next 60 min"
    : x(t1) - xn - 16 > textWidth("Forecast · after ~7 days of data", "11.5px 'Readex Pro Variable'")
      ? "Forecast · after ~7 days of data" : "Forecast · soon";

  const ticks: number[] = [];
  for (let t = Math.ceil(t0 / 7200000) * 7200000; t <= t1; t += 7200000) ticks.push(t);

  return (
    <section className="horizon" aria-label="Today around the prayers">
      <h3>
        Today around the prayers <span className="ar" lang="ar">مواقيت الصلاة</span>
        <small>{mataf ? "people per m² on the Mataf, all cameras" : `crowd density in the ${cameraLabel.toLowerCase()} zone (people per 100 × 100 px)`} · 5-min averages · forecast area ahead</small>
      </h3>
      <div className="hzscroll" ref={boxRef}>
        <svg ref={svgRef} className="hz" height={H} role="img"
          aria-label={`Crowd over the last ${BACK_H} hours with prayer times`}>
          <defs>
            <linearGradient id="hz-area" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#4fb3a8" stopOpacity=".55" /><stop offset="1" stopColor="#4fb3a8" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="hz-fc" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#c8a45c" stopOpacity=".2" /><stop offset="1" stopColor="#c8a45c" stopOpacity="0" />
            </linearGradient>
          </defs>
          {W > 0 && (
            <>
              {ticks.map((t) => (
                <g key={t}>
                  <line x1={x(t)} x2={x(t)} y1={base} y2={base + 5} stroke="rgba(236,231,220,.25)" />
                  {Math.abs(x(t) - xn) > 30 && x(t) > 18 && x(t) < W - 18 && (
                    <text x={x(t)} y={base + 20} fill="#8d9b9c" fontSize="10.5" textAnchor="middle">{makkahTime(t)}</text>
                  )}
                </g>
              ))}
              {runs.map((r, i) => {
                const d = r.map(([t, v], k) => `${k ? "L" : "M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`).join("");
                const x0 = x(r[0][0]), x1 = r.length > 1 ? x(r[r.length - 1][0]) : x0 + 2;
                return (
                  <g key={i}>
                    <path d={`${d}L${x1},${base}L${x0},${base}Z`} fill="url(#hz-area)" />
                    {r.length > 1 ? <path d={d} fill="none" stroke="#7fd0c4" strokeWidth="1.6" strokeLinejoin="round" />
                      : <circle cx={x0} cy={y(r[0][1])} r="1.8" fill="#7fd0c4" />}
                  </g>
                );
              })}
              {!pts.length && (
                <text x={(4 + xn) / 2} y={(top + base) / 2} fill="#8d9b9c" fontSize="12" textAnchor="middle">
                  No crowd readings from this camera in the last {BACK_H} h yet
                </text>
              )}
              <rect x={xn} y={top - 8} width={Math.max(0, x(t1) - xn)} height={base - top + 8} fill="url(#hz-fc)" />
              <text x={xn + 10} y={top + 6} fill="#c8a45c" fontSize="11.5">{forecastText}</text>
              {fpts.length > 0 && (() => {
                const from = last && now - last[0] < 60 * 60000 ? last : null;
                const path = (from ? [from, ...fpts] : fpts).map(([t, v], i) => `${i ? "L" : "M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`).join("");
                const thr = mataf ? null : forecast?.threshold_density;  // the Mataf: the level lines (below)
                return (
                  <g>
                    {thr != null && y(thr) > top - 8 && (
                      <line x1={xn} x2={x(t1)} y1={y(thr)} y2={y(thr)} stroke="#ff6f52" strokeOpacity=".6" strokeDasharray="3 3" />
                    )}
                    <path d={path} fill="none" stroke="#c8a45c" strokeWidth="1.8" strokeDasharray="5 4" />
                    {fpts.map(([t, v], i) => (
                      <g key={t}>
                        <circle cx={x(t)} cy={y(v)} r="3.4" fill="#0f1a1d" stroke="#c8a45c" strokeWidth="1.6" />
                        {i === fpts.length - 1 && (
                          <text x={x(t) + 6} y={y(v) - 6} fill="#e2c482" fontSize="11">{mataf ? fmtM2(v) : fmtDensity(v)}</text>
                        )}
                      </g>
                    ))}
                  </g>
                );
              })()}
              {mataf && CROWD_LEVELS.slice(0, 3).filter((l) => l.upTo < vmax).map((l, i) => (
                <g key={l.name}>
                  <line x1="4" x2={W - 4} y1={y(l.upTo)} y2={y(l.upTo)} strokeOpacity=".5" strokeDasharray="2 4"
                    stroke={`rgb(${CROWD_LEVELS[i + 1].rgb.join(",")})`} />
                  <text x={W - 8} y={y(l.upTo) - 4} fill="#8d9b9c" fontSize="10.5" textAnchor="end">
                    {CROWD_LEVELS[i + 1].name} from {l.upTo.toFixed(1)}
                  </text>
                </g>
              ))}
              <line x1="4" x2={W - 4} y1={base} y2={base} stroke="rgba(236,231,220,.18)" />
              {placed.map(({ p, X, row, left }) => {
                const ty = 16 + row * ROW_H;
                const tx = left ? X - 5 : X + 5;
                const anchor = left ? "end" : "start";
                return (
                  <g key={p.utc}>
                    <line x1={X} x2={X} y1={ty - 12} y2={base} stroke="rgba(236,231,220,.28)" strokeDasharray={p.name === "sunrise" ? "2 3" : undefined} />
                    <text x={tx} y={ty} fill="#ece7dc" fontSize="12" fontWeight="500" textAnchor={anchor}>
                      {PRAYER_EN[p.name] ?? p.name} <tspan fill="#8d9b9c" fontWeight="400">{p.makkah}</tspan>
                    </text>
                    <text x={left ? X - 5 : X + 5} y={ty + 17} fill="#8d9b9c" fontSize="12.5" className="ar-l" textAnchor={anchor}
                      direction="ltr">{PRAYER_AR[p.name]}</text>
                  </g>
                );
              })}
              <line x1={xn} x2={xn} y1={6} y2={base + 6} stroke="#c8a45c" strokeWidth="2" />
              {last && now - last[0] < 30 * 60000 && (
                <>
                  <circle cx={x(last[0])} cy={y(last[1])} r="4.5" fill="#c8a45c" />
                  <circle cx={x(last[0])} cy={y(last[1])} r="10" fill="none" stroke="#c8a45c" strokeOpacity=".5" />
                </>
              )}
              <text x={xn} y={base + 20} fill="#c8a45c" fontSize="10.5" textAnchor="middle">now</text>
            </>
          )}
        </svg>
      </div>
    </section>
  );
}
