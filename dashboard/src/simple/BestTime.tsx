// The best time to go (user 2026-11-03), on /simple and the main dashboard: the lightest two hours left today and
// tomorrow's, from past days' readings since 1 Nov, outside the prayers (history.lightestWindows), each with its level
// word; when even the lightest time of the day is crowded, it says so plainly.
import { useMemo } from "react";
import { CROWD_LEVELS, crowdLevel, type CrowdLevel, type PrayerTime } from "../api";
import { lightestWindows, type Point, type Window } from "./history";
import { hourSpan, LEVEL_WORD, T, type Lang } from "./i18n";
import "./besttime.css";

const levelRgb = (l: CrowdLevel) => `rgb(${CROWD_LEVELS.find((x) => x.name === l)!.rgb.join(",")})`;

/** times: today's prayer times (undefined while loading, null when the API has none). */
export default function BestTime({ pts, times, now, lang }: {
  pts: Point[] | null; times: Record<string, PrayerTime> | null | undefined; now: number; lang: Lang;
}) {
  const t = T[lang];
  const tenMin = Math.floor(now / 600_000);
  const w = useMemo(() => (pts && times ? lightestWindows(pts, now, times) : null), [pts, times, tenMin]);
  if (!w) return <p className="best-sub">{pts && times !== undefined ? t.notEnough : t.loading}</p>;
  const busy = crowdLevel(w.tomorrow.v) === "crowded" || crowdLevel(w.tomorrow.v) === "packed";
  const line = (label: string, x: Window | null) => (
    <p className="best-line">
      <span className="best-l">{label}:</span>{" "}
      {x ? (
        <>
          <b>{hourSpan(lang, x.start, x.start + 2 * 3600_000)}</b>
          <span className="best-lv"><i style={{ background: levelRgb(crowdLevel(x.v)) }} />{LEVEL_WORD[lang][crowdLevel(x.v)]}</span>
        </>
      ) : <span className="best-l">{t.noneLeft}</span>}
    </p>
  );
  return (
    <div className="best">
      {busy && <p className="best-plain">{t.busyDay}</p>}
      {line(t.lightToday, w.today)}
      {line(t.lightTomorrow, w.tomorrow)}
      <p className="best-note">{t.estimateNote}</p>
    </div>
  );
}
