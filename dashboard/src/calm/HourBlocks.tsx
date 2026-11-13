import { useMemo, useState } from "react";
import { crowdLevel, type CrowdLevel, type MatafBucket } from "../api";
import { clock, LEVEL_WORD, type Lang } from "../simple/i18n";

const HOURS = 6;
const HOUR = 3600_000; // Makkah is UTC+3, so its hours start with UTC's

/** The last 6 hours as 6 blocks (user 2026-11-03, in place of the line): one per clock hour, the newest the hour so far,
 * coloured by the level of the median of that hour's 5-minute Mataf readings; grey with no reading. The hour under
 * each; hover or touch shows the hour and the level word. Display only: the readings are unchanged. */
export default function HourBlocks({ series, now, lang }: { series: MatafBucket[]; now: number; lang: Lang }) {
  const [hover, setHover] = useState<number | null>(null);
  const hour0 = Math.floor(now / HOUR) * HOUR;
  const blocks = useMemo(() => Array.from({ length: HOURS }, (_, i) => {
    const start = hour0 - (HOURS - 1 - i) * HOUR;
    const vs = series
      .filter((b) => b.ring === "all" && b.people_m2 != null)
      .map((b) => ({ t: Date.parse(b.bucket_utc), v: b.people_m2 as number }))
      .filter((p) => p.t >= start && p.t < start + HOUR)
      .map((p) => p.v)
      .sort((a, b) => a - b);
    const n = vs.length;
    const level: CrowdLevel | null = n ? crowdLevel(n % 2 ? vs[(n - 1) / 2] : (vs[n / 2 - 1] + vs[n / 2]) / 2) : null;
    return { start, level, last: i === HOURS - 1 };
  }), [series, hour0]);
  const word = (b: (typeof blocks)[number]) => (b.level ? LEVEL_WORD[lang][b.level] : lang === "ar" ? "لا قراءة" : "no reading");
  const span = (b: (typeof blocks)[number]) => `${clock(lang, b.start)} – ${b.last ? (lang === "ar" ? "الآن" : "now") : clock(lang, b.start + HOUR)}`;
  const h = hover != null ? blocks[hover] : null;
  return (
    <div className="hours" dir="ltr" onPointerLeave={() => setHover(null)}>
      <ol>
        {blocks.map((b, i) => (
          <li key={b.start} onPointerEnter={() => setHover(i)} onPointerDown={() => setHover(i)}>
            <i className={b.level ?? "none"} role="img" aria-label={`${span(b)}: ${word(b)}`} />
            <span className={b.last ? "cur" : undefined}>{clock(lang, b.start, false)}</span>
          </li>
        ))}
      </ol>
      {h && (
        <span className="tip" style={{ left: `${((hover! + 0.5) / HOURS) * 100}%` }} dir={lang === "ar" ? "rtl" : "ltr"}>
          <bdi>{span(h)}</bdi> · {word(h)}
        </span>
      )}
    </div>
  );
}
