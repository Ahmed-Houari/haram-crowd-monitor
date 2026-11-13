// The forecast line (user 2026-11-07), on /simple and the main dashboard: the Mataf's level in 30 minutes and in an
// hour, from the nightly forecast (scripts/forecast_nightly.py, /api/forecast/next), marked as a forecast. It went on
// the pages after it beat "same time yesterday" on every day since 1 Nov, Friday included. Words and colours only, no
// numbers; nothing shows without a live reading or a model. Display only: the readings are unchanged.
import { useEffect, useState } from "react";
import { CROWD_LEVELS, crowdLevel, getJSON, type CrowdLevel } from "../api";
import { LEVEL_WORD, T, type Lang } from "./i18n";
import "./forecast.css";

type ForecastNext = { available: boolean; items?: { minutes: number; target_utc: string; people_m2: number }[] };

const levelRgb = (l: CrowdLevel) => `rgb(${CROWD_LEVELS.find((x) => x.name === l)!.rgb.join(",")})`;

export default function ForecastLine({ lang }: { lang: Lang }) {
  const t = T[lang];
  const [fc, setFc] = useState<ForecastNext | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => getJSON<ForecastNext>("/api/forecast/next").then((d) => alive && setFc(d)).catch(() => alive && setFc(null));
    load();
    const id = window.setInterval(load, 60_000);
    return () => { alive = false; window.clearInterval(id); };
  }, []);
  const items = fc?.available ? (fc.items ?? []).filter((x) => t.fcIn[x.minutes]) : [];
  if (!items.length) return null;
  return (
    <p className="fc">
      <span className="fc-tag">{t.forecast}</span>
      {items.map((x) => {
        const l = crowdLevel(x.people_m2);
        return (
          <span key={x.minutes} className="fc-item">
            <span className="fc-when">{t.fcIn[x.minutes]}:</span>
            <span className="fc-lv"><i style={{ background: levelRgb(l) }} aria-hidden="true" />{LEVEL_WORD[lang][l]}</span>
          </span>
        );
      })}
    </p>
  );
}
