import { ago, hijriLabel, makkahDay, makkahTime, type PrayerToday, type Status } from "../api";

interface Props {
  status: Status | null;
  connected: boolean;
  prayer: PrayerToday | null;
  now: number;
}

/** Brand, LIVE state (live / stale / offline), collector facts, Makkah clock + Hijri date. */
export default function Header({ status, connected, prayer, now }: Props) {
  const state = !connected || !status?.running ? "off" : status.fresh ? "on" : "stale";
  const word = state === "on" ? "LIVE" : state === "stale" ? "STALE" : "OFFLINE";
  const hint =
    state === "on" ? "" : !connected ? "no connection to the API" : !status?.running ? "collector stopped"
      : `last frame ${ago(status.last_frame_utc, now)}`;
  const accepted = status ? Object.values(status.accepted_today).reduce((a, b) => a + b, 0) : null;
  return (
    <header className="top">
      <div className="brand">
        <svg viewBox="0 0 40 40" aria-hidden="true">
          <g fill="none" stroke="#c8a45c" strokeWidth="1.5">
            <rect x="9" y="9" width="22" height="22" />
            <rect x="9" y="9" width="22" height="22" transform="rotate(45 20 20)" />
            <circle cx="20" cy="20" r="6.5" />
          </g>
          <rect x="17" y="17" width="6" height="6" fill="#c8a45c" />
        </svg>
        <div>
          <span className="en">Haram Crowd Monitor</span>
          <span className="ar" lang="ar">مرصد حشود المسجد الحرام</span>
        </div>
      </div>
      <div className={`live ${state}`} role="status" title={hint || "Collector running, frames arriving"}>
        <span className="beat" />
        <b>{word}</b>
        {state === "on" ? (
          <svg className="ecg" viewBox="0 0 46 16" aria-hidden="true"><path d="M0 9h12l3-6 4 11 3-8 2 3h22" /></svg>
        ) : (
          <small>{hint}</small>
        )}
      </div>
      <div className="facts">
        <div><span>Frames today</span><b>{status ? status.frames_today.toLocaleString("en-US") : "–"}</b></div>
        <div><span>Counted today</span><b>{accepted == null ? "–" : accepted.toLocaleString("en-US")}</b></div>
        <div><span>Uptime today</span><b>{status?.uptime_today_pct != null ? `${status.uptime_today_pct}%` : "–"}</b></div>
      </div>
      <div className="clock">
        <b className="num">{makkahTime(now)}</b>
        <span>Makkah · {makkahDay(now)}{prayer?.hijri_date ? ` · ${hijriLabel(prayer.hijri_date)}` : ""}</span>
      </div>
    </header>
  );
}
