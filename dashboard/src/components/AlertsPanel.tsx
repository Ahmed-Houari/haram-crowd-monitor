import { Car, Clock, Users } from "lucide-react";
import { makkahTime, type AlertItem, type Alerts } from "../api";

const TITLE: Record<AlertItem["kind"], string> = {
  crowd_now: "Crowded now",
  crowd_forecast: "Crowded within 30 min",
  road: "Road congestion",
};
const ICON = { crowd_now: Users, crowd_forecast: Clock, road: Car };
const RULE_LABEL: Record<string, string> = { crowd_now: "Crowded now", crowd_forecast: "Forecast", road: "Roads" };

/** Open alerts first, then the last 24 h; below, each rule and whether it can fire yet. */
export default function AlertsPanel({ alerts, sample }: { alerts: Alerts | null; sample: boolean }) {
  const items = alerts?.items ?? [];
  const open = items.filter((a) => a.active).length;
  return (
    <div className="pane alerts">
      <h3>
        Alerts <span className="ar" lang="ar">التنبيهات</span>
        <small>{open ? `${open} open` : "none open"}{sample ? " · sample" : ""}</small>
      </h3>
      {items.length ? (
        <ul className="alert-list">
          {items.slice(0, 5).map((a) => {
            const Icon = ICON[a.kind] ?? Users;
            return (
              <li key={a.id} className={`alert ${a.severity}${a.active ? " on" : ""}`}>
                <Icon aria-hidden="true" />
                <div>
                  <b>{TITLE[a.kind] ?? a.kind}</b>
                  <span>{a.message}</span>
                  <small>
                    {a.active ? `open since ${makkahTime(a.start_utc)}` : `${makkahTime(a.start_utc)} – ${makkahTime(a.end_utc)}`}
                    {a.reopened ? ` · came back ${a.reopened}x within the ${alerts?.cooldown_min ?? 30}-min cooldown` : ""}
                  </small>
                </div>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="alert-empty">No alerts in the last 24 h.</p>
      )}
      {alerts && (
        <ul className="rules">
          {Object.entries(alerts.rules).map(([k, r]) => (
            <li key={k} className={r.ready ? "ready" : "wait"}>
              <i aria-hidden="true" />
              <b>{RULE_LABEL[k] ?? k}</b> {r.note}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
