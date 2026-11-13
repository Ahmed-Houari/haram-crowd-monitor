import { fmtM2, makkahTime, minutesSince, viewLabel, type CrowdNow, type View } from "../api";

interface Props {
  views: View[];
  crowd: CrowdNow[];
  /** each camera's own people per m², where it has one */
  m2: Record<string, { v: number; at: string }>;
  heroId: string | null;
  onPick: (id: string | null) => void;
  now: number;
  staleMin: number;
}

/** One chip per counting camera; picking one shows its number in the hero (cameras are not comparable). */
export default function CameraChips({ views, crowd, m2, heroId, onPick, now, staleMin }: Props) {
  return (
    <div className="camchips" role="group" aria-label="Counting cameras">
      {views.map((v) => {
        const c = crowd.find((x) => x.view_id === v.id);
        const own = m2[v.id];
        const stale = own != null && minutesSince(own.at, now) > staleMin;
        const color = c || own ? (v.twin?.color ?? "#ff6f52") : "#56656a";
        return (
          <button key={v.id} type="button" className="chip" disabled={!c} aria-pressed={c ? heroId === v.id : undefined}
            onClick={() => onPick(heroId === v.id ? null : v.id)}
            title={c ? `Show the ${viewLabel(v, v.id).toLowerCase()} in the big number` : v.quality_note ?? undefined}>
            <i style={{ background: color }} />
            <small>{viewLabel(v, v.id)}</small>
            {own ? <><b>{fmtM2(own.v)}</b><small className="u">/m²</small></> : c ? <small className="none">no m² reading yet</small> : null}
            {own && stale && <small>at {makkahTime(own.at)}</small>}
            {c && v.quality === "low" && <span className="q">lower quality</span>}
            {!c && !own && <small>{v.status === "pending" ? (v.has_reference ? "new camera · pending review" : "new camera · waiting for reference")
              : v.has_reference ? "no reading yet" : "waiting for reference"}</small>}
          </button>
        );
      })}
    </div>
  );
}
