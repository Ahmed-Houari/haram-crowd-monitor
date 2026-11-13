import { CROWD_LEVELS, LEVELS_HELP } from "../api";

/** The crowd-safety levels as a legend: four equal segments (quiet fades from empty floor), the boundaries in people
 * per m² under them and the level names below. Used by the twin's legend and the AI view. */
export default function LevelScale({ unit = true }: { unit?: boolean }) {
  return (
    <div className="lvscale" title={LEVELS_HELP}>
      <div className="bar">{CROWD_LEVELS.map((l) => <i key={l.name} className={l.name} />)}</div>
      <div className="ticks">
        {CROWD_LEVELS.map((l, i) => <em key={l.name}>{i ? CROWD_LEVELS[i - 1].upTo.toFixed(1) : ""}</em>)}
      </div>
      <div className="words">{CROWD_LEVELS.map((l) => <em key={l.name} className={l.name}>{l.name}</em>)}</div>
      {unit && <small>people per m² · Fruin walkway levels</small>}
    </div>
  );
}
