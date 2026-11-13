// The colour key in plain words (user 2026-11-03): the four level colours and "striped = estimate", in a corner of the
// main dashboard's twin and on /simple (without the stripe line, user 2026-11-03: nothing there is striped). The
// colours are the floor's level hues (api CROWD_LEVELS).
import { CROWD_LEVELS } from "../api";
import { LEVEL_WORD, T, type Lang } from "./i18n";
import "./levelkey.css";

export default function LevelKey({ lang, estimate = true }: { lang: Lang; estimate?: boolean }) {
  return (
    <ul className="lvkey" aria-label={T[lang].keyLabel}>
      {CROWD_LEVELS.map((l) => (
        <li key={l.name}><i style={{ background: `rgb(${l.rgb.join(",")})` }} />{LEVEL_WORD[lang][l.name]}</li>
      ))}
      {estimate && <li><i className="est" />{T[lang].striped}</li>}
    </ul>
  );
}
