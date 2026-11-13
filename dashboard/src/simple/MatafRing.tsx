// The answer drawn as the place itself: the Mataf seen from above, its floor ring filled in the crowd level's colour
// around the Kaaba (Kiswa gold band, Hijr Ismail's arc). Grey when there is no live reading. Geometry only.

interface Props {
  colour: string | null; // the level's colour; null = no live reading
  label: string;
}

const C = 150; // centre of the 300 x 300 view

export default function MatafRing({ colour, label }: Props) {
  const fill = colour ?? "#3a4749";
  const live = colour != null;
  return (
    <svg className={`ring${live ? " live" : ""}`} viewBox="0 0 300 300" role="img" aria-label={label}>
      <defs>
        <radialGradient id="floor" cx="50%" cy="50%" r="50%">
          <stop offset="18%" stopColor={fill} stopOpacity={live ? 1 : 0.55} />
          <stop offset="86%" stopColor={fill} stopOpacity={live ? 0.78 : 0.4} />
          <stop offset="100%" stopColor={fill} stopOpacity={live ? 0.5 : 0.3} />
        </radialGradient>
        <radialGradient id="pool" cx="50%" cy="50%" r="50%">
          <stop offset="55%" stopColor={fill} stopOpacity={live ? 0.28 : 0} />
          <stop offset="100%" stopColor={fill} stopOpacity="0" />
        </radialGradient>
      </defs>
      {/* the light the crowd's colour throws on the portico */}
      <circle cx={C} cy={C} r={148} fill="url(#pool)" className="pool" />
      {/* portico edge and the floor ring */}
      <circle cx={C} cy={C} r={128} fill="none" stroke="rgba(236,231,220,.22)" strokeWidth="1" />
      <circle cx={C} cy={C} r={118} fill="url(#floor)" />
      {/* tawaf lanes: faint rings in the floor */}
      {[52, 70, 88, 104].map((r) => (
        <circle key={r} cx={C} cy={C} r={r} fill="none" stroke="rgba(10,19,21,.28)" strokeWidth="1" strokeDasharray="2 5" />
      ))}
      {/* the Kaaba from above, its Kiswa band in gold, and Hijr Ismail: the low semicircular wall off one side */}
      <g transform={`rotate(-14 ${C} ${C})`}>
        <path d={`M ${C - 25} ${C - 15} A 15 17 0 0 0 ${C - 25} ${C + 15}`} fill="rgba(236,231,220,.06)" stroke="#ece7dc"
          strokeOpacity=".6" strokeWidth="2" strokeLinecap="round" />
        <rect x={C - 21} y={C - 18} width="42" height="36" rx="1.5" fill="#07090a" stroke="rgba(10,19,21,.9)" strokeWidth="2" />
        <rect x={C - 16} y={C - 13} width="32" height="26" fill="none" stroke="#c8a45c" strokeWidth="2.4" />
      </g>
    </svg>
  );
}
