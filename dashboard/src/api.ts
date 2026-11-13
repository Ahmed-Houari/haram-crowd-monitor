// Types + data access for the FastAPI backend (backend/app.py). All times are UTC ISO strings.

export interface Status {
  running: boolean;
  /** collector frames < 3 min, shot service frame < 3 min and Mataf reading < 15 min old (backend FRESH_*_MIN) */
  fresh: boolean;
  /** what is not recent: "collector frames", "shot service", "Mataf readings"; null when fresh */
  stale_reason?: string[] | null;
  last_frame_utc: string | null;
  last_reading_utc: string | null;
  last_shot_frame_utc?: string | null;
  last_mataf_utc?: string | null;
  frames_today: number;
  accepted_today: Record<string, number>;
  uptime_today_pct: number | null;
  last_traffic_poll_utc: string | null;
  yt_dlp_lookups_today: number;
  waiting_for_reference: string[];
  server_time_utc: string;
}

/** Schematic placement of a camera in the 3D Mataf twin (config/twin.json). */
export interface TwinCam {
  label: string;
  label_ar: string;
  color: string;
  ang: number;
  r: number;
  h: number;
  hfov: number;
  near: number;
  far: number;
  aim: number;
  placed?: boolean; // false: a new camera the API put on a placeholder spot (the twin moves it onto the gallery)
}

export interface View {
  id: string;
  quality: "ok" | "low" | "untested" | null;
  quality_note: string | null;
  zoom_tolerance: number;
  zoom_range?: [number, number];
  zones: string[];
  has_reference: boolean;
  has_heatmap: boolean;
  status?: "approved" | "pending";
  label?: string | null;
  references?: { id: string; period?: string; status: string; quality?: string; preview?: string }[];
  twin: TwinCam | null;
  /** a calibrated camera only the people-per-m² metric uses (config/cameras.json, no per-camera zone) */
  metric_only?: boolean;
}

/**
 * Crowd numbers (since 2026-10-24): `density` = people per 10,000 px (a 100 x 100 px patch) of the VISIBLE part of the
 * camera's floor zone, measured in its fixed reference view, so zoomed or partly visible shots stay comparable: the
 * main value. `visible` = visible fraction of the zone; `count` = people in the whole zone, only when it was fully
 * visible (else null); `count_visible` = people in the visible part.
 */
export interface CrowdLatest {
  view_id: string;
  zone: string;
  minute_utc: string;
  source: string;
  density: number | null;
  visible?: number;
  count: number | null;
  count_visible?: number;
  zoom: number | null;
  n_frames: number;
}

/** quiet / busy / crowded vs this camera's own history, or calibrating until enough days are collected. */
export interface Level {
  state: "calibrating" | "quiet" | "busy" | "crowded";
  days: number;
  need_days: number;
  buckets: number;
  need_buckets: number;
  pcts: [number, number];
  history_days: number;
  low?: number;
  high?: number;
  rank?: number;
}

/** Latest 5-min average vs the same time of day on earlier days (needs a week of data). */
export interface Usual {
  state: "waiting" | "no_history_at_time" | "ok";
  days: number;
  need_days: number;
  usual?: number;
  pct?: number | null;
  days_at_time?: number;
}

export interface CrowdNow {
  view_id: string;
  zone: string;
  source: string;
  minute_utc: string;
  density: number | null;
  /** missing = an API from before the visible-part metric (levels, trend and spark are counts there) */
  visible?: number;
  count: number | null;
  count_visible?: number;
  /** this camera's own people per m² (view-independent metric), null while it has none */
  people_m2?: number | null;
  people_m2_utc?: string | null;
  density_5min?: number;
  trend: { ref_utc: string; ref_density: number; pct: number } | null;
  spark: [string, number][];
  level: Level;
  usual: Usual | null;
}

/** "What the AI sees" for one camera (backend/aiview.py), in heatmap pixel coordinates. */
export interface AiView {
  view_id: string;
  ts: string | null;
  size: [number, number];
  zones: Record<string, [number, number][]>;
  label: [number, number];
  outline: string | null;
  density?: number | null;
  visible?: number | null;
  count: number | null;
  level: string;
  k: number;
  heat_url: string;
  sample: boolean;
  /** the floor and the Kaaba behind the heat (calibrated cameras); the heat is then absolute people per m² */
  bg_url?: string | null;
  absolute?: boolean;
  /** the camera's own people per m²: the heat's colour scale follows it (same scale as the twin) */
  people_m2?: number | null;
  people_m2_utc?: string | null;
  /** people_m2 is this heatmap's average over the metric's ring, 1-25 m from the Kaaba's walls */
  people_m2_ring?: boolean;
  /** this heatmap's far Mataf (25 m to the courtyard's edge), and how sure it is ("approximate") */
  people_m2_far?: number | null;
  far_confidence?: string | null;
  /** how the live shot service placed this frame (scripts/live_shot.py): calibrated / registered / approximate */
  method?: string | null;
  how?: string | null;
  /** the current shot without a known camera: relative colours, approximate people per m² */
  approximate?: boolean;
}

/** The shot on air as the live shot service last analysed it (every 1-2 s). */
export interface Shot {
  ts: string | null;
  method: "calibrated" | "registered" | "approximate" | "none";
  how: string;
  camera: string | null;
  people_m2: number | null;
  rows: number | null;
  age_s: number | null;
}

export interface CrowdBucket {
  bucket_utc: string;
  view_id: string;
  zone: string;
  source: string;
  density: number | null;
  visible?: number;
  count: number | null;
  n_frames: number;
}

export interface TrafficArea {
  area: "r1km" | "r3km";
  road_category: string;
  n_segments: number;
  length_km: number;
  mean_level: number | null;
  congested_pct: number | null;
  closed_pct: number | null;
}

export interface TrafficLatest {
  poll_utc: string | null;
  areas: TrafficArea[];
}

export interface TrafficPoint {
  poll_utc: string;
  area: "r1km" | "r3km";
  mean_level: number | null;
  congested_pct: number | null;
}

export interface PrayerTime {
  utc: string;
  makkah: string;
}

export interface PrayerToday {
  date: string;
  hijri_date?: string;
  times: Record<string, PrayerTime> | null;
  next: { name: string; utc: string; minutes: number } | null;
  prev: { name: string; utc: string; minutes: number } | null;
}

export interface PrayerEvent {
  name: string;
  utc: string;
  makkah: string;
}

/** Forecast per camera (edge/forecast.py); available once a model is trained on real data. */
export interface ForecastPoint {
  h: number;
  utc: string;
  density: number;
}

export interface ForecastCam {
  view_id: string;
  zone: string;
  issued_utc: string;
  threshold_density: number | null;
  predictions: ForecastPoint[];
  crowded_within_30: boolean;
}

export interface Forecast {
  available: boolean;
  reason?: string;
  trained_utc?: string;
  cameras?: ForecastCam[];
}

/** Alert engine output (edge/alerts.py). */
export interface AlertItem {
  id: string;
  kind: "crowd_now" | "crowd_forecast" | "road";
  key: string;
  severity: "watch" | "high";
  place: string;
  message: string;
  start_utc: string;
  last_on_utc: string;
  end_utc: string | null;
  value: number | null;
  peak: number | null;
  threshold: number | null;
  reopened: number;
  active: boolean;
}

export interface Alerts {
  items: AlertItem[];
  rules: Record<string, { ready: boolean; note: string }>;
  cooldown_min: number;
}

/** View-independent metric (2026-10-24): people per m² on the Mataf floor around the Kaaba, all cameras together. */
export interface MatafNow {
  minute_utc: string;
  people_m2: number;
  rings: Record<string, number>; // "all", "inner" (1-10 m from the Kaaba's walls), "outer" (10-25 m)
  area_m2: number;
  n_frames: number;
  cameras: string[]; // reference camera ids, "kaaba" = a camera fitted to the Kaaba
  density_5min: number;
  trend: { ref_utc: string; ref_density: number; pct: number } | null;
  spark: [string, number][];
  level: Level;
  usual: Usual | null;
  /** people per m² per floor sector ("i0".."o11": 12 angles x inner / outer ring), pooled over the last minutes;
   * only sectors with enough floor seen (the twin shows the ring average for the others) */
  sectors?: Record<string, { people_m2: number; area_m2: number; frames: number }>;
  /** each camera's own people per m² from its latest frames */
  by_camera?: Record<string, { people_m2: number; minute_utc: string; n_frames: number }>;
  sector_window_min?: number;
  /** far Mataf: 25 m from the Kaaba's walls out to the courtyard's edge, pooled over the last minutes; a separate,
   * approximate value (not in people_m2, the rings or the forecast) */
  far?: { people_m2: number; minute_utc: string; minutes: number; n_frames: number; area_m2: number;
    confidence: string; from_m: number } | null;
  /** the reading's source since v4 (2026-10-30): "v4" (count model) or "v4+speed"; null before */
  model?: string | null;
  /** the tawaf speed sensor's last minute (backend get_speed): per ring the reading, the count model's value, the
   * walking speed and the shares of walking / estimated / stopped sectors; null when no sector walked */
  speed?: {
    ts: string; camera: string; pairs: number; used: boolean; sector_deg: number;
    rings: Record<string, { value: number; count: number; speed_ms: number | null; walking: number; slow: number;
      fast: number; estimated: number; stopped: number }>;
  } | null;
}

export interface MatafBucket {
  bucket_utc: string;
  ring: string;
  people_m2: number | null;
  area_m2: number;
  n_frames: number;
}

/** The merged top view of the Mataf floor (backend/floormap.py, /api/mataf/floor). */
export interface FloorMeta {
  key: string;
  png: string;
  extent_m: number;
  cell_m: number;
  fresh_min: number;
  max_age_min: number;
  cameras: { view_id: string; ts: string; age_min: number; cells: number; mean_people_m2: number | null; p95_people_m2: number | null }[];
  /** twin-local (x, z) m where the floor behind the Kaaba (estimated, hatched) is; null when none is hidden */
  hidden_at?: [number, number] | null;
}

export interface Snapshot {
  type: "snapshot";
  /** the calibrated camera the live broadcast shows now (identified in the last 40 s), null for another shot */
  on_air?: { view_id: string | null; method?: string; last_identified: { view_id: string; ts: string } | null };
  /** the current shot (live shot service), null when it does not run */
  shot?: Shot | null;
  /** the merged floor map's metadata; its texture URL changes with every new map */
  floor?: FloorMeta;
  status: Status;
  crowd: CrowdLatest[];
  /** view id -> time of its latest live heatmap ("What the AI sees" reloads when it changes) */
  heatmaps?: Record<string, string | null>;
  crowd_now?: CrowdNow[];
  mataf_now?: MatafNow | null;
  forecast?: Forecast;
  alerts?: Alerts;
  traffic: TrafficLatest;
  prayer: PrayerToday;
}

export interface Meta {
  youtube_id: string | null;
  /** SBA's HLS stream of the same broadcast (the player's fallback) and how long YouTube is still paused (s) */
  sba_hls?: string | null;
  youtube_paused_s?: number;
  haram: { lat: number; lon: number };
}

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

/** Live messages over the WebSocket (snapshots + "reading" events), reconnecting with backoff (1 s doubling to 30 s). */
export function connectLive(
  onSnapshot: (s: Snapshot) => void,
  onReading: (now: CrowdNow[], mataf?: MatafNow | null) => void,
  onState: (connected: boolean) => void,
): () => void {
  let ws: WebSocket | null = null;
  let stopped = false;
  let delay = 1000;
  let timer: number | undefined;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => {
      delay = 1000;
      onState(true);
    };
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "snapshot") onSnapshot(msg as Snapshot);
      else if (msg.type === "reading") onReading(msg.crowd_now as CrowdNow[], msg.mataf_now as MatafNow | null | undefined);
    };
    ws.onclose = () => {
      onState(false);
      if (!stopped) timer = window.setTimeout(open, (delay = Math.min(delay * 2, 30000)));
    };
  };
  open();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
    ws?.close();
  };
}

const makkahFmt = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Riyadh", hour: "2-digit", minute: "2-digit" });
const makkahWeekday = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Riyadh", weekday: "long" });

/** "14:05" in Makkah time. */
export const makkahTime = (iso: string | number | null | undefined) =>
  iso == null ? "–" : makkahFmt.format(new Date(iso));
export const makkahDay = (t: number) => makkahWeekday.format(new Date(t));

/** "3 min ago" / "2.1 h ago". */
export function ago(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "never";
  const min = (now - new Date(iso).getTime()) / 60000;
  if (min < 1) return "just now";
  if (min < 120) return `${Math.round(min)} min ago`;
  if (min < 48 * 60) return `${(min / 60).toFixed(1)} h ago`;
  return `${Math.round(min / 1440)} days ago`;
}

export const minutesSince = (iso: string | null | undefined, now = Date.now()) =>
  iso ? (now - new Date(iso).getTime()) / 60000 : Infinity;

const HIJRI_MONTHS = ["Muharram", "Safar", "Rabi al-Awwal", "Rabi al-Thani", "Jumada al-Ula", "Jumada al-Akhirah",
  "Rajab", "Sha'ban", "Ramadan", "Shawwal", "Dhu al-Qadah", "Dhu al-Hijjah"];

/** "1448-05-13" -> "13 Jumada al-Ula 1448 AH". */
export function hijriLabel(h: string | undefined): string {
  const m = h?.match(/^(\d+)-(\d+)-(\d+)$/);
  if (!m) return "";
  return `${+m[3]} ${HIJRI_MONTHS[+m[2] - 1] ?? m[2]} ${m[1]} AH`;
}

export const fmtInt = (v: number) => Math.round(v).toLocaleString("en-US");
/** Density (people per 100 x 100 px of floor): one decimal. */
export const fmtDensity = (v: number) => v.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
export const DENSITY_UNIT = "people per 100 × 100 px of floor";
export const DENSITY_HELP =
  "Density in the visible part of this camera's floor zone: people per 100 × 100-pixel patch of floor, measured in the " +
  "camera's fixed reference view, so zoomed-in or partly visible shots stay comparable. Compare it with this camera's " +
  "own history, not with other cameras (each sees the floor at its own angle).";
/** People per m² on the Mataf floor: two decimals. */
/**
 * Crowd-safety levels in people per m² (user decision 2026-10-27): Fruin's walkway level of service (Fruin 1971,
 * Pedestrian Planning and Design; FHWA-RD-98-107 Table 3): quiet < 0.7 (LOS A-C), busy 0.7-1.1 (D), crowded 1.1-2.2
 * (E), packed from 2.2 (F). Full prayer rows are ~1.3 (1.22 m x 0.64 m per person). The dashboard's one scale (twin,
 * AI view, map, legends; backend/aiview.py LEVELS and colours). Display only: the readings themselves are unchanged.
 */
export const CROWD_LEVELS = [
  { name: "quiet", upTo: 0.7, rgb: [47, 174, 156] },
  { name: "busy", upTo: 1.1, rgb: [230, 192, 79] },
  { name: "crowded", upTo: 2.2, rgb: [242, 106, 61] },
  { name: "packed", upTo: Infinity, rgb: [196, 31, 107] },
] as const;
export type CrowdLevel = (typeof CROWD_LEVELS)[number]["name"];
export const LEVELS_HELP = "Crowd-safety levels from Fruin's walkway level of service: quiet below 0.7, busy 0.7–1.1, " +
  "crowded 1.1–2.2, packed from 2.2 people per m². Full prayer rows are about 1.3 per m² (crowded).";
const EMPTY_RGB = [36, 90, 90];
/** The level of a people-per-m² value. */
export const crowdLevel = (v: number): CrowdLevel => CROWD_LEVELS.find((l) => v < l.upTo)!.name;
/** The level's colour (quiet fades from the empty-floor colour to teal), as CSS rgb(); none when there is no value. */
export function crowdColour(v: number | null, none = "#56656a"): string {
  if (v == null) return none;
  const i = CROWD_LEVELS.findIndex((l) => v < l.upTo);
  const c = i ? CROWD_LEVELS[i].rgb : EMPTY_RGB.map((e, j) => e + (CROWD_LEVELS[0].rgb[j] - e) * Math.max(0, v) / 0.7);
  return `rgb(${c.map((x) => Math.round(x)).join(",")})`;
}
export const fmtM2 = (v: number) => v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const M2_UNIT = "people per m² of floor";
export const M2_HELP =
  "People per square metre of the Mataf floor around the Kaaba (1 to 25 m from its walls), measured from every " +
  "camera whose position is known (fitted once to the Kaaba, whose size is known): the same unit for every camera, " +
  "zoom and framing, corrected for the counting model missing very small heads.";
/** The zone counts as fully visible from 99 % (people in the whole zone are shown only then). */
export const FULL_VISIBLE = 0.99;

export const PRAYER_EN: Record<string, string> = {
  fajr: "Fajr", sunrise: "Sunrise", dhuhr: "Dhuhr", asr: "Asr", maghrib: "Maghrib", isha: "Isha",
};
export const PRAYER_AR: Record<string, string> = {
  fajr: "الفجر", sunrise: "الشروق", dhuhr: "الظهر", asr: "العصر", maghrib: "المغرب", isha: "العشاء",
};

/** Fallback names when config/twin.json has no entry for a view. */
export const VIEW_LABELS: Record<string, string> = {
  cam0_mataf_low: "Low camera",
  cam1_mataf_wide: "Wide camera",
  cam5_roof: "Roof camera",
};
export const viewLabel = (v: View | undefined, id: string) => v?.twin?.label ?? VIEW_LABELS[id] ?? id;
