// Words for the main dashboard (user 2026-11-01): Arabic first, English on request. One short line per item; the
// explanations sit behind the (i) buttons. People per m² only inside "Details".
import type { Lang } from "../simple/i18n";

export const W = {
  ar: {
    brand: "مرصد حشود المسجد الحرام",
    crowdNow: "المطاف الآن",
    people: "شخص على المطاف",
    about: "نحو",
    live: "مباشر",
    notLive: "غير مباشر",
    offline: "لا اتصال",
    updated: (rel: string) => `آخر تحديث ${rel}`,
    lastAt: (t: string) => `آخر قراءة الساعة ${t}`,
    noReading: "لا توجد قراءة الآن",
    loading: "جارٍ التحميل…",
    twin: "المطاف من الأعلى",
    video: "البث المباشر",
    roads: "الطرق حول الحرم",
    closed: "مغلقة",
    play: "تشغيل البث المباشر",
    trend: "آخر ٦ ساعات",
    sixAgo: "قبل ٦ ساعات",
    now: "الآن",
    details: "التفاصيل",
    switchTo: "English",
    switchLabel: "Switch to English",
    info: "شرح",
    close: "إغلاق",
    sbaNote: "البث من SBA",
    privacy: "عدّ فقط: بلا وجوه ولا هويات ولا تتبّع",
    d: {
      reading: "قراءة المطاف",
      perM2: "شخص/م²",
      fiveMin: "متوسط ٥ دقائق",
      inner: "القريب",
      outer: "الأبعد",
      far: "المطاف البعيد",
      cameras: "الكاميرات",
      peopleRule: (a: string) => `عدد الناس = متوسط ٥ دقائق (شخص/م²) × ${a} م² (الأرض من ١ إلى ٢٥ م من جدران الكعبة)، مقرّب.`,
      keyFlat: "لون مسطّح: مستوى الزحام في ذلك الجزء",
      keyHatch: "مظلَّل: تقدير (خلف الكعبة، جزء لم يُرَ، حساس السرعة)",
      keyNone: "رمادي باهت: لم تره كاميرا في ١٠ دقائق",
    },
    i: {
      crowd: [
        "تقدير لعدد الناس على أرض المطاف حتى ٢٥ متراً من الكعبة، من كل الكاميرات المعايَرة.",
        "متوسط آخر ٥ دقائق، مقرّب.",
        "هادي ← متوسط ← زحمة ← زحمة شديدة. صفوف الصلاة الممتلئة = زحمة.",
      ],
      twin: [
        "المطاف من الأعلى. اللون هو مستوى الزحام في كل جزء.",
        "المظلَّل بخطوط خفيفة تقدير: خلف الكعبة أو جزء لم تره الكاميرا.",
        "الرمادي الباهت: لم تره أي كاميرا في آخر ١٠ دقائق.",
      ],
      video: ["البث الرسمي المباشر من المسجد الحرام.", "يبدأ عند الضغط، بلا صوت."],
      roads: [
        "حركة السيارات حول الحرم من TomTom، تُحدَّث كل ٥ دقائق.",
        "الساحات حول المسجد للمشاة، بلا بيانات سيارات. لون المسجد: مستوى الزحام على المطاف.",
        "للتكبير: Ctrl مع عجلة الفأرة، أو إصبعان على الجوال.",
      ],
      trend: ["الزحام على المطاف في آخر ٦ ساعات: مربع لكل ساعة، لونه مستوى الزحام في تلك الساعة (وسيط قراءاتها كل ٥ دقائق).",
        "الرمادي: لا قراءة في تلك الساعة. المربع الأخير: الساعة الحالية حتى الآن."],
    },
  },
  en: {
    brand: "Haram Crowd Monitor",
    crowdNow: "The Mataf now",
    people: "people on the Mataf",
    about: "about",
    live: "Live",
    notLive: "Not live",
    offline: "No connection",
    updated: (rel: string) => `Updated ${rel}`,
    lastAt: (t: string) => `Last reading at ${t}`,
    noReading: "No reading right now",
    loading: "Loading…",
    twin: "The Mataf from above",
    video: "Live broadcast",
    roads: "Roads around the Haram",
    closed: "Closed",
    play: "Play the live broadcast",
    trend: "Last 6 hours",
    sixAgo: "6 h ago",
    now: "now",
    details: "Details",
    switchTo: "عربي",
    switchLabel: "التبديل إلى العربية",
    info: "About this",
    close: "Close",
    sbaNote: "SBA stream",
    privacy: "Counting only: no faces, no identity, no tracking",
    d: {
      reading: "Mataf reading",
      perM2: "people/m²",
      fiveMin: "5-min mean",
      inner: "inner",
      outer: "outer",
      far: "far Mataf",
      cameras: "Cameras",
      peopleRule: (a: string) => `People on the Mataf = 5-min people/m² × ${a} m² (floor 1–25 m from the Kaaba's walls), rounded.`,
      keyFlat: "flat colour: the level of that floor",
      keyHatch: "hatched: estimated (behind the Kaaba, unseen floor, speed sensor)",
      keyNone: "faint grey: no camera in 10 min",
    },
    i: {
      crowd: [
        "An estimate of the people on the Mataf floor up to 25 m from the Kaaba, from every calibrated camera.",
        "Average of the last 5 minutes, rounded.",
        "Quiet, Moderate, Crowded, Very crowded. Full prayer rows count as Crowded.",
      ],
      twin: [
        "The Mataf seen from above. Colour is the crowd level of each part.",
        "Light hatching is an estimate: behind the Kaaba, or floor no camera saw.",
        "Faint grey: no camera saw it in the last 10 minutes.",
      ],
      video: ["The official live broadcast from Masjid al-Haram.", "Starts when you press play, muted."],
      roads: [
        "Car traffic around the Haram from TomTom, every 5 minutes.",
        "The squares around the mosque are for walking, with no car data. The mosque's colour: the crowd level on the Mataf.",
        "To zoom: Ctrl and the mouse wheel, or two fingers on a phone.",
      ],
      trend: ["The crowd on the Mataf over the last 6 hours: one block per hour, coloured by that hour's level (the median of its 5-minute readings).",
        "Grey: no reading that hour. The last block is the current hour so far."],
    },
  },
} as const satisfies Record<Lang, unknown>;

const NF: Record<Lang, Intl.NumberFormat> = {
  ar: new Intl.NumberFormat("ar-SA-u-nu-arab"),
  en: new Intl.NumberFormat("en-US"),
};
export const fmtNum = (lang: Lang, v: number) => NF[lang].format(v);

/** Full floor 1-25 m from the Kaaba's walls (edge/tawaf_speed.ring_area(1, 25), the headline ring): the reading
 * (people per m², area-weighted over the floor seen) times this is the approximate number of people on it. */
export const RING_AREA_M2 = 3012;

/** People on the Mataf floor, rounded so it does not pretend to be exact: to 50 under 1000, else to 100. */
export function approxPeople(m2: number): number {
  const n = m2 * RING_AREA_M2;
  const step = n < 1000 ? 50 : 100;
  return Math.max(0, Math.round(n / step) * step);
}
