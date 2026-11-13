// Words for the visitor page (/simple): Arabic first, English on request. Crowd levels are words, never numbers.
import type { CrowdLevel } from "../api";

export type Lang = "ar" | "en";

export const LEVEL_WORD: Record<Lang, Record<CrowdLevel, string>> = {
  ar: { quiet: "هادي", busy: "متوسط", crowded: "زحمة", packed: "زحمة شديدة" },
  en: { quiet: "Quiet", busy: "Moderate", crowded: "Crowded", packed: "Very crowded" },
};

export const PRAYER: Record<Lang, Record<string, string>> = {
  ar: { fajr: "الفجر", sunrise: "الشروق", dhuhr: "الظهر", asr: "العصر", maghrib: "المغرب", isha: "العشاء" },
  en: { fajr: "Fajr", sunrise: "Sunrise", dhuhr: "Dhuhr", asr: "Asr", maghrib: "Maghrib", isha: "Isha" },
};

export const T = {
  ar: {
    title: "المطاف الآن",
    switchTo: "English",
    switchLabel: "Switch to English",
    noLive: "لا توجد قراءة مباشرة الآن",
    lastReading: (t: string) => `آخر قراءة الساعة ${t}`,
    loading: "جارٍ التحميل…",
    nextPrayer: "الصلاة القادمة",
    inTime: "بعد",
    expected: "المتوقع",
    notEnough: "لا يكفي السجل بعد",
    bestTitle: "أفضل وقت للذهاب",
    lightToday: "أخف وقت باقي اليوم",
    lightTomorrow: "أخف وقت بكرة",
    noneLeft: "لم يبقَ وقت كافٍ",
    busyDay: "المطاف مزدحم أغلب اليوم",
    roads: "الطرق حول الحرم",
    free: "سالكة",
    slow: "بطيئة",
    jam: "مزدحمة",
    roadsOld: "حالة الطرق غير محدّثة",
    updated: (rel: string) => `آخر تحديث ${rel}`,
    never: "لا توجد قراءات بعد",
    sample: "بيانات تجريبية للعرض فقط",
    privacy: "عدّ فقط: بلا وجوه ولا هويات ولا تتبّع",
    estimateNote: "من قراءات الأيام الماضية منذ ١ نوفمبر، خارج أوقات الصلاة",
    forecast: "توقّع",
    fcIn: { 30: "بعد ٣٠ دقيقة", 60: "بعد ساعة" } as Record<number, string>,
    keyLabel: "دليل الألوان",
    striped: "مخطط = تقدير",
  },
  en: {
    title: "The Mataf now",
    switchTo: "عربي",
    switchLabel: "التبديل إلى العربية",
    noLive: "No live reading right now",
    lastReading: (t: string) => `Last reading at ${t}`,
    loading: "Loading…",
    nextPrayer: "Next prayer",
    inTime: "in",
    expected: "Expected",
    notEnough: "Not enough history yet",
    bestTitle: "Best time to go",
    lightToday: "Lightest time left today",
    lightTomorrow: "Lightest time tomorrow",
    noneLeft: "not enough of today left",
    busyDay: "The Mataf is crowded most of the day",
    roads: "Roads around the Haram",
    free: "Clear",
    slow: "Slow",
    jam: "Jammed",
    roadsOld: "Road status is out of date",
    updated: (rel: string) => `Updated ${rel}`,
    never: "No readings yet",
    sample: "Sample data, for display only",
    privacy: "Counting only: no faces, no identity, no tracking",
    estimateNote: "from past days' readings since 1 Nov, outside prayer times",
    forecast: "Forecast",
    fcIn: { 30: "In 30 min", 60: "In 1 hour" } as Record<number, string>,
    keyLabel: "Colour key",
    striped: "Striped = estimate",
  },
} as const;

const LOCALE: Record<Lang, string> = { ar: "ar-SA-u-nu-arab-ca-gregory", en: "en-GB" };

/** "٣:٢١ م" / "3:21 pm", Makkah time. */
export function clock(lang: Lang, t: number | string, withMinutes = true): string {
  return new Intl.DateTimeFormat(LOCALE[lang], {
    timeZone: "Asia/Riyadh", hour: "numeric", ...(withMinutes ? { minute: "2-digit" } : {}), hour12: true,
  }).format(new Date(t));
}

/** "من ٣ إلى ٥ م" / "3–5 pm", "من ١٢:٣٠ إلى ٢:٣٠ م": Makkah time, minutes only off the hour, the day period once
 * when both ends share it. */
export function hourSpan(lang: Lang, a: number, b: number): string {
  const f = new Intl.DateTimeFormat(LOCALE[lang], { timeZone: "Asia/Riyadh", hour: "numeric", minute: "2-digit", hour12: true });
  const at = (t: number) => {
    const p = f.formatToParts(new Date(t));
    const h = p.find((x) => x.type === "hour")!.value, m = p.find((x) => x.type === "minute")!.value;
    return { h: new Date(t).getUTCMinutes() ? `${h}:${m}` : h, d: p.find((x) => x.type === "dayPeriod")?.value ?? "" };
  };
  const x = at(a), y = at(b);
  if (lang === "ar") return x.d === y.d ? `من ${x.h} إلى ${y.h} ${y.d}` : `من ${x.h} ${x.d} إلى ${y.h} ${y.d}`;
  return x.d === y.d ? `${x.h}–${y.h} ${y.d}` : `${x.h} ${x.d} – ${y.h} ${y.d}`;
}

/** "قبل ٣ دقائق" / "3 minutes ago". */
export function relative(lang: Lang, iso: string, now: number): string {
  const min = Math.max(0, Math.round((now - Date.parse(iso)) / 60000));
  const rtf = new Intl.RelativeTimeFormat(lang === "ar" ? "ar-u-nu-arab" : "en", { numeric: "auto" });
  if (min < 1) return rtf.format(0, "minute");
  if (min < 90) return rtf.format(-min, "minute");
  if (min < 36 * 60) return rtf.format(-Math.round(min / 60), "hour");
  return rtf.format(-Math.round(min / 1440), "day");
}

/** Countdown "١:٠٤:٢٢" / "1:04:22" (hours only when needed). */
export function countdown(lang: Lang, ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const nf = new Intl.NumberFormat(lang === "ar" ? "ar-u-nu-arab" : "en", { minimumIntegerDigits: 2, useGrouping: false });
  const n1 = new Intl.NumberFormat(lang === "ar" ? "ar-u-nu-arab" : "en", { useGrouping: false });
  return h ? `${n1.format(h)}:${nf.format(m)}:${nf.format(sec)}` : `${n1.format(m)}:${nf.format(sec)}`;
}
