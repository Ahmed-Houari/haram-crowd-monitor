import { useEffect, useRef, useState, type RefObject } from "react";

/** True while a media query matches (live). */
export function useMedia(query: string): boolean {
  const [on, setOn] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const m = matchMedia(query);
    const f = () => setOn(m.matches);
    m.addEventListener("change", f);
    f();
    return () => m.removeEventListener("change", f);
  }, [query]);
  return on;
}

export const useReducedMotion = () => useMedia("(prefers-reduced-motion: reduce)");

/** Current time, refreshed every `ms`. */
export function useNow(ms: number): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(t);
  }, [ms]);
  return now;
}

/** Smooth count-up / tween to `target` (ease-out quart); jumps straight there with reduced motion. */
export function useTween(target: number | null, reduce: boolean, ms = 1400): number | null {
  const [shown, setShown] = useState<number | null>(reduce ? target : target == null ? null : 0);
  const from = useRef(shown ?? 0);
  useEffect(() => {
    if (target == null) return setShown(null);
    if (reduce) return setShown(target);
    const start = performance.now();
    const a = from.current;
    let raf = 0;
    const step = (t: number) => {
      const k = Math.min(1, (t - start) / ms);
      const v = a + (target - a) * (1 - Math.pow(1 - k, 4));
      from.current = v;
      setShown(v);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, reduce, ms]);
  return shown;
}

/** True while the element is on screen and the tab is visible (pauses the 3D render loop otherwise). */
export function useOnScreen(ref: RefObject<Element | null>): boolean {
  const [inView, setInView] = useState(true);
  const [shown, setShown] = useState(document.visibilityState === "visible");
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { rootMargin: "80px" });
    io.observe(el);
    const vis = () => setShown(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", vis);
    return () => {
      io.disconnect();
      document.removeEventListener("visibilitychange", vis);
    };
  }, [ref]);
  return inView && shown;
}

/** Element width in CSS px (ResizeObserver). */
export function useWidth(ref: RefObject<Element | null>): number {
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}
