import { useEffect, useId, useRef, useState } from "react";
import { Info as InfoIcon, X } from "lucide-react";

/** The (i) button: the explanation of one item, in short lines, in a small bubble next to it. Closes on a click
 * outside, on Escape, or with its own close button. */
export default function Info({ lines, label, close }: { lines: readonly string[]; label: string; close: string }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLSpanElement>(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  return (
    <span className="info" ref={box}>
      <button type="button" className="info-btn" aria-label={label} aria-expanded={open} aria-controls={id}
        onClick={() => setOpen(!open)}>
        <InfoIcon size={15} strokeWidth={1.8} aria-hidden="true" />
      </button>
      {open && (
        <span className="info-pop" id={id} role="note">
          {lines.map((l) => <span key={l}>{l}</span>)}
          <button type="button" className="info-x" aria-label={close} onClick={() => setOpen(false)}>
            <X size={14} strokeWidth={1.8} aria-hidden="true" />
          </button>
        </span>
      )}
    </span>
  );
}
