"""Step 2c: click tool to draw zones and the broadcast-graphics mask on each reference frame.

Opens one OpenCV window per reference view (config/views.json). Zones are stored per view in
reference-frame pixels; the graphics mask is shared by all views (the overlay sits at the same
place on every camera), so a mask polygon drawn on any view applies to all of them.

Mouse:  left click = add point      right click = remove last point
Keys:   Enter = finish polygon (ZONE mode: then type its name, Enter again)
        m = switch ZONE / MASK mode   u = undo last finished polygon (current mode)
        x x = clear the whole mask    s = save   n = save + next view   q = save + quit

Run from the project root:  .venv/Scripts/python scripts/draw_zones.py [--view ID]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from edge.view_matcher import PROJECT_ROOT, graphics_mask, load_config, save_config, zones_in_ref_px  # noqa: E402

WIN = "Haram zones"
SAFE_MARGIN = 0.10  # zones inside this margin stay visible when the camera zooms in up to ~20 %
NAME_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_")
GREEN, RED, YELLOW, WHITE, CYAN = (0, 200, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (255, 255, 0)


def draw_overlay(ref: np.ndarray, mask_polys: list, zones: dict, lw: int = 2, grid: bool = True,
                 vertex_numbers: bool = False) -> np.ndarray:
    """Reference frame with graphics mask (red), zones (green), 10 % grid and the cyan safe box."""
    h, w = ref.shape[:2]
    overlay = ref.copy()
    for poly in mask_polys:
        cv2.fillPoly(overlay, [np.int32(poly)], RED)
    for poly in zones.values():
        cv2.fillPoly(overlay, [np.int32(poly)], GREEN)
    img = cv2.addWeighted(overlay, 0.35, ref, 0.65, 0)
    for poly in mask_polys:
        cv2.polylines(img, [np.int32(poly)], True, RED, lw)
    for name, poly in zones.items():
        p = np.int32(poly)
        cv2.polylines(img, [p], True, GREEN, lw)
        cx, cy = p.mean(0).astype(int)
        cv2.putText(img, name, (cx - 60, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 4, cv2.LINE_AA)
        if vertex_numbers:
            for k, (x, y) in enumerate(p, 1):
                cv2.circle(img, (int(x), int(y)), 16, YELLOW, -1)
                cv2.putText(img, str(k), (int(x) - 9 if k < 10 else int(x) - 15, int(y) + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if grid:  # 10 % grid, labels = percent of width (top) and height (left)
        for k in range(1, 10):
            x, y = w * k // 10, h * k // 10
            cv2.line(img, (x, 0), (x, h), (160, 160, 160), 1)
            cv2.line(img, (0, y), (w, y), (160, 160, 160), 1)
            cv2.putText(img, str(10 * k), (x + 4, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
            cv2.putText(img, str(10 * k), (6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    mx, my = int(SAFE_MARGIN * w), int(SAFE_MARGIN * h)
    cv2.rectangle(img, (mx, my), (w - mx, h - my), CYAN, lw)
    return img


class Editor:
    def __init__(self, config: dict, view: dict, ref: np.ndarray, display_w: int, draft: bool = False,
                 msg: str = "Left-click to add points. Enter to finish a polygon."):
        self.config, self.view, self.ref, self.draft = config, view, ref, draft
        self.h, self.w = ref.shape[:2]
        self.s = display_w / self.w
        self.points: list[tuple[int, int]] = []
        self.mode = "ZONE"
        self.typing: str | None = None
        self.msg = msg
        self.clear_armed = False
        self.grid = True

    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.typing is not None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((round(x / self.s), round(y / self.s)))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def handle_key(self, k: int) -> str | None:
        """Returns 'next' or 'quit' when the view is done."""
        if k == 255:
            return None
        if self.typing is not None:
            return self._type_name(k)
        if k != ord("x"):
            self.clear_armed = False
        if k == 13:
            self._finish_polygon()
        elif k == ord("m"):
            self.mode = "MASK" if self.mode == "ZONE" else "ZONE"
            self.msg = f"Mode: {self.mode}"
        elif k == ord("g"):
            self.grid = not self.grid
        elif k == ord("u"):
            self._undo()
        elif k == ord("x"):
            if self.clear_armed:
                self.config["graphics_mask"] = []
                self.msg, self.clear_armed = "Graphics mask cleared.", False
            else:
                self.msg, self.clear_armed = "Press x again to clear the WHOLE graphics mask.", True
        elif k == ord("s"):
            save_config(self.config)
            self.msg = "Saved config/views.json"
        elif k == ord("n"):
            save_config(self.config)
            return "next"
        elif k in (ord("q"), 27):
            save_config(self.config)
            return "quit"
        return None

    def _finish_polygon(self) -> None:
        if len(self.points) < 3:
            self.msg = "Need at least 3 points."
            return
        if self.mode == "MASK":
            self.config["graphics_mask"].append([list(p) for p in self.points])
            self.points = []
            self.msg = "Mask polygon added."
        else:
            self.typing = ""
            self.msg = "Type the zone name (a-z, 0-9, _), Enter to confirm, Esc to cancel."

    def _type_name(self, k: int) -> None:
        if k == 27:
            self.typing, self.msg = None, "Name cancelled; polygon kept, press Enter to try again."
        elif k == 8:
            self.typing = self.typing[:-1]
        elif k == 13:
            name = self.typing
            if not name:
                self.msg = "Name is empty."
                return
            self.view["zones"][name] = [list(p) for p in self.points]
            self.msg = f"Zone '{name}' saved in memory (s to write file). " + self._check_zone(self.points)
            self.points, self.typing = [], None
        else:
            ch = "_" if chr(k) in "-_" else chr(k).lower()
            if ch in NAME_CHARS and len(self.typing) < 30:
                self.typing += ch
        return None

    def _check_zone(self, pts: list[tuple[int, int]]) -> str:
        zmask = np.zeros((self.h, self.w), np.uint8)
        cv2.fillPoly(zmask, [np.int32(pts)], 1)
        overlap = (zmask & graphics_mask(self.config, (self.w, self.h))).sum() / max(zmask.sum(), 1)
        if overlap > 0.01:
            return f"WARNING: {overlap:.0%} under graphics mask -> it will never count. Press u and redraw."
        xs, ys = np.array(pts)[:, 0], np.array(pts)[:, 1]
        m = SAFE_MARGIN
        if xs.min() < m * self.w or xs.max() > (1 - m) * self.w or ys.min() < m * self.h or ys.max() > (1 - m) * self.h:
            return "Note: zone crosses the cyan guide -> dropped when the camera zooms in."
        return "OK."

    def _undo(self) -> None:
        if self.mode == "MASK" and self.config["graphics_mask"]:
            self.config["graphics_mask"].pop()
            self.msg = "Removed last mask polygon."
        elif self.mode == "ZONE" and self.view["zones"]:
            name = list(self.view["zones"])[-1]
            del self.view["zones"][name]
            self.msg = f"Removed zone '{name}'."

    def render(self, idx: int, total: int) -> np.ndarray:
        lw = max(2, round(2 / self.s))
        img = draw_overlay(self.ref, self.config["graphics_mask"], self.view["zones"], lw, self.grid)
        if self.points:
            p = np.int32(self.points)
            cv2.polylines(img, [p], False, YELLOW, lw)
            for q in p:
                cv2.circle(img, tuple(int(v) for v in q), 3 * lw, YELLOW, -1)
        disp = cv2.resize(img, (round(self.w * self.s), round(self.h * self.s)), interpolation=cv2.INTER_AREA)

        top = np.full((34, disp.shape[1], 3), 40, np.uint8)
        draft = "DRAFT on survey thumbnail   " if self.draft else ""
        cv2.putText(top, f"{draft}View {idx}/{total}: {self.view['id']}   MODE: {self.mode}   zones: "
                    f"{', '.join(self.view['zones']) or '-'}   mask polygons: {len(self.config['graphics_mask'])}",
                    (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW if self.mode == "ZONE" else (80, 80, 255), 1, cv2.LINE_AA)
        bottom = np.full((58, disp.shape[1], 3), 40, np.uint8)
        help_line = "L-click add | R-click remove | Enter finish | m mode | u undo | x x clear mask | g grid | s save | n next | q quit"
        cv2.putText(bottom, help_line, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        line2 = f"Zone name: {self.typing}_" if self.typing is not None else self.msg
        cv2.putText(bottom, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv2.LINE_AA)
        return np.vstack([top, disp, bottom])


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw zones + graphics mask on reference frames.")
    parser.add_argument("--view", help="only this view id")
    parser.add_argument("--width", type=int, default=1280, help="display width in px")
    args = parser.parse_args()

    config = load_config()
    config.setdefault("graphics_mask", [])
    views = [v for v in config["views"] if args.view in (None, v["id"])]
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    for i, view in enumerate(views, 1):
        ref = cv2.imread(str(PROJECT_ROOT / view["ref_image"]))
        draft, msg = False, "Left-click to add points. Enter to finish a polygon."
        if ref is None:
            # No full-res reference yet: draft on the upscaled survey thumbnail. capture_refs.py
            # later saves the thumbnail->reference homography and the zones move with it.
            thumb = cv2.imread(str(PROJECT_ROOT / view["source_thumb"]))
            if thumb is None:
                print(f"Skipping {view['id']}: no reference image and no survey thumbnail.")
                continue
            ref = cv2.resize(thumb, tuple(config["ref_size"]), interpolation=cv2.INTER_CUBIC)
            view["zones_base"] = "thumb"
            draft = True
        elif view.get("zones_base") == "thumb":
            view["zones"] = zones_in_ref_px(view, config)
            view["zones_base"] = "ref"
            msg = "Zones moved from the thumbnail draft onto the full-res reference. Check them."
        ed = Editor(config, view, ref, args.width, draft, msg)
        cv2.setMouseCallback(WIN, ed.on_mouse)
        action = None
        while action is None:
            cv2.imshow(WIN, ed.render(i, len(views)))
            action = ed.handle_key(cv2.waitKey(30) & 0xFF)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                save_config(config)
                action = "quit"
        print(f"{view['id']}: zones = {list(view['zones'])}")
        if action == "quit":
            break
    cv2.destroyAllWindows()
    save_config(config)
    print(f"Saved config/views.json  (graphics mask polygons: {len(config['graphics_mask'])})")


if __name__ == "__main__":
    main()
