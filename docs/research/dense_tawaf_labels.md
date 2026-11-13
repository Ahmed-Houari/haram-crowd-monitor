# Dense tawaf labels (> 2 people/m²): test plan

User request 2026-10-29 ~23:30 Makkah: find reliable labels for dense tawaf, pick 2-3 methods, test them on "start",
one overlay per method. Research only while the services are stopped (no GPU). No training, no deploy.

## Problem

18:46 Makkah, packed tawaf (user: clearly > 2.2): cam0 inner ring reads v3 1.44, original DM-Count 1.07
(`data/jobs/dense_test.out`). Close-ups after Isha (13-15 px heads): v3 1.5-1.9. Literature puts tawaf near the Kaaba
at 6-8 people/m² in peak season, ~3.7 mid-Mataf, ~2 at 20-30 m (Haj Research Centre figures via secondary sources;
microscopic tawaf models). Our cameras see heads at 6-8 px, where density models under-count: on HAJJv2 Testing_12
(dense Mataf, ~1700 people, trusted labels) DM-Count QNRF, DM-Count SHA and APGCC all read about -20 %.

## Reference sets (no hand counts)

1. **HAJJv2 Testing_12 + Training_12** (45 frames, dense Mataf, ~6-7 px heads, trusted labels, local only): the
   one absolute check. A method must get near 0 % bias here before its labels are believed.
2. **Our dense frames**: right grab 18:43-18:49 Makkah (66 calibrated cam0 frames, 6-8 px) and the registered
   close-ups 20:15-20:16 (13-15 px), in data/kept_frames/recordings/screen_20261029T145041Z_DISPLAY1 (local).
3. **Zoom consistency**: the same floor patch counted at 1x, 2x, 3x upscale. A trustworthy count stops changing
   (plateau); an under-counting one keeps rising.

## Methods (most promising first)

### A. Zoom-in tiling with our DM-Count (no download)

Cut the inner ring into tiles, upscale each 2x / 3x so heads reach ~12-20 px, run DM-Count (QNRF original and v3),
sum. Literature: zooming dense small-head regions is the standard fix for small-head under-counting (Learn to
Scale, ZoomCount, plug-and-play rescaling, GigaZoom). Test: bias on Testing_12 at 1x / 2x / 3x, plateau on our
frames, then m² density for 18:46. Risk: over-count on big heads (only tile where head px < ~10).

### B. CLIP-EBC (newer dense-crowd model, download needed)

Blockwise-classification counter on CLIP (ICME 2025), best NWPU-Crowd results among models with released weights
(ViT-B/16 MAE 61.3; STEERER 63.7, APGCC 71.7, PET 74.4). MIT licence, weights on the GitHub release page, sliding
window built in. Test: Testing_12 bias, our dense frames, zoom consistency, speed / VRAM. Same author's newer ZIP
(2025, QNRF MAE 69.4) has no clear code licence: only if CLIP-EBC looks good and its licence checks out.
**Needs approval:** download of the CLIP-EBC weights (GitHub release, size to be checked), own folder, untrusted
(`python -I`, `weights_only=True`).

### C. Head detector on close-ups (download needed)

A detector trained on crowd heads (RPEE-Heads YOLOv9 / RT-DETR, mAP ~91 % on crowd-video heads; or a CrowdHuman
YOLOv8 head model) counts boxes on close-ups with heads >= 12 px. That is an independent count, not a density map.
Its count on a placed close-up gives people per m² for that patch, and so a reference for A and B at the same
moment (like `scripts/zoom_calib.py`). Head boxes only: no faces, no identity, no tracking; boxes become numbers,
nothing is kept. Not usable at 6-8 px. **Needs approval:** model download (RPEE-Heads DOI 10.34735/ped.2024.2, or
the CrowdHuman YOLOv8 Google Drive), and your OK that head detection for counting fits the privacy rule.

Not picked: APGCC (already tested, -20 % on Testing_12, misses people on zoomed cam0), YOLO-World / SAM3Count
(HAJJv2 paper: fail on dense frames), STEERER / PET (no better than CLIP-EBC, older).

## On "start" (one job at a time on the GPU)

1. A on Testing_12 / Training_12 (1x / 2x / 3x), then on our 66 + close-up frames. ~10 min.
2. B, if approved: same sets. ~15 min incl. setup.
3. C, if approved: close-ups only. ~10 min.
4. One overlay per method on the 18:46 frame (C on a 20:15 close-up): density map or head points over the frame,
   inner-ring people/m², Testing_12 bias in the caption. Local files only.
5. Short table: Testing_12 bias, 18:46 inner density, zoom plateau yes / no, ms per frame, VRAM.

## Results (run 2026-10-30, one GPU job at a time, live services running alongside)

Same tiling for every method: 384 px source tiles + 32 px margin, enlarged 1x / 2x / 3x, margin cut, summed back onto
8 x 8 px cells of the original frame. Our frames: cam0, inner ring 1-10 m, raw people per m^2 (no head-size
correction); 18:43-18:49 Makkah, 66 calibrated frames, heads ~6.6 px. Testing_12: 19 frames, GT mean 1699.

| Method | Testing_12 bias 1x / 2x / 3x | 18:46 inner raw /m^2 1x / 2x / 3x | Plateau 2x-3x | ms per 1920x1080 frame 1x / 2x / 3x | VRAM peak |
|---|---|---|---|---|---|
| A original DM-Count QNRF, tiled | -19.5 / **-4.5** / -3.8 % | 0.82 / 0.94 / 0.90 | yes | 143 / 575 / 1298 | 2.0 GB |
| A v3 (qnrf_ft3), tiled | -1.3 / +18 / +25 % | 1.40 / 1.51 / 1.38 | no (over-counts) | 140 / 584 / 1382 | 2.0 GB |
| B CLIP-EBC (NWPU) | -28.8 / -40.5 / -47.4 % | 0.73 / 0.49 / 0.42 | no (falls with zoom) | 756 / 3543 / 7158 | 1.9 GB |
| C head detector (RPEE-Heads YOLOv8x) | -57 / -74 / -85 % | 0.07 / 0.03 / 0.03 | no | 268 / 595 / 1192 | 0.6 GB |

Training_12 (26 frames) agrees: A original 2x -4.7 %, v3 1x +0.9 %, B 1x -25.7 %, C 1x -53.6 %. Close-ups with
heads >= 12 px (20:15-20:16, 6 frames): A original 1.07-1.27, v3 live 1.30-1.44, C 0.17-0.31 people/m^2.

- **A is the most accurate on HAJJv2:** the original DM-Count at 2x tiles reads -4.5 % on Testing_12 and the count
  plateaus between 2x and 3x, so at 2x the model is no longer missing small heads.
- **v3 should not be tiled:** its head-size correction is built in, so zooming corrects twice (+18 % at 2x, +25 % at
  3x). Untiled v3 is already near zero on Testing_12 (-1.3 %).
- **B fails:** under-counts dense heads and gets worse with zoom. The release asset named `ViT_B_16` holds ViT-L/14
  weights (patch 14, width 1024, 24 layers), so its exact training settings are uncertain; either way it is not usable.
- **C fails:** a detector does not separate packed, overlapping heads. On the 20:14 close-up it found 23 heads in a
  ring that holds a few hundred. Not usable as a dense reference.
- **Likely cause of the low 18:46 reading** (A plateaus at ~0.9-1.0 /m^2, below v3 live 1.42 and below the expected
  > 2.2): not head size, since zoom no longer changes the count. cam0's low angle hides heads behind the people in front,
  which no zoom recovers (HAJJv2 is filmed from higher up), and the 1-10 m ring includes the partly empty Hijr Ismail
  floor, which pulls the ring average down.

Safety: CLIP-EBC's `models/clip/_clip/__init__.py` auto-downloads OpenAI CLIP jit pickles on import; it was bypassed
and the checkpoint loaded with `weights_only=True`, strict. The YOLOv8x file is a full ultralytics pickle: read with a
restricted unpickler (no code from the file runs) and run with a forward pass written for the test. Head centres stayed
in memory and became counts; no faces, boxes or crops were written. Weights in models/clip_ebc (MIT) and
models/rpee_heads (CC BY-SA 4.0), gitignored; per-frame numbers in data/jobs/dense_eval_{A,B,C}.out (local).

## Sources

- HAJJv2 zero-shot benchmark: https://arxiv.org/html/2607.07322v1
- CLIP-EBC: https://arxiv.org/html/2403.09281, https://github.com/Yiming-M/CLIP-EBC
- ZIP: https://arxiv.org/html/2506.19955
- Zoom-in counting: https://arxiv.org/pdf/1907.12428, https://arxiv.org/pdf/2002.12256, https://arxiv.org/pdf/2001.01786
- Head detection: https://arxiv.org/html/2411.18164v1, https://ped.fz-juelich.de/da/doku.php?id=rpee_heads, https://github.com/Owen718/Head-Detection-Yolov8
- Tawaf densities: https://arxiv.org/pdf/1501.06496, https://informs-sim.org/wsc13papers/includes/files/020.pdf
