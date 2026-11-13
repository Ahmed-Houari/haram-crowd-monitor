# View-independent crowd metric: research notes (2026-10-24)

Requirement (user, 2026-10-24): count in all hours, day and night, although the broadcast keeps changing cameras,
framing and zoom; research first, build an offline prototype, measure, then go live. This file records the sources,
the design decisions and the measurements.

## 1. Idea

The per-camera metric (density per 10k px of a fixed zone in one reference view) only counts frames that match a
known framing at almost the same zoom, and its numbers cannot be compared between cameras. The new metric is
**people per square metre on the Mataf floor**: for every frame whose camera is known (focal length, rotation,
position in metres), each cell of the crowd model's density map is traced to the floor, so a reading is people per m²
in rings around the Kaaba, the same unit for every camera, framing and zoom.

The camera comes from one of two paths:

1. **ptz**: the broadcast cameras are fixed pan / tilt / zoom cameras. Each one is calibrated once
   (`scripts/calibrate_cameras.py` -> `config/cameras.json`); every later frame of that camera is matched to its
   reference frame with ALIKED + LightGlue, and because a camera that only rotates and zooms gives
   H ~ K_t R K_ref^-1, the frame's focal length and rotation come straight from the homography (`edge/ptz.py`).
   The same relation is the check that a match is real (K_t^-1 H K_ref must be a scaled rotation), and the
   reference focal length is self-calibrated from many such matches (rotating-camera self-calibration).
   **The rotation check alone is not enough.** Drawing each matched camera's Kaaba box on the saved thumbnails
   showed matches that pass it (error 0.06-0.15) with the box up to a Kaaba-width beside the Kaaba, mostly for the
   roof camera (cam5), by day and night, directly and through keyframes: a rotation that fits the distant,
   repetitive arcades is not necessarily the right one. So when the Kaaba is in view (>= 60 % of its projected box
   in the frame, off the graphics) every ptz camera is checked on it: the OWLv2 + SAM 2.1 mask must cover >= 50 % of
   the projected box above 2 m and >= 70 % of the mask must lie inside the projected box; a camera that fails is
   dropped. Without the Kaaba in view only a rotation error <= 0.05 passes (thumbnail replay: those were aligned).
   Keyframes: a checked match (error <= 0.04, 100-400 inliers) keeps its features in memory as one more reference
   of that camera, so matching follows the light from night into day; a keyframe behind a failed check is dropped.
2. **kaaba**: otherwise, a camera fitted to the Kaaba's silhouette (OWLv2 box -> SAM 2.1 mask -> box of the
   Kaaba's size fitted with an occlusion-aware outline cost, `edge/kaaba_geom.py`), accepted only under a strict gate.
   In the live shadow runs this path rarely passes (night 2026-10-24: 0 of 31 attempts; the fit is also slow and
   rationed): the Kaaba is far more useful for calibrating each broadcast camera once and for checking ptz cameras
   than for fitting every frame.

## 2. Geometry of the Kaaba and the Mataf

| Quantity | Value used | Source |
|---|---|---|
| Wall Black Stone corner -> Shami corner (door wall, NE) | 11.68 m | Umm al-Qura University Hajj research centre survey, via [Saudipedia](https://saudipedia.com/en/a-list-of-the-dimensions-of-the-four-sides-of-the-holy-kaaba) |
| Shami -> Western corner (Hijr side, NW) | 9.90 m | same |
| Western -> Yamani corner (SW) | 12.04 m | same |
| Yamani -> Black Stone corner (SE) | 10.18 m | same |
| Model box | 11.86 x 10.04 m (means of opposite walls) | |
| Kiswa height | 14 m, hanging from the roof edge to the Shadharwan | [Saudi Press Agency via Al Jazeera](https://www.aljazeera.com/features/2026/5/25/whats-inside-the-kaaba-and-what-covers-it-the-story-of-the-kiswah) |
| Shadharwan (base) height | ~0.25 m | Arabic reference works (e.g. mawdoo3 / Umm al-Qura figures) |
| Box modelled (black Kiswa) | z 0.25 -> 14.25 m | overall heights published range 13.1-15 m ([Wikipedia](https://en.wikipedia.org/wiki/Kaaba), [Britannica](https://www.britannica.com/topic/Kaaba-shrine-Mecca-Saudi-Arabia)) |
| Check of the height | fitted 13.86-14.46 m on the three reference frames (free height) | this work |
| Door | 2.2 m above the floor, 3.1-3.18 m x 1.71-1.9 m | [Al Jazeera encyclopedia](https://www.ajnet.me/encyclopedia/2015/9/6/الكعبة-المشرفة), Arabic sources |
| Kiswa belt (hizam) | 0.95 m wide, upper third | Al Jazeera, SPA |
| Hijr Ismail wall | 1.31 m high, ~8.44 m from the NW wall | [WikiShia](https://en.wikishia.net/view/Hijr_Isma%27il), Wikipedia |
| Maqam Ibrahim | ~13 m from the door wall | [Saudipedia](https://saudipedia.com/en/maqam-ibrahim) |
| Head height | 1.55 m (density maps put people at their heads) | |

World frame: origin on the floor under the Kaaba's centre, z up, x along the long walls. The Kaaba looks the same
from opposite sides, so a camera and its 180-degree twin give the same floor geometry; the metric does not need to
know which wall is which. Rings: 1-10 m ("inner") and 10-25 m ("outer") from the walls.

## 3. Models and licences

| Component | Model / library | Licence | Use |
|---|---|---|---|
| Crowd density | DM-Count (VGG-19), UCF-QNRF weights | MIT code; weights trained on UCF-QNRF (research dataset) | density map (unchanged) |
| Kaaba box | OWLv2 base patch16 ensemble (google) | Apache-2.0 | text-prompted detection, ~0.1-0.25 s |
| Kaaba mask | SAM 2.1 hiera-large (Meta) | Apache-2.0 | mask from box + point, on a crop |
| Matching | ALIKED-n16 + LightGlue (kornia 0.9) | ALIKED BSD-3-Clause; LightGlue code and weights Apache-2.0; kornia Apache-2.0 | day / night matching to reference frames |
| View classifier | DINOv2-small (Meta) + logistic regression | Apache-2.0 (DINOv2 moved from CC-BY-NC) | Kaaba / Mataf / other |
| Point counter (evaluation only, 2026-10-25) | APGCC (ECCV 2024), VGG16-BN + IFI decoder, ShanghaiTech-A weights | MIT code; weights trained on ShanghaiTech A (research dataset) | HAJJv2 benchmark, zoomed-frame check; not live |
| Libraries | transformers 5.20, scikit-learn, SciPy, OpenCV | Apache-2.0 / BSD-3 / BSD-3 / Apache-2.0 | |

Considered and not used: SuperPoint (Magic Leap licence, non-commercial: LightGlue README); SAM 3 (custom "SAM
Licence", gated download that would need the user to accept terms); Grounding DINO tiny (Apache-2.0, works, slower
than OWLv2 here); DISK + LightGlue (Apache-2.0, tested: much weaker than ALIKED by day); RoMa (MIT, heavier, not
needed once ALIKED + LightGlue matched day frames).

## 4. Literature that shaped the design

- Ground-plane density and camera geometry for crowd counting: Q. Zhang and A. B. Chan, "Wide-Area Crowd Counting via
  Ground-Plane Density Maps and Multi-View Fusion CNNs" (CVPR 2019); Q. Zhang, A. B. Chan, "Calibration-free
  Multi-view Crowd Counting" (ECCV 2022); Q. Zhang et al., "Cross-View Cross-Scene Multi-View Crowd Counting"
  (CVPR 2021).
- Perspective / scale and counting bias: Shi et al., "Revisiting Perspective Information for Efficient Crowd
  Counting" (PACNN, CVPR 2019); Wang et al., "Distribution Matching for Crowd Counting" (DM-Count, NeurIPS 2020).
- Rotating-camera self-calibration: R. Hartley, "Self-Calibration of Stationary Cameras" (IJCV 1997);
  Agapito, Hayman, Reid, "Self-Calibration of Rotating and Zooming Cameras" (IJCV 2001).
- Matching: Lindenberger, Sarlin, Pollefeys, "LightGlue: Local Feature Matching at Light Speed" (ICCV 2023);
  Zhao et al., "ALIKED" (IEEE TIM 2023).
- Segmentation / detection: Ravi et al., "SAM 2" (2024); Minderer et al., "Scaling Open-Vocabulary Object Detection"
  (OWLv2, NeurIPS 2023); Oquab et al., "DINOv2" (TMLR 2024).

## 5. Zoom bias of the crowd model

DM-Count misses small heads. With the camera known, every density cell has a head width in px (0.22 m at its
distance), so the bias is a function of head size: the share of people found is g(h) = 1 - exp(-(h / h0)^k), fitted
on the three calibrated reference frames digitally zoomed 0.8-2.0x, comparing counts on the same 2 m floor tiles
(`scripts/zoom_bias.py` -> `models/zoom_bias.json`). Each cell's count is divided by g(its head width). Digital zoom
adds no detail, so this is a proxy for optical zoom; the shadow run checks it on real zoomed frames.

Live refit (`scripts/zoom_bias_live.py`): the shadow runs store per frame the raw count and floor area per head-size
bin, so any (h0, k) can be replayed on real frames. The digital-zoom fit (h0 3.51 px, k 1.79) left run 5's cameras 53 %
apart (cam1 wide, heads ~5 px, vs cam0 zoomed 1.4-2.4x, heads ~10 px): it under-corrects small heads, because
zooming a picture cannot show people the model already missed. Fitted on one run and checked on the other, the two
runs disagree (few camera switches per run), so the deployed g is the minimax over both runs: h0 6.5 px, k 1.2
(g(3) 0.33, g(5) 0.52, g(8) 0.72, g(12) 0.88). Pooled over runs 4 and 5, different cameras within 5 min: median 16.3 %
(p90 32.4 %, n 174; digital-zoom g 20.0 % / 63.7 %; no correction 92 %); same camera, zoom > 20 % apart: 11.3 % /
38.2 % (15.1 % / 53.1 %; 23.7 % / 68.0 %). On the digital-zoom test it over-corrects instead (max drift 0.8-2.0x:
cam0 23 %, cam1 12 %, cam5 22 %; `data/views/2d/zoom_bias_g.json`). To re-check by day.

## 6. Privacy

Counting only; no faces, no tracking. Frames stay in memory; PTZ keyframes are feature points and descriptors in
memory, never frames. The research wrote only numbers (CSV, JSON, NPY of DINOv2 embeddings), outline-only drawings
(lines on black), and contact sheets of the already stored survey thumbnails (some with the projected Kaaba box drawn
on them) in `data/views/discover/` (the allowed place for discovery sheets).

## 7. Measurements (2026-10-24)

Coverage replay (`scripts/replay_m2.py`): the 257 labelled survey thumbnails (480x270, one every 5 min, 2026-10-23
23:07 -> 10-24 19:22 Makkah; 136 day, 121 night), upscaled to 1080p, in time order. Counted minutes per hour from the
share p of counted frames: between 60p and 60(1-(1-p)^4), middle value.

| variant | day frames | day min/h | night frames | night min/h | hours with no count (of 21) |
|---|---|---|---|---|---|
| current metric (SIFT, visible part of the fixed zone) | 4 | 4.3 | 20 | 20.4 | 9 |
| ptz, rotation check only | 24 | 21.5 | 59 | 42.6 | 2 (but see below) |
| ptz + keyframes, rotation check only | 40 | 31.4 | 65 | 44.7 | 0 (but see below) |
| ptz + Kaaba check | 18 | 17.0 | 47 | 37.5 | 5 |
| **ptz + keyframes + Kaaba check (the live configuration)** | **23** | **20.8** (10.1-31.4) | **48** | **37.9** (23.8-52.1) | **4** (07-11 h) |

The two "rotation check only" rows are not usable: drawn on the thumbnails, a large share of their cameras put the
Kaaba up to a Kaaba-width beside the real one (sheets `data/views/discover/kiswa_*.jpg`). With the Kaaba check the
counted cameras sit on the Kaaba (`replay_m2_day_0.jpg`, `replay_m2_night_*.jpg`); the remaining offsets are <= 1/3
of the Kaaba's width on thumbnails, and on live 1080p frames the check scores are higher (recall 0.88-0.94, precision
0.95-0.99 vs ~0.75-0.9 on thumbnails).

Chained keyframes vs the calibrated reference alone (69 thumbnails where both pass): focal length ratio median 1.001
(p90 3 %), rotation difference median 2.0 deg (p90 4.5 deg), visible ring area median 3.6 % (p90 29 %).

Focal-length sweep with the Kaaba check (`scripts/kaaba_check_calibration.py`, 25 values 0.6-1.8x per camera): cam0
aligned on 100 % of the checked thumbnails for f 1284-1690 px (calibrated 1553), cam1 on 96-100 % for 1878-2058 px
(calibrated 1892), cam5 at most 74 % for any f (plateau 2247-2578, calibrated 2481). The rest of the "cam5" frames
match cam5's distant arcades but not its Kaaba: Kaaba fits put that camera on cam5's line of sight ~95-125 m from the
Kaaba (cam5: 148 m), i.e. probably a second roof camera; it needs its own reference and calibration.

Shadow run 4 (live, 2026-10-24 21:09-21:54 Makkah, 181 frames, before the Kaaba check; `data/shadow/eval_m2_run4.json`):
- coverage 32.6 counted min/hour (current metric 20.9) over 46 night minutes;
- same moment (<= 5 min), different cameras: median 15.2 %, p90 33.7 % (n 131); on the floor both see (45-degree
  sectors around the Kaaba): 15.0 % / 32.1 %; without the head-size correction 93 % / 131 %; current metric 37.3 % /
  41.6 % (n 15; all its accepted frames since 2026-10-23: 35.8 % / 63.0 %, n 908);
- same camera, similar zoom (noise floor, the crowd moves): 7.3 % / 26.3 %; same camera, real optical zoom differing
  by > 20 %: 13.4 % / 37.5 % corrected, 21.5 % / 60.8 % raw (n 187);
- per stage (ms, median / p90): matching 288 / 408, OWLv2 228 / 290, SAM 2.1 568 / 716, DM-Count 132 / 209, Kaaba
  fit 0 / 5335 (rationed), whole shadow frame 1389 / 7630; peak VRAM per frame 2.7 GB (6.6 GB on the first frame);
- fallback (item 6): 94 frames without a camera in 3 DINOv2 framing clusters; scale factors against calibrated
  readings within +-5 min vary with CV 0.39-1.01 (stable would be < 0.2): not used.

View classifier (`scripts/view_classifier.py`, DINOv2-small CLS + logistic regression, 257 labels): 5-fold CV accuracy
0.868 (Kaaba 0.91 F1, Mataf without the Kaaba 0.81, other 0.87); trained on day, tested on night 0.810; night -> day
0.860.

Matching cost vs the number of references (worst case, a frame that matches nothing): 4 refs 65 ms, 16 refs 271 ms,
28 refs 453 ms, 34 refs 611 ms (RTX 5060 at 42 W, beside the collectors).

Shadow run 5 (live, 2026-10-24 21:55-22:55 Makkah, 237 frames, the live configuration: keyframes + Kaaba check;
`data/shadow/eval_m2_run5_ptz.json`, ptz readings only):
- coverage 23.6 counted min/hour (current metric 13.8) over 61 night minutes; 70 ptz readings, 43 through keyframes;
- Kaaba check: 67 passed, 5 rejected, 17 without the Kaaba in view; median recall 0.83, precision 0.92;
- per stage (ms, median / p90): matching 406 / 643, OWLv2 208 / 332, SAM 2.1 382 / 413, DM-Count 111 / 172;
- per-frame Kaaba fits (not used live): 9 accepted, unstable (same camera 45 s apart: 1.24 vs 1.94 people/m²);
- fallback: 158 frames without a camera in 6 clusters, one stable (CV 0.08, 27 frames), the rest CV 0.40-0.79.

Switch-over: crowd collector restarted once at 22:58 with `config/m2.json` "enabled": true, "kaaba_fit": false;
first live minutes cam0 (through a keyframe) 1.22-1.41 people/m², 4 frames per minute as before, no errors.

## 8. Ground truth without hand counting (2026-10-25)

HAJJv2-CrowdCount (arXiv 2607.07322): human whole-frame counts at 1 fps on the 18 HAJJv2 clips (383 frames),
`scripts/eval_hajjv2.py`. Labels from github.com/reem-8899/HAJJv2-CrowdCount (licence "TBC"), videos from the HAJJv2
release (no licence file): local evaluation only, nothing redistributed.

- The labels of the sparse and medium clips count only clearly separate people (Testing_8 t=5: 69 labelled, far more
  visible). Every density or point counter reads 2-4x them, and detectors look best there. Only the dense clip
  (Testing_12 / Training_12: Mataf near the Maqam, ~1700 people, 720x576, heads ~6-7 px) is usable.
- There: DM-Count QNRF -21.5 % / -20.1 % (test / train clip), DM-Count SHA -20.0 % / -16.0 %, APGCC SHHA -20.3 % /
  -20.6 %. APGCC on the 8-clip headline set: MAE 165, bias +123 % (paper: 153, +139 %): the wrapper reproduces it.
- Implied head-size correction at ~6.5 px: g ~ 0.79. Live g (h0 6.5, k 1.2) 0.63; synthetic (3.51, 1.79) 0.95.
  Same form with k 1.2 through this point: h0 ~ 4.5. One SD, compressed clip: a hint, not a refit.
- APGCC (`edge/apgcc_model.py`, MIT; SHHA weights) is no better than DM-Count on the dense clip and misses many
  people on our zoomed cam0 frames (51 vs DM-Count 109 in one sector, many visible people without a point).
  ~115 ms per 1080p frame, <= 1.8 GB VRAM: speed is not the issue, accuracy is.

Self-calibration from zoomed-in frames (`scripts/zoom_calib.py`): a floor patch where heads are >= 10 px wide is
the reference for the same patch in other cameras' frames within +-5 min. Per-sector numbers are logged with every
reading since 2026-10-25 15:31 Makkah (`crowd_m2_frame.sectors`, 12 angles x 2 rings). The first ring-level estimate
(older rows) is cam5 0.75 [0.59-0.88], cam1 0.94 [0.76-1.08], cam1 against its own zoomed frames 1.01. The cam5 factor
moves to ~0.8-0.9 under the HAJJv2-implied g, so the factors and g need a joint fit on sector data (2-3 days).
