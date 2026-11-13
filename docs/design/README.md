# Dashboard redesign — research and directions (2026-10-23)

Status: 3 directions mocked up, waiting for the user's pick. Mockups: `docs/design/mockups/`
(`a-night`, `b-marble`, `c-mawaqit`; HTML + PNG). Crowd numbers in the mockups are labelled
samples (no reference view accepted yet); road, prayer and status data are a real snapshot.

## 1. Tools reused from EnerTwin

| Area | EnerTwin | Reuse here |
|---|---|---|
| App stack | React 19 + Vite + TypeScript, Vitest | same (already) |
| Charts | recharts | same (already) |
| Icons | lucide-react | add in the build |
| 3D | Three.js via @react-three/fiber + drei (building twin) | not needed now (a 3D Mataf could come later) |
| Fonts | Archivo variable, self-hosted (`public/fonts`, OFL), wide light tabular numerals | same approach (self-hosted variable font, tabular numerals), different face |
| Design system | `DESIGN.md` (named tokens, OKLCH tonal ramps, one meaning per signal colour, Do's/Don'ts) | write `DESIGN.md` for the chosen direction |
| Motion | CSS keyframes, `prefers-reduced-motion` | same |
| Verification | Playwright screenshots at real sizes | `playwright` + installed Chrome (used for these mockups) |

Method: Operate mode (a working screen: the world lends type, palette,
density and ONE signature move; layout and controls stay familiar), the old look is an
anti-reference, craft floor (no card grids, no gradient text, no glass decoration, no eyebrow
labels, themed scrollbars/focus/selection, tabular numerals, dark/light chosen from the use scene).

## 2. References (real systems and design work)

1. Grand Mosque smart crowd system, Gulf News — live gate in/out counts, density and bottleneck flags for field teams: what Haram operators actually watch. https://gulfnews.com/world/gulf/saudi/smart-crowd-system-goes-live-at-makkahs-grand-mosque-ahead-of-ramadan-1.500413192
2. Grand Mosque crowd technology in Ramadan, Arab News — reporting on the central engineering command centre (density and flow data feeding risk levels). https://www.arabnews.pk/node/2636582/saudi-arabia
3. Armored Things at the LAFC stadium, Stadia Magazine — live density and flow on one dashboard, alerts when a space overcrowds. https://stadia-magazine.com/?p=10893
4. Automsoft, 90,000-seat stadium — density heat map compared against recommended capacity, bottlenecks flagged live. https://www.automsoft.com/?p=591
5. FootfallCam public venues — live occupancy by area, rule-based alerts, drill-down into hotspots. https://www.footfallcam.com/Industries/PublicVenue
6. IndoorAtlas GroundSage — crowd density as area / geofence heat maps for operations teams. https://support.indooratlas.com/support/solutions/articles/36000295467-groundsage-dashboard
7. Crowdcom (Dutch event crowd management) — one shared live map as the control room's common picture; incidents posted on the same map. https://crowdcom.nl
8. Fuselab Creative, smart-city digital twin — data sits where it happens (congestion on its street), layered map, detail only on focus. https://fuselabcreative.com/get-inspired/smart-city-digital-twin-dashboard/
9. Dribbble concept work — "Traffic Monitoring" for a smart-city platform (rooman_v) and "Live Stadium Monitoring" (Fuselab): map-led, alert-first, dark. https://dribbble.com/rooman_v · https://dribbble.com/fuselabcreative
10. Situation-awareness guidance — Stephen Few, "Designing dashboards for situation awareness"; Activu SOC dashboard checklist: top-level indicators first, drill-down, design for subtraction, colour only for meaning. https://www.perceptualedge.com/blog/?p=45 · https://www.activu.com/security-operations-center-dashboard-best-practices-a-checklist-for-critical-situational-awareness/

## 3. Common patterns

- **Layout:** the map or the site plan is the centre; a thin status strip on top; live feeds in a side column; details only on focus.
- **Live status:** one unmistakable state (live / stale / down) with the time of the last update; counts against capacity; bottlenecks flagged.
- **Alerts:** few, rule-based, tied to a place and a start time, newest first, severity colour; alert fatigue is the known failure.
- **Map:** muted, desaturated basemap so the data colours carry the meaning; data drawn where it happens; layers toggle.
- **Colour:** neutral ground, one brand accent, red/amber/green reserved for state only.
- **Dense data:** summary first, detail on demand, tables and small multiples with tabular numerals; remove before adding.
- **Dark vs light:** control rooms run dark (dim rooms, long shifts, screens on all night); public and office views run light.

## 4. Directions (mockups in `mockups/`)

**A · Layl al-Haram (night).** Deep teal-black ground, marble-white text, Kiswa gold only for the
brand mark and "now". Signature: the *prayer horizon*, a rolling 24 h band with the five prayers,
a now-line, recent crowd intensity and the forecast zone ahead. Map-led layout. Alexandria +
Readex Pro (Arabic and Latin). Inspired by 2, 7, 8, 10. Best for: a 24/7 control-room screen.

**B · Marble (daylight).** Warm marble-white ground, Haram-green header, light map. Signature: the
*Mataf plan*, a schematic top-down plan where each camera's counted zone is a sector coloured by
density, plus a prayer bar. Readex Pro + Reem Kufi. Inspired by 1, 4, 5, 6. Best for: daytime and
public/portfolio viewing.

**C · Mawaqit board.** Deep green prayer-board panels, signage type (Overpass + Noto Kufi Arabic),
big numerals. Signature: the *prayer-period board*, today's periods as rows (crowd per camera, road
congestion), the current period live, the next row reserved for the forecast. Inspired by 3, 10
and mosque prayer-time boards. Best for: reading the day the way the forecast will (by prayer).

All three keep every current element (status, live broadcast, crowd by camera, roads map, road
charts, prayer times, privacy note) and reserve places for alerts and the forecast. All use the
OpenFreeMap vector basemap (free, no API key, no usage limits, attribution) restyled per direction,
with identical TomTom state colours. The TomTom basemap was rejected: it would put the API key in
the browser. CARTO now needs an API key.

## 5. Chosen: A2 "Layl al-Haram + Mataf twin" (user pick 2026-10-24, revised mockup)

Mockup: `mockups/a2-twin.html` (`a2-twin-desktop.png`, `a2-twin-mobile.png`; motion preview in
`data/design/a2-twin-motion.mp4`, not committed). User upgrades over A:
1. Hero = stylized 3D Mataf (schematic): plinth, Mataf floor with rings and a tick dial, portico
   ring, Kaaba with the Kiswa-gold band, Hijr Ismail; each counted zone is a floor sector glowing by
   density (teal → gold → coral, radial glow texture + coloured light pool); camera markers with view
   lines and labels. Slow sway, glow breathing; static with reduced motion.
2. Map: OpenFreeMap dark, free-flowing roads muted grey-teal (#3e5a5c, thinner, no casing); only
   slow (amber) and congested (red) stand out.
3. Biggest thing on the page: the current crowd number (Alexandria 200, 112 px desktop / 76 px
   mobile), trend vs 30 min ago, camera + time of the last reading, pulse on a new reading.
4. Prayer horizon: rolling 20 h back / 4 h ahead, crowd area curve, five prayers in English +
   Arabic (Reem Kufi), now-line, forecast cone (placeholder until step 7).
5. Life: count-up/tween on numbers, pulse on new reading, LIVE heartbeat (ripple + ECG line);
   `prefers-reduced-motion` turns all of it off.
6. Bilingual identity: Reem Kufi for Arabic labels, Alexandria (numerals, headings) + Readex Pro
   (UI); faint 8-fold star lattice in the header only (7 % opacity, masked); gold = accent only.
7. "What the camera sees" (YouTube) next to "What the AI sees" (latest density-only heatmap PNG).
8. Mobile: number first, twin below (350 px), camera list, camera/AI side by side, horizon scrolls
   and opens at "now".

Build notes: react-three-fiber + drei for the twin (lazy chunk); MapLibre + OpenFreeMap (lazy
chunk); fonts self-hosted (@fontsource-variable); API additions needed: latest reading per camera
with the value 30 min earlier (trend) and a live "new reading" event on the WebSocket; stagger
prayer labels that collide (Maghrib/Isha on phones); zone sectors are schematic angles per camera
(config), not real geometry.

## 6. Build (step 6b, 2026-10-24)

Built in `dashboard/src` (screens: `screens/a2-build-desktop.png`, `a2-build-mobile.png`, `a2-build-kaaba.png`;
sample crowd data, labelled on the page). Additions over the mockup, from the user:

1. **Meaning for the number, kept honest.** The label says "people in the counted zone" and a note says it is
   one camera's floor zone, not the whole Mataf. A level badge (quiet / busy / crowded) compares the latest
   5-min average with that camera's own last 28 days (lowest 40 % / top 20 %); it shows "Calibrating · x of 3
   days" (with a small progress ring) until there is enough history. Under it: a 2 h sparkline of the
   per-minute readings (gaps = camera off air; the faint band = that camera's busy range) and, after 7 days,
   "X % above/below usual for this time" (median of earlier days at the same time). Before that the line says
   when the comparison will start.
2. **Kaaba details, quiet.** Real proportions (12.9 x 11.0 x 13.1 m), gold Kiswa band two thirds up with thin
   rims, the gold door raised above the floor near the Black Stone corner, the low open-ended Hijr Ismail arc on
   the north-west face, Maqam Ibrahim in front of the door (1.5x scale so it reads). No glow, no labels.
3. **Prayer labels never collide.** Each label goes into the first row where it clears its neighbour (text
   width measured with the real fonts); labels near the right edge flip left; on phones the band opens at "now"
   without cutting a label. Same idea for the small road chart.
4. Camera chips pick which camera the big number shows (default: freshest reading from a good-quality camera).
   Twin footprints follow the data: heat strength = the level, no heat until a camera has a reading, dashed
   outline while a camera has no reference frame. Camera labels avoid the number block.
5. Fixed in the port: the mockup's heat texture read intensity from the red channel (always 255 with additive
   white blobs), which caused contour artifacts; it now reads alpha.

## 7. Readable AI view + realistic Kaaba (2026-10-24, after the first real reading)

**What the AI sees.** The density map now sits on thin outlines of the building traced from each camera's
reference frame (lines only; people areas removed first, see the README's Privacy), in the dashboard's own colour
ramp with a quiet / busy / crowded scale, the zone drawn dashed in the camera's colour and the count written on
the zone. Honest caveat shown under the scale: the collector scales every heatmap to its own busiest spot, so
until a camera is calibrated the colours are relative within the frame; after that their strength follows the
camera's level (same rule as the twin). A true per-pixel scale needs the collector to save its scale factor
(one line in `scripts/collect_crowd.py`, later, with a collector restart).

**Kaaba, Hijr Ismail, Maqam Ibrahim.** No free 3D model with a clear CC0 / CC-BY licence was found (Free3D's
Kaaba is "editorial only"; a Sketchfab model lists no licence and needs a login; CGTrader is paid), so it is
built procedurally, all textures drawn on canvases at load (no downloads):
- Kaaba 12.86 x 11.03 x 13.1 m; Kiswa as black silk (sheen) with an abstract woven black-on-black pattern;
  Hizam in the upper third of the walls: gold panels (16 in all, 4 per wall) with roundels, abstract ornament that
  suggests calligraphy from a distance. User decision 2026-10-24: no Quran text and no real text anywhere on
  the model (the door curtain is ornament only too).
- Door on the north-east wall near the Black Stone corner: the gold door (3.1 x 1.9 m, foot 2.13 m above the
  floor) under its gold curtain (Sitara, ~5.75 x 3.5 m, Wikipedia average), drawn fully gold, hanging upwards from
  the door's foot.
- Black Stone set into the east corner at ~1.5 m, its silver frame wrapping the corner flush with both walls;
  Shadharwan (sloping marble base) on three walls, not on the Hijr side.
- Hijr Ismail: white marble semicircle, 1.32 m high, 0.9 m thick, 8.44 m from the wall at its middle, open
  ~2.3 m at both ends.
- Maqam Ibrahim ~11 m in front of the door: marble base, gilded brass cage (0.8 m across) around a crystal
  cover, small dome with a crescent; drawn 1.5x so it reads at dashboard size.
- White marble Mataf (dimmed a little so the heat keeps its contrast), soft studio environment light, one
  floodlight with soft shadows (shadow map rendered once).
Sources: Wikipedia (Kaaba, Black Stone, Hijr Ismail, Maqam Ibrahim), Saudipedia (Maqam Ibrahim, Hijr Ismail),
WikiShia (Hijr Isma'il), Al Jazeera "What's inside the Kaaba and what covers it?" (2026), WAM (Kiswa pieces),
Wikipedia "Sitara (textile)" (door curtain size).

## 8. Performance guard, framing, the Haram around the Mataf, a readable crowd (2026-10-25; the architecture and the hex tiles are superseded by §9, the performance guard and the framing stay)

Screens: `screens/twin-arch-desktop.png`, `screens/twin-arch-phone.png`.

**Performance.** The twin renders on demand from a capped clock: 30 fps on desktop, 20 on phones (the motion is a
slow sway, so 60+ added nothing), nothing at all when it is scrolled fully off screen or the tab is hidden, and one
frame when it comes back. Pixel ratio capped at 1.75 (1.5 on phones), shadow map 2048 rendered once per change.
Measured with the laptop's GPU (`node dashboard/scripts/perf.mjs`, Chrome with D3D11 on the RTX 5060, the crowd
collector running): 30.0 fps on screen, 0 off screen, 0 with the tab hidden; the live video beside it played
1,201 frames in 40 s with 0 dropped.

**Framing.** The camera is fitted to the whole model and every camera marker with its label, at both ends of the
sway: on desktop in the free area right of the number block and above the camera chips, on phones in a 440 px strip.
Both look along the courtyard's long axis so the model stands upright in the taller-than-wide free areas.

**The Haram around the Mataf** (`dashboard/src/components/haramArch.ts`, clean massing, no text or ornament):
- Courtyard 164 x 110 m with the Kaaba in the middle, its sides parallel to the Kaaba's walls; the Ottoman enclosure
  was about 192 x 132 m (Archnet, Masjid al-Haram).
- The Ottoman riwaq: 14 m deep (three aisles), on three sides of the courtyard but not on the Mas'a side, as rebuilt
  after the 2013-2016 Mataf expansion (Madain Project, King Abdullah expansion); pointed arches on slender piers
  every 5.2 m, warm light inside, a small dome on a drum over every bay (240 domes, instanced).
- The Saudi Mataf building around it, open to the courtyard on every level: ground floor and first floor (arcades,
  9.5 m and 7.5 m), a mezzanine set back 2.5 m (paired openings), the roof at 21 m with a parapet and an 18 m roof
  ring (the tawaf terrace on the roof), the outer roof a step higher; levels as reported for the Mataf expansion
  (ground floor, first floor, mezzanines, roof; Saudi Press Agency, "Mataf area ... 107,000 circumambulators per
  hour"). Storey heights are typical values (schematic).
- Night palette: muted warm stone, the court-facing arcades lightest, the outer mass dark, the marble court dark so
  the crowd is the brightest thing; one key light from the viewer's side, soft studio fill.

**The crowd.** The pastel blob is gone. Hexagonal tiles (1.75 m) cover the counted Mataf, 1-25 m from the Kaaba's
walls, coloured on a fixed scale of people per m² (crowd-safety levels since 2026-10-27, see below; before 0 to 4+) by the live view-independent
metric: the inner ring's value up to 8 m, the outer ring's from 12 m, a blend between; a ring nobody saw this minute
takes the all-floor average. Tiles inside a live camera's view are bright and stand a little higher with density;
the rest fade towards the floor ("ring average"). Without the metric, each live camera's footprint is tiled by that
camera's level. Footprints are thin outlines in the camera's colour (dashed while a camera waits for its
reference), clipped to the courtyard.

**Cameras.** Markers stand at the calibrated cameras' distances and heights (`config/twin.json`: low camera 62 m /
15 m on the first-floor gallery, wide camera 84 m / 26 m on the roof; the roof camera is really 148 m out and 87 m
up, drawn on the building's outer edge at 34 m); their side of the Kaaba is a choice (the calibration frame is
symmetric under a half turn). New cameras without a place go to the free spot on the gallery farthest from the
others. Labels avoid the number block and each other; phones label only cameras with a reading.

## 9. The round Mataf from reference photos; live AI view; one unit (2026-10-26)

User feedback on §8: the rectangular courtyard with rows of domes looked like a stadium; the earlier round twin was
better. The twin is rebuilt from real photos, not invented.

**References** (downloaded to `data/refs/`, gitignored, not committed; `data/refs/sources.json`), plus our own broadcast
frames (`data/samples/m2_*`, roof / wide / low cameras):
- `ref1_hajj2025_a.jpg`: [Pilgrims begin this year's Hajj rituals with a circumambulation of the Kaaba (2025)](https://commons.wikimedia.org/wiki/File:Pilgrims_begin_this_year%27s_Hajj_rituals_with_a_circumambulation_of_the_Kaaba_(2025).jpg), CC BY-SA 4.0
- `ref2_hajj2025_b.jpg`: [same series (2)](https://commons.wikimedia.org/wiki/File:Pilgrims_begin_this_year%27s_Hajj_rituals_with_a_circumambulation_of_the_Kaaba_(2025)_(2).jpg), CC BY-SA 4.0
- `ref3_2019.jpg`: [Great Mosque of Mecca1](https://commons.wikimedia.org/wiki/File:Great_Mosque_of_Mecca1.jpg) (aerial, 2019, Saudipics.com), CC BY-SA 4.0
- `ref4_hajj2018.jpg`: [The Kaaba during Hajj - edited](https://commons.wikimedia.org/wiki/File:The_Kaaba_during_Hajj_-_edited.jpg) (Adli Wahid, 2018), CC BY-SA 4.0
- `ref5_tawaf2018_a.jpg`, `ref6_tawaf2018_b.jpg`: [Tawaf around Kaaba 01](https://commons.wikimedia.org/wiki/File:Tawaf_around_Kaaba_01.jpg), [02](https://commons.wikimedia.org/wiki/File:Tawaf_around_Kaaba_02.jpg) (Mohammad Hossein Velayati, 2018), CC BY 4.0
- `ref8_2015.jpg`: [Kaaba, Masjid Al-Haram, Mecca - panoramio](https://commons.wikimedia.org/wiki/File:Kaaba,_Masjid_Al-Haram,_Mecca,_Saudi_Arabia_-_panoramio.jpg) (2015, the temporary Mataf bridge), CC BY-SA 3.0
- `ref9_iss2015.jpg`: [ISS-44 Mecca, Saudi Arabia](https://commons.wikimedia.org/wiki/File:ISS-44_Mecca,_Saudi_Arabia.jpg) (NASA, top-down, too coarse to use), public domain

**What they show, and the model** (`dashboard/src/components/matafArch.ts`): an open, round white-marble floor around
the Kaaba (r 62 m to the gallery face); a ring of white galleries open to it, large round-headed arches on the ground
and first floor (48 bays) with warm light inside, a band of smaller paired openings above (set back 2 m); the flat roof
ring at 23.5 m with a parapet and slim light poles along its inner edge, faint concentric rows on it (the prayer-mat
arcs of the aerial view); a lower terrace beyond, its outside dark so the ring sits in the night rather than reading as
a freestanding arena. The Ottoman arcade with its small domes is left out (user decision). No text on the model.

Iterations (screenshot next to `ref3` and the roof camera frame; side by side in `data/refs/compare/`):
1. Dark drum + two tiers of heavy arches read as a Roman amphitheatre: lighter stone, slimmer piers, no closed drum.
2. Light roof ring and outer terrace overpowered the floor: tighter ring (roof to 78 m, terrace to 88 m), darker roof.
3. Pale outer wall still read as an arena: the outside goes dark; labels only for cameras with a reading.

**Crowd.** 12 x 2 ring wedges around the Kaaba (12 angles, 1-10 m and 10-25 m from its walls: the metric's floor
sectors, `crowd_m2_frame.sectors`), coloured by people per m² on the crowd-safety levels (see below; before 0 to 4+): opaque
where cameras saw enough of that sector in the last 10 minutes (`/api/mataf/now` "sectors"), faded at the ring's
average otherwise, grey where nothing is recent. Legend under the twin. Sector angles use the same half-turn choice
of the Kaaba frame as the camera positions.

**What the AI sees.** Follows the camera with the newest heatmap (or the picked one), reloads when a new one lands
(the snapshot carries each heatmap's time) and every minute, cross-fades, and says its age ("roof camera · 00:15 ·
12 min ago"; older than 15 min: "no new counted frame for ..."). The heat is smoothed on the server (3x upscale,
Gaussian) and coloured on the twin's people-per-m² scale when the camera has its own reading (the zone's average takes
that value; approximate, per image pixel); the building lines are thinner and fainter.

**One unit.** People per m² wherever it exists: the camera chips and a picked camera's big number show the camera's
own people per m² (`by_camera`, `/api/crowd/now` "people_m2"), or say "no m² reading yet" in plain words; the AI view's
label likewise; the prayer horizon keeps the Mataf series while a camera is picked. No pixel unit is shown.

**Caching.** `index.html` is served with `Cache-Control: no-cache` (a normal reload shows a new build); hashed assets as
immutable. **Phones:** the camera and AI panes stack full width; the twin strip is 390 px.
**Performance** (laptop GPU, collector running, live API): 30 fps on screen, 0 off screen and with the tab hidden, live
video 1,200 frames in 40 s with 0 dropped.

## 10. The road map explains the empty area around the Haram (2026-10-26; the walking area is completed in §11)

Screens: `screens/map-pedestrian-desktop.png`, `screens/map-pedestrian-phone.png`. The map keeps its style.

- **Pedestrian zone.** The squares around the Haram have no car traffic, so TomTom has nothing to colour there. Their
  real outlines from OpenStreetMap (relations 9331006 Western Haram Square and 9331009 North Haram Square, tagged
  highway=pedestrian + place=square; buildings inside them are holes) get a subtle warm fill and a thin edge, and one
  small label, "Pedestrian zone – no car traffic", placed in the open part of the largest square and running west,
  off the mosque. Data: `dashboard/public/osm/haram_area.geojson`, made by `scripts/fetch_osm_haram.py` (OSM API,
  ODbL; the map's attribution credits OpenStreetMap). Squares on the south (Ajyad) and east sides are not mapped as
  areas in OSM yet, so they are not shaded.
- **The Haram's own footprint** (relation 1472531, building=mosque; the Mataf courtyard is its hole) takes the colour of
  the Mataf's live people per m² on the crowd-safety levels (the colour of its level), grey when there is no recent
  reading. It replaces the gold dot. Legend: "Haram: 1.26 people/m² · crowded" with the level's colour.
- **Tunnels** (vector tiles, brunnel=tunnel) are thin grey dashes under the TomTom lines: where TomTom has data the
  coloured line covers them, elsewhere they stay visible. Legend: "tunnel, no data".
- The legend is two rows on desktop and a compact block at the bottom left on phones (it was hidden there before).

## 11. A useful twin: where the crowd really is; leaner GPU; readable AI view; all walking areas (2026-10-26)

Screens: `screens/twin-floor-vs-ai.png` (the twin's floor next to "What the AI sees", same minute),
`screens/map-walking-desktop.png`, `screens/map-walking-phone.png`.

**One top view of the floor** (`backend/floormap.py`, `/api/mataf/floor`, `.png`). Every camera's latest live heatmap
is projected onto the Mataf floor at its real position with the calibrated camera (`config/cameras.json`, the m2
metric's camera) and the frame's homography to its reference: floor cell (0.75 m, head plane) -> reference px ->
frame px -> heatmap px; the projection's area factor turns people per pixel into people per m², and the live
head-size correction applies as in the metric. Not seen by a camera: behind / on the Kaaba (its silhouette in that
frame), under broadcast graphics, outside the frame. Cameras merge per cell, newest reading wins; a reading fades
from 2 to 10 minutes; grey where no camera saw the floor in the last 10 minutes. Check (2026-10-26 01:50): the floor
means 1-25 m from the walls per camera (0.99 / 0.94 / 1.30) match the metric's readings of the same frames (1.06 /
1.06 / 1.40). The twin drapes this texture over its floor disc; recomputed when a heatmap changes and every minute.

**One colour scale everywhere**: people per m², stops at 0 / 1 / 2 / 3 / 4+ (dim teal, teal, gold, orange, coral;
`backend/aiview.py` RAMP_M2, CSS `--m2-ramp`): twin floor, AI view, the map's Haram footprint, all legends.

**The edge, low detail** (`matafArch.ts`, matched to the roof camera frame and the 2019 aerial): white marble floor;
the golden Ottoman arcade ring (a dark recessed band for its arches, small white domes on top, instanced); the grey
Mataf expansion in three stepped levels (the middle one, the open gallery, dark). No modelled arches, light poles,
lamps or thin lines; every recess a real offset (no coplanar faces); Lambert materials, no textures. The default view
is more top-down (58 degrees desktop, 62 phone) so the floor pattern reads.

**GPU budget** (the crowd collector shares the GPU): no shadow map (was 2048² rendered on changes), pixel ratio <= 1.5
(was 1.75), 24 fps desktop / 20 phone (was 30 / 20), nothing off screen or in a hidden tab, the canvas only right of
the number block on desktop (a smaller framebuffer), half-size Kaaba belt textures, MapLibre pixel ratio <= 1.5,
~10 draw calls for the building (was ~40 with per-bay arcades). Measured with `node dashboard/scripts/gpumem.mjs`
(real GPU, Windows GPU counters): the dashboard tab's GPU process 264 MB -> 149 MB dedicated (an empty Chrome: 34
MB); the collector 2.4 GB meanwhile; 24 fps, live video 0 dropped frames (`perf.mjs`).

**Flicker.** Not reproduced on this machine: the previous build (94f1626) ran 4 min on the real GPU with the live video
playing: 0 context losses, 0 canvas resets, 0 blank samples (`node dashboard/scripts/flicker.mjs`); no browser crash
reports since 2026-10-09 and no GPU driver resets (TDR) in the Windows log. A forced context loss
(`WEBGL_lose_context`) recovered in both builds. So the likely trigger is the GPU running short of memory in the
user's browser (8 GB card: the collector 2.4-3.3 GB, plus games / other apps seen at 5.4-6.6 GB in total), which
makes Chrome drop WebGL contexts without a crash. Fixes: half the dashboard's VRAM (above); a lost context now shows a
plain message ("3D view paused: the graphics card is short of memory") and redraws when restored, so if the panel
still flickers WITHOUT that message, the cause is elsewhere (the next suspect: the hybrid-GPU laptop presenting the
browser's NVIDIA frames on the AMD-driven display).

**What the AI sees** for calibrated cameras: the same projection the other way (`floormap.image_density`): people per
m² at every heatmap pixel on the floor, smoothed over ~1.2 m of floor (so a nearby person is not a speck and far
crowds keep their shape), leftover specks removed, absolute colours; behind it a backdrop (`/api/aiview/<id>/bg.png`):
the floor a little lighter, the Kaaba a dark block, the building lines faint. Uncalibrated cameras keep the relative
colours. `?ai=<view>` pins one camera (review).

**Walking areas on the map**: OSM has only the Western and North Haram Squares (searched 2026-10-26: highway=pedestrian
areas, area:highway=*, place=square, footway areas, man_made=bridge). The rest is DERIVED and approximate
(`scripts/fetch_osm_haram.py` derive): open ground within 400 m of the Kaaba, inside the ring road, minus buildings and
street-level car roads (tunnels excluded: the plaza is above them), minus the mosque and the mapped squares; south
(Ajyad) and east (Mas'a side) now shaded. Features carry "approx": true.

## 12. "What the AI sees": alignment check and sync with the live video (2026-10-26)

**Alignment** (`scripts/debug_ai_alignment.py`, local-only overlays in `data/samples/debug_align/`, never committed or
served): the AI view's pipeline rebuilt on the 3 newest saved frames per calibrated camera (DM-Count -> the collector's
heatmap -> reference -> frame homography -> `floormap.image_density` -> `aiview.heat_m2`), compared with the model's
density on the real frame. Share of the floor's density mass inside the drawn (>= 25 % opacity) heat: low camera 94 %,
wide 100 %, roof 100 %; offset <= 1 px; no mirroring (correlation as drawn 0.71-0.95, flipped -0.17-0.69). Fixed on
the way: sparse zoomed-in crowds (~0.2-0.3 people per m² after smoothing) fell under the drawing threshold (one low
camera frame at 65 %); quiet ground now shows a faint dim teal (25 % opacity at ~0.2 people per m²). The AI view's
label for calibrated cameras is that heatmap's own people per m² over the zone (area-weighted).

**Sync**: the snapshot carries `on_air` (the camera of the newest frame the collector identified, if under 40 s old;
none = another shot). The AI view shows the on-air camera when its counted frame is under a minute old ("In sync with
the live video"); otherwise it says "Live video is on another camera – this is the last reading from the <camera>,
N min ago" (or, same camera but older: "Live video shows the <camera>; its last counted frame is N min old"). A camera
picked in the chips still wins. Heatmaps come only from accepted per-camera readings, a few per hour per camera.

## 13. Final polish; the dashboard design is FROZEN for v1 (2026-10-26)

**Design frozen for v1.** From here on: fixes only (bugs, data, wording); no new visual changes without a new decision
by the user. Reference screens: `screens/v1-desktop.png`, `screens/v1-phone.png`.

- Twin: the grey Mataf expansion is thin and low (66-76 m, 6 and 10 m high; it was 66-92 m and up to 25 m), the view
  is fitted to the ring, so the floor fills most of the panel; the canvas starts right of the number block (no overlap).
- Camera labels: small chips (dot + short name) on the outer ring, pushed outwards from the floor's centre, never on the
  crowd; full details on hover, keyboard focus or tap. Markers stand on the ring at each camera's bearing.
- Blind spots: a light, faint veil (RGBA 196/202/202/34) with feathered edges (~1.2 m), so "not seen" never looks like
  an object.
- Map label: two short lines, placed where its text box (~600 x 210 m at the default zoom) lies most on walking area and
  clear of the Haram (`scripts/fetch_osm_haram.py` label_spot: 51 % on walking area; the plazas are smaller than the
  text at this zoom, so a full fit is impossible without hiding the label).
- The Mataf number stays the hero when its reading is old ("at the last reading"); the map's Haram fill uses only a
  fresh one. A picked camera without a recent reading says so plainly.
- Heatmaps for every m2 frame (collector, one restart 2026-10-26 16:34).

Post-freeze change (user decision 2026-10-26): road states on the map and in its legend are "Clear · سالك" (green
#3fa56a, thinner), "Slow · بطيء" (orange #f28a1e) and "Traffic jam · زحمة" (red #e8384f); free-flowing roads are no
longer muted grey-teal.

Post-freeze change (user request 2026-10-26, live shot service): "What the AI sees" follows the shot on air; a shot
without a known camera is shown on a relative scale, labelled "current shot, approximate", and never on the twin; under
the number a small teal line says how the frame was placed ("calibrated camera", "position by image registration",
"from the prayer rows"); its people per m² is the heatmap's average over the 1-25 m ring ("1–25 m from the Kaaba").
A Mataf reading older than a minute shows its age next to the number (amber, e.g. "4 min ago"); "now" only up to 2
minutes; the map legend adds the age to the Haram's reading.

Post-freeze data addition (user request 2026-10-26): a fourth calibrated camera, "Roof camera B" (cam8_roof_b), gets a chip and a twin marker in soft blue (#9fb4ff), same pattern as the others; no other visual change.

Post-freeze change (user decision 2026-10-28, game mode removed): the header's LIVE badge no longer has the game-mode
variant ("game mode · every 30 s" in gold); it shows LIVE / STALE / OFFLINE only.

## Crowd-safety levels (user decision 2026-10-27)

The one people-per-m² scale of the dashboard (twin floor, AI view, map, legends, overlays, the Mataf hero's level,
alerts) is four crowd-safety levels from Fruin's walkway level of service (Fruin 1971, *Pedestrian Planning and
Design*; checked against FHWA-RD-98-107 Table 3: LOS D 0.9-1.4, E 0.5-0.9, F < 0.5 m² per person = 15 / 10 / 5 ft²):

| Level | People per m² | Fruin LOS | Colour |
|---|---|---|---|
| quiet | below 0.7 | A-C | teal `#2fae9c`, fading to `#245a5a` on empty floor |
| busy | 0.7-1.1 | D | gold `#e6c04f` |
| crowded | 1.1-2.2 | E | orange `#f26a3d` |
| packed | from 2.2 | F | crimson `#c41f6b` |

A level belongs to an area, not to a few people (user decision 2026-10-27): the twin floor and the AI view colour the
people per m² averaged over at least 25 m² of floor around each point (area-weighted; the user's "e.g. 10 m²" still
showed groups of 7-8 people on empty floor as busy, 0.71-0.75; over 25 m² they read 0.44-0.53, a real tawaf crowd
1.19).

Why: the old ramp (0 -> 4+, red only at 4) showed full prayer rows (~1.3 per m², 1.22 m x 0.64 m per person) as quiet.
2.2 is Fruin's 5 ft² (2.15); the FHWA table's rounded metric value would be 2.0. Neighbouring colours pass the dataviz
validator's colour-blind checks on the dark ground (CVD dE >= 13, normal dE >= 19, contrast >= 3:1); the level words
are always written next to the colour (legend, badge, map). Code: `backend/aiview.py` LEVELS / ramp_m2,
`dashboard/src/api.ts` CROWD_LEVELS / crowdColour, `LevelScale.tsx`, CSS `--lv-*`. Display only: readings, DB and
forecast unchanged. The per-camera legacy heroes (people per 100 x 100 px) keep their own relative levels.

