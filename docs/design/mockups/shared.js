// Shared helpers for the three design-direction mockups (static, data from mockdata.js).
const M = window.MOCK;
const HARAM = [39.8262, 21.4225];
const fmt = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Riyadh", hour: "2-digit", minute: "2-digit" });
const hm = (iso) => fmt.format(new Date(iso));
const PRAYERS = ["fajr", "sunrise", "dhuhr", "asr", "maghrib", "isha"];
const PNAME = { fajr: "Fajr", sunrise: "Sunrise", dhuhr: "Dhuhr", asr: "Asr", maghrib: "Maghrib", isha: "Isha" };
const PAR = { fajr: "الفجر", sunrise: "الشروق", dhuhr: "الظهر", asr: "العصر", maghrib: "المغرب", isha: "العشاء" };
const TT = { free: "#2fb344", slow: "#f59f00", jam: "#e8384f", closed: "#7a828c" };
const levelColor = (l) => (l == null ? TT.closed : l >= 0.8 ? TT.free : l >= 0.5 ? TT.slow : TT.jam);

function areaOf(a) {
  return M.traffic.areas.find((x) => x.area === a && x.road_category === "all");
}

/** SVG line chart. series: [[iso, value], ...] or {name: [[iso, v]]}. */
function lineChart(el, seriesMap, o = {}) {
  const W = el.clientWidth || 600, H = o.height || 120, pl = o.padLeft ?? 34, pb = 18, pt = 6, pr = 6;
  const all = Object.values(seriesMap).flat();
  const t0 = o.t0 ?? Math.min(...all.map((p) => +new Date(p[0])));
  const t1 = o.t1 ?? Math.max(...all.map((p) => +new Date(p[0])));
  const vmax = o.vmax ?? Math.max(...all.map((p) => p[1])) * 1.1;
  const x = (t) => pl + ((t - t0) / (t1 - t0)) * (W - pl - pr);
  const y = (v) => pt + (1 - v / vmax) * (H - pt - pb);
  let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" class="chart">`;
  for (let g = 0; g <= 2; g++) {
    const v = (vmax / 2) * g;
    s += `<line x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${pl - 6}" y="${y(v) + 4}" class="tick" text-anchor="end">${Math.round(v)}${o.unit || ""}</text>`;
  }
  if (o.prayers && M.prayer.times) {
    for (const p of PRAYERS) {
      const t = +new Date(M.prayer.times[p].utc);
      for (const off of [0, -86400000, 86400000]) {
        const tt = t + off;
        if (tt > t0 && tt < t1) s += `<line x1="${x(tt)}" x2="${x(tt)}" y1="${pt}" y2="${H - pb}" class="pline"/><text x="${x(tt) + 3}" y="${pt + 10}" class="ptext">${PNAME[p]}</text>`;
      }
    }
  }
  if (o.nowFrac != null) {
    const nx = pl + o.nowFrac * (W - pl - pr);
    s += `<rect x="${nx}" y="${pt}" width="${W - pr - nx}" height="${H - pt - pb}" class="fzone"/><text x="${nx + 6}" y="${H - pb - 6}" class="ftext">forecast · coming in step 7</text>`;
  }
  const hours = 6;
  for (let i = 0; i <= hours; i++) {
    const t = t0 + ((t1 - t0) * i) / hours;
    s += `<text x="${x(t)}" y="${H - 4}" class="tick" text-anchor="middle">${hm(new Date(t).toISOString())}</text>`;
  }
  Object.entries(seriesMap).forEach(([name, pts], i) => {
    const d = pts.map((p, k) => `${k ? "L" : "M"}${x(+new Date(p[0])).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
    s += `<path d="${d}" class="ln ln${i}" fill="none"/>`;
  });
  el.innerHTML = s + "</svg>";
}

function trafficSeries() {
  const r1 = [], r3 = [];
  for (const p of M.trafficSeries) (p.area === "r1km" ? r1 : r3).push([p.poll_utc, p.congested_pct]);
  return { "1 km": r1, "3 km": r3 };
}

function circle(lon, lat, km, n = 96) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    pts.push([lon + (km / (111.32 * Math.cos((lat * Math.PI) / 180))) * Math.cos(a), lat + (km / 110.57) * Math.sin(a)]);
  }
  return { type: "Feature", geometry: { type: "LineString", coordinates: pts }, properties: {} };
}

/** MapLibre + OpenFreeMap vector basemap, recoloured to the direction's palette, TomTom segments on top. */
function initMap(el, theme) {
  const map = new maplibregl.Map({
    container: el, style: `https://tiles.openfreemap.org/styles/${theme.base}`, center: HARAM, zoom: theme.zoom || 13.4,
    attributionControl: { compact: true }, preserveDrawingBuffer: true,
  });
  map.on("load", () => {
    for (const l of map.getStyle().layers) {
      const sl = l["source-layer"] || "";
      try {
        if (l.type === "background") map.setPaintProperty(l.id, "background-color", theme.land);
        else if (sl === "water" && l.type === "fill") map.setPaintProperty(l.id, "fill-color", theme.water);
        else if (sl === "waterway") map.setPaintProperty(l.id, "line-color", theme.water);
        else if ((sl === "landuse" || sl === "landcover" || sl === "park") && l.type === "fill") {
          map.setPaintProperty(l.id, "fill-color", theme.park);
          map.setPaintProperty(l.id, "fill-opacity", 0.5);
        } else if (sl === "building" && l.type === "fill") map.setPaintProperty(l.id, "fill-color", theme.building);
        else if (sl === "building" && l.type === "fill-extrusion") map.setLayoutProperty(l.id, "visibility", "none");
        else if (sl === "transportation" && l.type === "line") map.setPaintProperty(l.id, "line-color", theme.road);
        else if (l.type === "symbol") {
          map.setPaintProperty(l.id, "text-color", theme.label);
          map.setPaintProperty(l.id, "text-halo-color", theme.labelHalo);
          if (sl === "poi") map.setLayoutProperty(l.id, "visibility", "none");
        }
      } catch (e) { /* layer without that property */ }
    }
    map.addSource("rings", { type: "geojson", data: { type: "FeatureCollection", features: [circle(...HARAM, 1), circle(...HARAM, 3)] } });
    map.addLayer({ id: "rings", type: "line", source: "rings", paint: { "line-color": theme.ring, "line-width": 1, "line-dasharray": [3, 3] } });
    map.addSource("seg", { type: "geojson", data: M.segments });
    const width = ["match", ["get", "category"], "motorway", 5, "trunk", 5, "primary", 4.2, "secondary", 3.4, "tertiary", 2.6, 1.8];
    map.addLayer({ id: "seg-casing", type: "line", source: "seg", layout: { "line-cap": "round", "line-join": "round" },
      filter: theme.free ? ["<", ["get", "level"], 0.8] : ["has", "category"],
      paint: { "line-color": theme.casing, "line-width": ["+", width, 2.2] } });
    map.addLayer({ id: "seg", type: "line", source: "seg", layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-width": theme.free ? ["case", [">=", ["get", "level"], 0.8], ["*", width, 0.7], width] : width,
        "line-opacity": theme.free ? ["case", [">=", ["get", "level"], 0.8], 0.75, 1] : 1,
        "line-color": ["case", ["get", "closed"], TT.closed, ["<", ["get", "level"], 0.5], TT.jam, ["<", ["get", "level"], 0.8], TT.slow, theme.free || TT.free] } });
    map.addSource("haram", { type: "geojson", data: { type: "Point", coordinates: HARAM } });
    map.addLayer({ id: "haram", type: "circle", source: "haram", paint: { "circle-radius": 7, "circle-color": theme.accent, "circle-stroke-color": theme.casing, "circle-stroke-width": 2 } });
  });
  return map;
}
