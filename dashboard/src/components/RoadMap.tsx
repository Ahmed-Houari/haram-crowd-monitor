// Lazy chunk: MapLibre + OpenFreeMap dark basemap (free, no key), muted to the A2 palette. TomTom segments in three
// plain colours (user decision 2026-10-26): clear green (thinner), slow orange, traffic jam red.
// Around the Haram (2026-10-26): the pedestrian squares (OpenStreetMap outlines, /osm/haram_area.geojson from
// scripts/fetch_osm_haram.py) get a subtle fill and a label, since TomTom has no car traffic there; the mosque's own
// footprint takes the colour of the Mataf's people per m² (the twin's scale); tunnels are thin grey dashes, under
// the TomTom lines, so a tunnel without traffic data is still visible.
import { useEffect, useRef } from "react";
import { Map as MlMap, setWorkerUrl, type ExpressionSpecification, type GeoJSONSource } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import { crowdColour as levelColour } from "../api";

// The worker is a separate file; Vite's pre-bundling breaks MapLibre's relative worker URL, so point at the
// bundled worker. In dev, Vite injects its client into that module, which fails inside MapLibre's blob
// worker: use the raw source as a blob instead (dead code in the build).
let workerReady: Promise<void> | null = null;
function initWorker(): Promise<void> {
  workerReady ??= (async () => {
    if (import.meta.env.DEV) {
      const src = (await import("maplibre-gl/dist/maplibre-gl-worker.mjs?raw")).default;
      setWorkerUrl(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
    } else setWorkerUrl(workerUrl);
  })();
  return workerReady;
}

interface Props {
  center: { lat: number; lon: number };
  reloadKey: string | null; // changes with each TomTom poll
  /** the Mataf's people per m² while it is fresh, else null (the footprint then stays grey) */
  crowdM2: number | null;
}

const C = { land: "#0d181a", water: "#0b2830", park: "#0f201d", building: "#142327", road: "#1c2e32", label: "#6f7f80",
  halo: "#0a1315", ring: "rgba(200,164,92,.45)", casing: "#0a1315", accent: "#c8a45c", free: "#3fa56a",
  slow: "#f28a1e", jam: "#e8384f", closed: "#7a828c", plaza: "#b9ad8a", tunnel: "#6f7c7e", noData: "#56656a" };

// people per m² -> colour: the dashboard's crowd-safety levels (api.ts CROWD_LEVELS)
const crowdColour = (v: number | null) => levelColour(v, C.noData);

function circle(lon: number, lat: number, km: number, n = 96) {
  const coordinates = Array.from({ length: n + 1 }, (_, i) => {
    const a = (i / n) * 2 * Math.PI;
    return [lon + (km / (111.32 * Math.cos((lat * Math.PI) / 180))) * Math.cos(a), lat + (km / 110.57) * Math.sin(a)];
  });
  return { type: "Feature" as const, geometry: { type: "LineString" as const, coordinates }, properties: {} };
}

function mute(map: MlMap) {
  for (const l of map.getStyle().layers) {
    const sl = ("source-layer" in l ? l["source-layer"] : "") || "";
    try {
      if (l.type === "background") map.setPaintProperty(l.id, "background-color", C.land);
      else if (sl === "water" && l.type === "fill") map.setPaintProperty(l.id, "fill-color", C.water);
      else if (sl === "waterway") map.setPaintProperty(l.id, "line-color", C.water);
      else if ((sl === "landuse" || sl === "landcover" || sl === "park") && l.type === "fill") {
        map.setPaintProperty(l.id, "fill-color", C.park);
        map.setPaintProperty(l.id, "fill-opacity", 0.5);
      } else if (sl === "building" && l.type === "fill") map.setPaintProperty(l.id, "fill-color", C.building);
      else if (sl === "building" && l.type === "fill-extrusion") map.setLayoutProperty(l.id, "visibility", "none");
      else if (sl === "transportation" && l.type === "line") map.setPaintProperty(l.id, "line-color", C.road);
      else if (l.type === "symbol") {
        map.setPaintProperty(l.id, "text-color", C.label);
        map.setPaintProperty(l.id, "text-halo-color", C.halo);
        if (sl === "poi") map.setLayoutProperty(l.id, "visibility", "none");
      }
    } catch {
      /* layer without that property */
    }
  }
}

export default function RoadMap({ center, reloadKey, crowdM2 }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const crowd = useRef(crowdM2);
  crowd.current = crowdM2;

  useEffect(() => {
    const box = el.current;
    if (!box) return;
    let map: MlMap | null = null;
    let gone = false;
    initWorker().then(() => {
      if (gone) return;
      map = build(box, center, () => crowd.current);
      mapRef.current = map;
    });
    return () => {
      gone = true;
      map?.remove();
      mapRef.current = null;
    };
  }, [center.lat, center.lon]);

  useEffect(() => {
    const map = mapRef.current;
    if (map?.isStyleLoaded() && map.getSource("seg")) loadSegments(map);
  }, [reloadKey]);

  useEffect(() => {
    const map = mapRef.current;
    if (map?.getLayer("haram-fill")) map.setPaintProperty("haram-fill", "fill-color", crowdColour(crowdM2));
  }, [crowdM2]);

  return <div ref={el} className="map" />;
}

function build(box: HTMLDivElement, center: Props["center"], crowdNow: () => number | null): MlMap {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const map = new MlMap({
    container: box, style: "https://tiles.openfreemap.org/styles/dark", center: [center.lon, center.lat],
    zoom: box.clientWidth < 500 ? 12.9 : 13.4, attributionControl: { compact: true }, fadeDuration: reduce ? 0 : 300,
    pixelRatio: Math.min(devicePixelRatio, 1.5), // GPU memory: the collector shares this GPU
    cooperativeGestures: true,
  });
  map.on("load", () => {
    mute(map);
    const firstRoad = map.getStyle().layers.find((l) => "source-layer" in l && l["source-layer"] === "transportation")?.id;
    // pedestrian squares (no car traffic) under the roads; the mosque's footprint by its crowd level above them
    map.addSource("osm", { type: "geojson", data: "/osm/haram_area.geojson" });
    map.addLayer({ id: "plaza-fill", type: "fill", source: "osm", filter: ["==", ["get", "kind"], "pedestrian"],
      paint: { "fill-color": C.plaza, "fill-opacity": 0.2 } }, firstRoad);
    map.addLayer({ id: "plaza-line", type: "line", source: "osm", filter: ["==", ["get", "kind"], "pedestrian"],
      paint: { "line-color": C.plaza, "line-opacity": 0.5, "line-width": 0.9 } }, firstRoad);
    // tunnels: thin grey dashes; where TomTom has data for one, its coloured line covers the dashes
    map.addLayer({ id: "tunnels", type: "line", source: "openmaptiles", "source-layer": "transportation",
      filter: ["all", ["==", ["get", "brunnel"], "tunnel"],
        ["match", ["get", "class"], ["motorway", "trunk", "primary", "secondary", "tertiary", "minor", "service"], true, false]],
      layout: { "line-cap": "butt", "line-join": "round" },
      paint: { "line-color": C.tunnel, "line-width": 1.1, "line-opacity": 0.85, "line-dasharray": [2.5, 2] } });
    map.addLayer({ id: "haram-fill", type: "fill", source: "osm", filter: ["==", ["get", "kind"], "haram"],
      paint: { "fill-color": crowdColour(crowdNow()), "fill-opacity": 0.82 } });
    map.addLayer({ id: "haram-line", type: "line", source: "osm", filter: ["==", ["get", "kind"], "haram"],
      paint: { "line-color": "#ece7dc", "line-opacity": 0.55, "line-width": 1 } });
    map.addSource("rings", { type: "geojson", data: { type: "FeatureCollection", features: [circle(center.lon, center.lat, 1), circle(center.lon, center.lat, 3)] } });
    map.addLayer({ id: "rings", type: "line", source: "rings", paint: { "line-color": C.ring, "line-width": 1, "line-dasharray": [3, 3] } });
    map.addSource("seg", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    const width: ExpressionSpecification = ["match", ["get", "category"], "motorway", 5, "trunk", 5, "primary", 4.2, "secondary", 3.4, "tertiary", 2.6, 1.8];
    map.addLayer({ id: "seg-casing", type: "line", source: "seg", layout: { "line-cap": "round", "line-join": "round" },
      filter: ["<", ["get", "level"], 0.8], paint: { "line-color": C.casing, "line-width": ["+", width, 2.2] } });
    map.addLayer({ id: "seg", type: "line", source: "seg", layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-width": ["case", [">=", ["get", "level"], 0.8], ["*", width, 0.7], width],
        "line-opacity": ["case", [">=", ["get", "level"], 0.8], 0.75, 1],
        "line-color": ["case", ["get", "closed"], C.closed, ["<", ["get", "level"], 0.5], C.jam, ["<", ["get", "level"], 0.8], C.slow, C.free] },
    });
    // one small label where its text box lies on open plaza (spot computed by the fetch script): why no traffic colour
    map.addLayer({ id: "plaza-label", type: "symbol", source: "osm", filter: ["==", ["get", "kind"], "label"],
      layout: { "text-field": ["get", "text"], "text-font": ["Noto Sans Regular"], "text-size": 10, "text-line-height": 1.15,
        "text-max-width": 20, "text-allow-overlap": true, "text-anchor": "center", "text-justify": "center" },
      paint: { "text-color": "#c4bea9", "text-halo-color": C.halo, "text-halo-width": 1.4 } });
    loadSegments(map);
  });
  return map;
}

function loadSegments(map: MlMap) {
  fetch("/api/traffic/segments")
    .then((r) => (r.ok ? r.json() : null))
    .then((geo) => geo && (map.getSource("seg") as GeoJSONSource | undefined)?.setData(geo))
    .catch(console.error);
}
