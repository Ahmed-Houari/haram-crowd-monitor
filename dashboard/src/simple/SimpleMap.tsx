// Lazy chunk: the roads around the Haram in three colours only (TomTom levels: clear green, slow orange, jammed
// red; closed roads count as jammed). OpenFreeMap dark basemap (free, no key), muted to the night palette, labels
// kept for orientation. A still map: no pan or zoom, so the page scrolls normally on a phone.
import { useEffect, useRef } from "react";
import { Map as MlMap, setWorkerUrl, type GeoJSONSource } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";

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

/** /api/traffic/segments: GeoJSON lines with properties level (0 jammed .. 1 free) and closed */
export interface RoadsData { type: "FeatureCollection"; features: object[] }

export const ROAD = { free: "#3fa56a", slow: "#f28a1e", jam: "#e8384f" };
const C = { land: "#0d181a", water: "#0b2830", park: "#0f201d", building: "#142327", road: "#1f3236", label: "#7d8b8c", halo: "#0a1315" };

interface Props {
  center: { lat: number; lon: number };
  roads: RoadsData | null;
  lang: "ar" | "en";
  label: string;
}

export default function SimpleMap({ center, roads, lang, label }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const ready = useRef(false);
  const latest = useRef(roads); // the roads may arrive before the map has loaded
  latest.current = roads;

  useEffect(() => {
    let gone = false;
    initWorker().then(() => {
      if (gone || !box.current) return;
      const m = new MlMap({
        container: box.current, style: "https://tiles.openfreemap.org/styles/dark", center: [center.lon, center.lat],
        zoom: 14.1, interactive: false, attributionControl: { compact: true },
      });
      map.current = m;
      m.on("load", () => {
        for (const l of m.getStyle().layers) {
          const sl = ("source-layer" in l ? l["source-layer"] : "") || "";
          try {
            if (l.type === "background") m.setPaintProperty(l.id, "background-color", C.land);
            else if (sl === "water" && l.type === "fill") m.setPaintProperty(l.id, "fill-color", C.water);
            else if ((sl === "landuse" || sl === "landcover" || sl === "park") && l.type === "fill") m.setPaintProperty(l.id, "fill-color", C.park);
            else if (sl === "building" && l.type === "fill") m.setPaintProperty(l.id, "fill-color", C.building);
            else if (sl === "building" && l.type === "fill-extrusion") m.setLayoutProperty(l.id, "visibility", "none");
            else if (sl === "transportation" && l.type === "line") m.setPaintProperty(l.id, "line-color", C.road);
            else if (l.type === "symbol") {
              if (sl === "transportation_name" || sl === "poi" || sl === "housenumber") m.setLayoutProperty(l.id, "visibility", "none");
              else {
                m.setPaintProperty(l.id, "text-color", C.label);
                m.setPaintProperty(l.id, "text-halo-color", C.halo);
                m.setLayoutProperty(l.id, "text-field", ["coalesce", ["get", `name:${lang}`], ["get", "name"]]);
              }
            }
          } catch { /* a layer without that property */ }
        }
        m.addSource("roads", { type: "geojson", data: (latest.current ?? { type: "FeatureCollection", features: [] }) as never });
        m.addLayer({ id: "roads", type: "line", source: "roads", layout: { "line-cap": "round", "line-join": "round" },
          paint: { "line-width": ["interpolate", ["linear"], ["zoom"], 12, 1.6, 15, 3.4],
            "line-color": ["case", ["get", "closed"], ROAD.jam, ["<", ["coalesce", ["get", "level"], 1], 0.5], ROAD.jam,
              ["<", ["coalesce", ["get", "level"], 1], 0.8], ROAD.slow, ROAD.free] } });
        m.addSource("haram", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "Point", coordinates: [center.lon, center.lat] } } });
        m.addLayer({ id: "haram", type: "circle", source: "haram",
          paint: { "circle-radius": 7, "circle-color": "#c8a45c", "circle-stroke-color": "#0a1315", "circle-stroke-width": 3 } });
        box.current?.querySelector(".maplibregl-compact-show")?.classList.remove("maplibregl-compact-show"); // credits behind "i"
        ready.current = true;
      });
    });
    return () => { gone = true; map.current?.remove(); map.current = null; ready.current = false; };
  }, [center.lat, center.lon, lang]);

  useEffect(() => {
    const src = map.current?.getSource("roads") as GeoJSONSource | undefined;
    if (ready.current && src && roads) src.setData(roads as never);
  }, [roads]);

  return <div ref={box} className="map" role="img" aria-label={label} />;
}
