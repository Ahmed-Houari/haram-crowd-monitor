import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the API and the WebSocket to the FastAPI backend (default :8000; API_PORT=8001 for a
// second, development copy of the API while the supervisor's API child keeps serving :8000).
const api = `127.0.0.1:${process.env.API_PORT ?? 8000}`;

export default defineConfig({
  plugins: [react()],
  // three/r3f (Twin) and MapLibre (RoadMap) are ~1 MB each, but both are lazy chunks loaded after first paint
  build: {
    chunkSizeWarningLimit: 1100,
    // two pages: the dashboard (/) and the visitor page (/simple/, its own small bundle; MapLibre stays a lazy chunk)
    rollupOptions: { input: { main: "index.html", simple: "simple/index.html" } },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": `http://${api}`,
      "/ws": { target: `ws://${api}`, ws: true },
    },
  },
});
