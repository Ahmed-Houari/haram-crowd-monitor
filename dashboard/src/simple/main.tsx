import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// self-hosted fonts (OFL), as the dashboard: Reem Kufi = the Arabic answer, Readex Pro = UI, Alexandria = numerals
import "@fontsource-variable/alexandria";
import "@fontsource-variable/readex-pro";
import "@fontsource/reem-kufi/500.css";
import "@fontsource/reem-kufi/600.css";
import "./simple.css";
import SimpleApp from "./SimpleApp";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SimpleApp />
  </StrictMode>,
);
