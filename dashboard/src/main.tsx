import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// self-hosted fonts (OFL): Alexandria = numerals + headings, Readex Pro = UI, Reem Kufi = Arabic labels
import "@fontsource-variable/alexandria";
import "@fontsource-variable/readex-pro";
import "@fontsource/reem-kufi/400.css";
import "@fontsource/reem-kufi/500.css";
import "./styles.css";
import App from "./App";
import CalmApp from "./calm/CalmApp";

// the main dashboard is the calm one since 2026-11-01 (user pick: option A); the previous engineer dashboard at /?ui=old
const OLD = new URLSearchParams(location.search).get("ui") === "old";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {OLD ? <App /> : <CalmApp />}
  </StrictMode>,
);
