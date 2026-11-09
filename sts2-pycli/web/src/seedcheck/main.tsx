import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../theme.css";
import "./seedcheck.css";
import SeedCheckPage from "./SeedCheckPage";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SeedCheckPage />
  </StrictMode>,
);
