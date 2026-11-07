import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../theme.css";
import "./frame.css";
import FrameApp from "./FrameApp";
import { installErrorLogging } from "./host";

installErrorLogging();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <FrameApp />
  </StrictMode>,
);
