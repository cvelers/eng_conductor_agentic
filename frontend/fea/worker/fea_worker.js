import { solveFEA } from "../solver/solver_manager.js";

/* global self */

self.onmessage = function (e) {
  const { type, payload } = e.data;

  if (type === "CANCEL") {
    self._cancelled = true;
    return;
  }

  if (type !== "SOLVE") return;

  self._cancelled = false;

  try {
    const result = solveFEA(payload.model, payload.loadCaseId, (update) => {
      self.postMessage({ type: "PROGRESS", payload: update });
    });

    if (!self._cancelled) {
      self.postMessage({ type: "RESULT", payload: result });
    }
  } catch (err) {
    self.postMessage({
      type: "ERROR",
      payload: {
        message: err?.message || String(err),
        phase: err?.phase || "unknown",
      },
    });
  }
};
