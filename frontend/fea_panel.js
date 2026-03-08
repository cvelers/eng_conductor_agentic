/**
 * FEA Panel Controller — manages the FEA UI panel, 3D viewer, and solver communication.
 * Instantiated per assistant message when FEA events are received.
 */

import { FEAModel } from "./fea/model.js";
import { loadProfileDb } from "./fea/profiles/profile_db.js";

let _sceneModule = null;
let _memberModule = null;
let _supportModule = null;
let _loadModule = null;
let _resultModule = null;

// Lazy-load Three.js-dependent modules
async function loadViewerModules() {
  if (_sceneModule) return;
  _sceneModule = await import("./viewer/scene.js");
  _memberModule = await import("./viewer/member_builder.js");
  _supportModule = await import("./viewer/support_vis.js");
  _loadModule = await import("./viewer/load_vis.js");
  _resultModule = await import("./viewer/result_vis.js");
}


export class FEAPanelController {
  constructor(panelEl) {
    this.panelEl = panelEl;
    this.model = new FEAModel();
    this.scene = null;
    this.worker = null;
    this.results = null;
    this.sessionId = null;
    this._viewerReady = false;
    this._initPromise = null;
    this._profileDbReady = false;

    // Start loading the profile DB immediately (no Three.js dependency)
    this._profileDbPromise = loadProfileDb().then(db => {
      this.model.setProfileDb(db);
      this._profileDbReady = true;
    }).catch(err => {
      console.warn("Failed to load profile DB:", err);
    });

    // Wire up toolbar buttons
    this._wireToolbar();
  }

  // ── Initialization (lazy) ──────────────────────────────────────

  async _ensureViewer() {
    if (this._viewerReady) return;
    if (this._initPromise) return this._initPromise;

    this._initPromise = (async () => {
      // Load profile database
      const db = await loadProfileDb();
      this.model.setProfileDb(db);

      // Load Three.js modules
      await loadViewerModules();

      // Init scene
      const container = this.panelEl.querySelector(".fea-viewer-container");
      this.scene = new _sceneModule.FEAScene(container);
      await this.scene.init();
      this._viewerReady = true;
    })();

    return this._initPromise;
  }

  // ── Command handling ───────────────────────────────────────────

  async handleCommands(commands) {
    // Ensure profile DB is loaded before applying commands (needed for assign_section)
    if (!this._profileDbReady) {
      await this._profileDbPromise;
    }
    // Apply commands to model data immediately (no Three.js dependency)
    for (const cmd of commands) {
      this.model.applyCommand(cmd);
    }

    // Then try to update the 3D viewer (best-effort)
    try {
      await this._ensureViewer();
      this._rebuildViewer();
    } catch (err) {
      console.warn("FEA viewer rebuild skipped:", err.message);
    }
  }

  async handleCommand(cmd) {
    await this.handleCommands([cmd]);
  }

  // ── Solve request ──────────────────────────────────────────────

  async handleSolveRequest(request) {
    // Solver doesn't need Three.js — don't block on viewer init
    this.sessionId = request.session_id || request.sessionId;
    const loadCaseId = request.load_case_id || request.loadCaseId || Object.keys(this.model.loadCases)[0] || "LC1";

    // Validate model
    const validation = this.model.validate();
    if (!validation.valid) {
      console.error("FEA model validation failed:", validation.errors);
      // Still try to solve — the worker will catch singular matrix
    }

    // Show progress
    this._showProgress("Preparing solver...", 0);

    // Create worker
    if (!this.worker) {
      this.worker = new Worker("/static/fea/worker/fea_worker.js?v=4");
      this.worker.onerror = (e) => {
        console.error("FEA Worker error:", e.message, e);
      };
    }

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.worker.terminate();
        this.worker = null;
        this._hideProgress();
        reject(new Error("Solver timeout (60s)"));
      }, 60000);

      this.worker.onmessage = async (e) => {
        const { type, payload } = e.data;

        if (type === "PROGRESS") {
          this._showProgress(payload.message, payload.percent);
        }

        if (type === "RESULT") {
          clearTimeout(timeout);
          this.results = payload;
          this._hideProgress();

          // Send results back to backend
          try {
            await fetch("/api/fea/results", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                session_id: this.sessionId,
                results: payload,
              }),
            });
          } catch (err) {
            console.error("Failed to send FEA results to backend:", err);
          }

          resolve(payload);
        }

        if (type === "ERROR") {
          clearTimeout(timeout);
          this._hideProgress();
          this._showError(payload.message);
          // Send error back to backend so the LLM can diagnose and fix
          try {
            await fetch("/api/fea/results", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                session_id: this.sessionId,
                results: {
                  error: true,
                  errorMessage: payload.message,
                  errorPhase: payload.phase || "unknown",
                },
              }),
            });
          } catch (err) {
            console.error("Failed to send FEA error to backend:", err);
          }
          reject(new Error(payload.message));
        }
      };

      // Send model to worker
      const serialized = this.model.serialize();
      this.worker.postMessage({
        type: "SOLVE",
        payload: { model: serialized, loadCaseId },
      });
    });
  }

  // ── View commands ──────────────────────────────────────────────

  async handleViewCommand(cmd) {
    await this._ensureViewer();

    switch (cmd.action) {
      case "fit_view":
        this.scene.fitToModel(this.model.getBoundingBox());
        break;
      case "set_view":
        this.scene.setView(cmd.preset || "iso");
        break;
      case "show_deformed":
        this._showDeformedShape(cmd.scale || cmd.scale_factor);
        break;
      case "show_results":
      case "show_moment_diagram":
        this._showForceDiagram(cmd.result_type || cmd.force_type || "M", cmd.scale);
        break;
      case "show_shear_diagram":
        this._showForceDiagram("V", cmd.scale);
        break;
      case "show_axial_diagram":
        this._showForceDiagram("N", cmd.scale);
        break;
      case "hide_results":
        this.scene.clearResults();
        break;
    }
  }

  // ── Viewer rebuild ─────────────────────────────────────────────

  _rebuildViewer() {
    if (!this._viewerReady) return;

    // Clear existing
    this.scene.clearMembers();
    this.scene.clearSupports();
    this.scene.clearLoads();
    this.scene.clearLabels();

    // Build members
    const meshes = _memberModule.buildAllMembers(this.model);
    for (const [elemId, mesh] of meshes) {
      this.scene.addMember(elemId, mesh);
    }

    // Build supports
    for (const [nodeId, rest] of Object.entries(this.model.restraints)) {
      const node = this.model.nodes[nodeId];
      if (!node) continue;
      const sym = _supportModule.createSupportSymbol(node, rest);
      if (sym) this.scene.addSupportMesh(sym);
    }

    // Build loads
    const bbox = this.model.getBoundingBox();
    const maxDim = Math.max(
      bbox.max.x - bbox.min.x,
      bbox.max.y - bbox.min.y,
      bbox.max.z - bbox.min.z,
    ) || 1000;

    for (const lcId of Object.keys(this.model.loadCases)) {
      const loadMeshes = _loadModule.buildAllLoads(this.model, lcId, maxDim);
      for (const m of loadMeshes) {
        this.scene.addLoadMesh(m);
      }
    }

    // Node labels
    for (const [nid, node] of Object.entries(this.model.nodes)) {
      this.scene.addLabel(nid, node);
    }

    // Fit camera
    this.scene.fitToModel(bbox);
  }

  // ── Result visualization ───────────────────────────────────────

  _showDeformedShape(scaleFactor) {
    if (!this.results || !this.results.displacements) return;
    this.scene.clearResults();

    const scale = scaleFactor || _resultModule.autoDeformScale(this.model, this.results.displacements);
    const group = _resultModule.createDeformedShape(this.model, this.results.displacements, scale);
    if (group) this.scene.addResultMesh(group);
  }

  _showForceDiagram(forceType, scaleFactor) {
    if (!this.results || !this.results.elementForces) return;
    this.scene.clearResults();

    const group = _resultModule.createForceDiagram(
      this.model, this.results.elementForces, forceType, scaleFactor || 1,
    );
    if (group) this.scene.addResultMesh(group);
  }

  // ── Result summary table ───────────────────────────────────────

  getResultSummary() {
    if (!this.results) return null;
    const mv = this.results.maxValues || {};
    const si = this.results.solverInfo || {};
    return {
      dofCount: si.dofCount,
      elementCount: si.elementCount,
      solveTimeMs: si.solveTimeMs,
      maxDisplacement: mv.maxDisplacement,
      maxMoment: mv.maxMoment,
      maxShear: mv.maxShear,
      displacements: this.results.displacements,
      reactions: this.results.reactions,
    };
  }

  populateResultsTable() {
    const container = this.panelEl.querySelector(".fea-results-table");
    if (!container || !this.results) return;

    const mv = this.results.maxValues || {};
    const si = this.results.solverInfo || {};

    // Convert N·mm to kN·m for display
    const maxM_kNm = mv.maxMoment ? (mv.maxMoment.value / 1e6).toFixed(2) : "—";
    const maxV_kN = mv.maxShear ? (mv.maxShear.value / 1e3).toFixed(2) : "—";
    const maxD_mm = mv.maxDisplacement ? mv.maxDisplacement.value.toFixed(3) : "—";

    container.innerHTML = `
      <table class="fea-summary-table">
        <tr><th colspan="2">Solver Info</th></tr>
        <tr><td>DOF count</td><td>${si.dofCount || "—"}</td></tr>
        <tr><td>Elements</td><td>${si.elementCount || "—"}</td></tr>
        <tr><td>Solve time</td><td>${si.solveTimeMs || "—"} ms</td></tr>
        <tr><th colspan="2">Max Values</th></tr>
        <tr><td>Max displacement</td><td>${maxD_mm} mm (${mv.maxDisplacement?.direction || ""}, node ${mv.maxDisplacement?.nodeId || "—"})</td></tr>
        <tr><td>Max bending moment</td><td>${maxM_kNm} kN·m (elem ${mv.maxMoment?.elementId || "—"})</td></tr>
        <tr><td>Max shear force</td><td>${maxV_kN} kN (elem ${mv.maxShear?.elementId || "—"})</td></tr>
      </table>
    `;

    // Reactions table
    if (this.results.reactions && Object.keys(this.results.reactions).length > 0) {
      let rhtml = '<table class="fea-summary-table"><tr><th>Node</th><th>Fx (kN)</th><th>Fy (kN)</th><th>Fz (kN)</th></tr>';
      for (const [nid, r] of Object.entries(this.results.reactions)) {
        rhtml += `<tr><td>${nid}</td><td>${((r.fx || 0) / 1e3).toFixed(2)}</td><td>${((r.fy || 0) / 1e3).toFixed(2)}</td><td>${((r.fz || 0) / 1e3).toFixed(2)}</td></tr>`;
      }
      rhtml += "</table>";
      container.innerHTML += rhtml;
    }

    container.classList.remove("hidden");
  }

  // ── UI helpers ─────────────────────────────────────────────────

  _showProgress(message, percent) {
    const el = this.panelEl.querySelector(".fea-progress");
    if (!el) return;
    el.classList.remove("hidden");
    const fill = el.querySelector(".fea-progress-fill");
    const text = el.querySelector(".fea-progress-text");
    if (fill) fill.style.width = `${percent || 0}%`;
    if (text) text.textContent = message || "";
  }

  _hideProgress() {
    const el = this.panelEl.querySelector(".fea-progress");
    if (el) el.classList.add("hidden");
  }

  _showError(msg) {
    const container = this.panelEl.querySelector(".fea-results-table");
    if (container) {
      container.innerHTML = `<div class="fea-error">Solver error: ${msg}</div>`;
      container.classList.remove("hidden");
    }
  }

  _wireToolbar() {
    // View preset buttons
    this.panelEl.querySelectorAll("[data-action]").forEach(btn => {
      btn.addEventListener("click", () => {
        const action = btn.dataset.action;
        if (action === "fit") this.scene?.fitToModel(this.model.getBoundingBox());
        else this.scene?.setView(action);
      });
    });

    // Result select dropdown
    const select = this.panelEl.querySelector(".fea-result-select");
    if (select) {
      select.addEventListener("change", () => {
        const val = select.value;
        if (val === "none") { this.scene?.clearResults(); return; }
        if (val === "displacement") { this._showDeformedShape(); return; }
        if (val === "bending_moment") { this._showForceDiagram("M"); return; }
        if (val === "shear_force") { this._showForceDiagram("V"); return; }
        if (val === "axial_force") { this._showForceDiagram("N"); return; }
      });
    }

    // Deformed toggle
    const defToggle = this.panelEl.querySelector(".fea-deformed-toggle");
    if (defToggle) {
      defToggle.addEventListener("change", () => {
        if (defToggle.checked) this._showDeformedShape();
        else this.scene?.clearResults();
      });
    }

    // Tab buttons
    this.panelEl.querySelectorAll(".fea-view-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.panelEl.querySelectorAll(".fea-view-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const view = btn.dataset.view;
        const viewerContainer = this.panelEl.querySelector(".fea-viewer-container");
        const resultsTable = this.panelEl.querySelector(".fea-results-table");
        if (view === "3d") {
          viewerContainer?.classList.remove("hidden");
          resultsTable?.classList.add("hidden");
        } else if (view === "results") {
          viewerContainer?.classList.add("hidden");
          resultsTable?.classList.remove("hidden");
          this.populateResultsTable();
        }
      });
    });

    // Collapse button
    const collapseBtn = this.panelEl.querySelector(".panel-collapse-btn");
    if (collapseBtn) {
      collapseBtn.addEventListener("click", () => {
        this.panelEl.classList.toggle("collapsed");
      });
    }
  }

  // ── Cleanup ────────────────────────────────────────────────────

  dispose() {
    if (this.scene) this.scene.dispose();
    if (this.worker) this.worker.terminate();
  }
}
