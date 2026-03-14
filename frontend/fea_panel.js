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
    this._profileDb = null;
    this._currentResultView = "none";

    // Start loading the profile DB immediately (no Three.js dependency)
    this._profileDbPromise = loadProfileDb().then(db => {
      this._profileDb = db;
      this.model.setProfileDb(db);
      this._profileDbReady = true;
    }).catch(err => {
      console.warn("Failed to load profile DB:", err);
    });

    // Wire up toolbar buttons
    this._wireToolbar();
    this._syncResultSelect();
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

  getSessionSnapshot() {
    const modelSnapshot = this.model?.serialize ? this.model.serialize() : null;
    return {
      model_snapshot: modelSnapshot,
      results_snapshot: this.results || null,
      model_summary: {
        analysis_type: this.model?.analysisType || "frame3d",
        node_count: Object.keys(this.model?.nodes || {}).length,
        element_count: Object.keys(this.model?.elements || {}).length,
        load_case_ids: Object.keys(this.model?.loadCases || {}),
        solved: !!this.results,
      },
    };
  }

  async restoreSessionSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== "object") return;
    const modelData = snapshot.model_snapshot;
    if (!modelData || typeof modelData !== "object") return;

    await this._profileDbPromise.catch(() => null);
    this.model = FEAModel.deserialize(modelData);
    if (this._profileDb) {
      this.model.setProfileDb(this._profileDb);
    }
    this.results = snapshot.results_snapshot && typeof snapshot.results_snapshot === "object"
      ? snapshot.results_snapshot
      : null;
    this._syncResultSelect();

    try {
      await this._ensureViewer();
      this._rebuildViewer();
      this._applyCurrentResultView();
    } catch (err) {
      console.warn("FEA viewer restore skipped:", err.message);
    }
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
    this._syncResultSelect();

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
      this.worker = new Worker("/static/fea/worker/fea_worker.js?v=5", { type: "module" });
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
          this._syncResultSelect();
          this._applyCurrentResultView();
          const resultsTable = this.panelEl.querySelector(".fea-results-table");
          if (resultsTable && !resultsTable.classList.contains("hidden")) {
            this.populateResultsTable();
          }

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
    const resolvedType = this._resolveForceComponent(forceType);

    const group = _resultModule.createForceDiagram(
      this.model, this.results.elementForces, resolvedType, scaleFactor || 1,
    );
    if (group) this.scene.addResultMesh(group);
  }

  _resolveForceComponent(forceType) {
    const mv = this.results?.maxValues || {};
    if (forceType === "M") {
      const direction = mv.maxMoment?.direction;
      if (direction === "My" || direction === "Mz" || direction === "Mx" || direction === "M") {
        return direction;
      }
    }
    if (forceType === "V") {
      const direction = mv.maxShear?.direction || mv.maxShearForce?.direction;
      if (direction === "Vy" || direction === "Vz" || direction === "V") {
        return direction;
      }
    }
    return forceType;
  }

  _applyCurrentResultView() {
    if (!this.scene) return;
    const select = this.panelEl.querySelector(".fea-result-select");
    const current = this._normalizeResultView(this._currentResultView || select?.value || "none");
    if (current === "none") {
      this.scene.clearResults();
      return;
    }
    if (current === "deformed") {
      this._showDeformedShape();
      return;
    }
    if (current === "moment") {
      this._showForceDiagram("M");
      return;
    }
    if (current === "shear") {
      this._showForceDiagram("V");
      return;
    }
    if (current === "axial") {
      this._showForceDiagram("N");
      return;
    }
    if (current === "torsion_mx") {
      this._showForceDiagram("Mx");
      return;
    }
    if (current === "moment_my") {
      this._showForceDiagram("My");
      return;
    }
    if (current === "moment_mz") {
      this._showForceDiagram("Mz");
      return;
    }
    if (current === "shear_vy") {
      this._showForceDiagram("Vy");
      return;
    }
    if (current === "shear_vz") {
      this._showForceDiagram("Vz");
      return;
    }
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
    const maxD_mm = mv.maxDisplacement ? mv.maxDisplacement.value.toFixed(3) : "—";
    const forceRows = this._buildForceSummaryRows();
    const forceRowsHtml = forceRows.map(row => (
      `<tr><td>${row.label}</td><td>${row.value}</td></tr>`
    )).join("");
    const forceNote = this._isComponentForceView()
      ? '<tr><td colspan="2">3D frame force diagrams use member local axes.</td></tr>'
      : "";

    container.innerHTML = `
      <table class="fea-summary-table">
        <tr><th colspan="2">Solver Info</th></tr>
        <tr><td>DOF count</td><td>${si.dofCount || "—"}</td></tr>
        <tr><td>Elements</td><td>${si.elementCount || "—"}</td></tr>
        <tr><td>Solve time</td><td>${si.solveTimeMs || "—"} ms</td></tr>
        <tr><th colspan="2">Max Values</th></tr>
        <tr><td>Max displacement</td><td>${maxD_mm} mm (${mv.maxDisplacement?.direction || ""}, node ${mv.maxDisplacement?.nodeId || "—"})</td></tr>
        ${forceRowsHtml}
        ${forceNote}
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
        this._currentResultView = this._normalizeResultView(select.value || "none");
        this._applyCurrentResultView();
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
    const header = this.panelEl.querySelector(".panel-header");
    const collapseBtn = this.panelEl.querySelector(".panel-collapse-btn");
    if (collapseBtn) {
      collapseBtn.addEventListener("click", (event) => {
        event.stopPropagation();
        this.panelEl.classList.toggle("collapsed");
      });
    }
    if (header) {
      header.addEventListener("click", (event) => {
        if (
          this.panelEl.classList.contains("collapsed")
          && !event.target.closest(".panel-collapse-btn")
        ) {
          this.panelEl.classList.remove("collapsed");
        }
      });
    }
  }

  // ── Cleanup ────────────────────────────────────────────────────

  dispose() {
    if (this.scene) this.scene.dispose();
    if (this.worker) this.worker.terminate();
  }

  _isComponentForceView() {
    if (this.model?.analysisType === "frame3d") return true;
    const elementForces = this.results?.elementForces || {};
    return Object.values(elementForces).some(forces => (
      Array.isArray(forces?.Mx) || Array.isArray(forces?.My) || Array.isArray(forces?.Mz)
      || Array.isArray(forces?.Vy) || Array.isArray(forces?.Vz)
    ));
  }

  _hasForceSeries(component) {
    const elementForces = this.results?.elementForces || {};
    const hasComponent = Object.values(elementForces).some(forces => (
      Array.isArray(forces?.[component]) && forces[component].length > 0
    ));
    if (hasComponent) return true;
    if (!this.results && this.model?.analysisType === "frame3d") {
      return ["N", "Mx", "My", "Mz", "Vy", "Vz"].includes(component);
    }
    if (!this.results && this.model?.analysisType === "beam2d") {
      return ["N", "M", "V"].includes(component);
    }
    return false;
  }

  _normalizeResultView(view) {
    if (!this._isComponentForceView()) return view;
    if (view === "moment") {
      const direction = this._resolveForceComponent("M");
      if (direction === "Mx") return "torsion_mx";
      if (direction === "My") return "moment_my";
      if (direction === "Mz") return "moment_mz";
      return this._hasForceSeries("Mz") ? "moment_mz" : "moment_my";
    }
    if (view === "shear") {
      const direction = this._resolveForceComponent("V");
      if (direction === "Vz") return "shear_vz";
      return "shear_vy";
    }
    return view;
  }

  _buildResultOptions() {
    const options = [
      { value: "none", label: "Model" },
      { value: "deformed", label: "Deformed" },
    ];

    if (this._isComponentForceView()) {
      if (this._hasForceSeries("N")) options.push({ value: "axial", label: "Axial N" });
      if (this._hasForceSeries("Mx")) options.push({ value: "torsion_mx", label: "Torsion Mx" });
      if (this._hasForceSeries("My")) options.push({ value: "moment_my", label: "Bending My" });
      if (this._hasForceSeries("Mz")) options.push({ value: "moment_mz", label: "Bending Mz" });
      if (this._hasForceSeries("Vy")) options.push({ value: "shear_vy", label: "Shear Vy" });
      if (this._hasForceSeries("Vz")) options.push({ value: "shear_vz", label: "Shear Vz" });
      return options;
    }

    options.push({ value: "moment", label: "Moment (M)" });
    options.push({ value: "shear", label: "Shear (V)" });
    options.push({ value: "axial", label: "Axial (N)" });
    return options;
  }

  _syncResultSelect() {
    const select = this.panelEl.querySelector(".fea-result-select");
    if (!select) return;

    const previousValue = this._normalizeResultView(select.value || this._currentResultView || "none");
    const options = this._buildResultOptions();
    select.innerHTML = options.map(option => (
      `<option value="${option.value}">${option.label}</option>`
    )).join("");

    const validValues = new Set(options.map(option => option.value));
    const nextValue = validValues.has(previousValue) ? previousValue : "none";
    select.value = nextValue;
    this._currentResultView = nextValue;
  }

  _getForceExtrema(component) {
    let best = null;
    const elementForces = this.results?.elementForces || {};
    for (const [elementId, forces] of Object.entries(elementForces)) {
      const series = Array.isArray(forces?.[component]) ? forces[component] : [];
      for (const value of series) {
        if (!best || Math.abs(value) > Math.abs(best.value)) {
          best = { elementId, value };
        }
      }
    }
    return best;
  }

  _formatForceSummary(component, label) {
    const force = this._getForceExtrema(component);
    if (!force) return null;
    const isMoment = component.startsWith("M");
    const scaled = isMoment ? (force.value / 1e6).toFixed(2) : (force.value / 1e3).toFixed(2);
    const unit = isMoment ? "kN·m" : "kN";
    return {
      label,
      value: `${scaled} ${unit} (elem ${force.elementId})`,
    };
  }

  _buildForceSummaryRows() {
    if (this._isComponentForceView()) {
      return [
        this._formatForceSummary("N", "Max axial force"),
        this._formatForceSummary("Mx", "Max torsion Mx"),
        this._formatForceSummary("My", "Max bending My"),
        this._formatForceSummary("Mz", "Max bending Mz"),
        this._formatForceSummary("Vy", "Max shear Vy"),
        this._formatForceSummary("Vz", "Max shear Vz"),
      ].filter(Boolean);
    }

    return [
      this._formatForceSummary("N", "Max axial force"),
      this._formatForceSummary("M", "Max bending moment"),
      this._formatForceSummary("V", "Max shear force"),
    ].filter(Boolean);
  }
}
