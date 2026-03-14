/**
 * Three.js scene manager for the FEA 3D viewer.
 * Lazy-loads Three.js via dynamic import.
 */

let THREE = null;
let OrbitControls = null;
let CSS2DRenderer = null;
let CSS2DObject = null;

const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js";
const ORBIT_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js";
const CSS2D_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/renderers/CSS2DRenderer.js";

async function loadThree() {
  if (THREE) return;
  THREE = await import(/* webpackIgnore: true */ THREE_CDN);
  const orbitMod = await import(/* webpackIgnore: true */ ORBIT_CDN);
  OrbitControls = orbitMod.OrbitControls;
  try {
    const css2dMod = await import(/* webpackIgnore: true */ CSS2D_CDN);
    CSS2DRenderer = css2dMod.CSS2DRenderer;
    CSS2DObject = css2dMod.CSS2DObject;
  } catch {
    // CSS2D optional — labels will be skipped if unavailable
  }
}

export function getThree() { return THREE; }

// ── Theme presets ───────────────────────────────────────────────
const THEMES = {
  dark: {
    background: 0x0e1015,
    gridMajor: 0x2a2d35,
    gridMinor: 0x1a1d24,
    groundPlane: 0x12141a,
    ambient: 0.55,
    name: "dark",
  },
  light: {
    background: 0xf0f2f5,
    gridMajor: 0xc0c4cc,
    gridMinor: 0xdcdfe6,
    groundPlane: 0xe8eaef,
    ambient: 0.75,
    name: "light",
  },
};

export class FEAScene {
  constructor(container) {
    this.container = container;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.labelRenderer = null;
    this.members = new Map();      // elemId → THREE.Mesh
    this.supportMeshes = [];
    this.loadMeshes = [];
    this.labelObjects = [];
    this.resultMeshes = [];
    this.profileBoxes = [];
    this._animId = null;
    this._ready = false;
    this._theme = "dark";
    this._grid = null;
    this._groundPlane = null;
    this._ambientLight = null;
    this._axes = null;
    this._lastBBox = null;
    this._diagramScale = 1.0;

    // Visibility state
    this._visibility = {
      members: true,
      loads: true,
      supports: true,
      labels: true,
      profiles: false,
      grid: true,
      axes: true,
    };
  }

  async init() {
    await loadThree();
    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 400;
    const theme = THEMES[this._theme];

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(theme.background);

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100000);
    this.camera.position.set(5000, 3000, 8000);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.container.querySelector("canvas.fea-canvas") || undefined,
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    if (!this.container.querySelector("canvas.fea-canvas")) {
      this.container.appendChild(this.renderer.domElement);
    }

    // Label renderer (CSS2D)
    if (CSS2DRenderer) {
      this.labelRenderer = new CSS2DRenderer();
      this.labelRenderer.setSize(width, height);
      this.labelRenderer.domElement.style.position = "absolute";
      this.labelRenderer.domElement.style.top = "0";
      this.labelRenderer.domElement.style.left = "0";
      this.labelRenderer.domElement.style.pointerEvents = "none";
      this.container.appendChild(this.labelRenderer.domElement);
    }

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.minDistance = 100;
    this.controls.maxDistance = 50000;

    // Lighting
    this._ambientLight = new THREE.AmbientLight(0xffffff, theme.ambient);
    this.scene.add(this._ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(5000, 10000, 7000);
    this.scene.add(dirLight);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dirLight2.position.set(-3000, -2000, -5000);
    this.scene.add(dirLight2);

    // Rim light from below-behind to highlight profile edges
    const dirLight3 = new THREE.DirectionalLight(0xaabbff, 0.25);
    dirLight3.position.set(-2000, -5000, 3000);
    this.scene.add(dirLight3);

    // Professional grid
    this._buildGrid(theme);

    // Axes (prominent, with labels)
    this._buildAxes();

    // Resize observer
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.container);

    this._ready = true;
    this._animate();
  }

  // ── Grid (no dimension labels) ───────────────────────────────

  _buildGrid(theme, bbox) {
    // Remove and dispose old grid
    if (this._grid) {
      this._grid.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
      this.scene.remove(this._grid);
      this._grid = null;
    }
    if (this._groundPlane) {
      this._groundPlane.geometry?.dispose();
      this._groundPlane.material?.dispose();
      this.scene.remove(this._groundPlane);
      this._groundPlane = null;
    }

    // Determine grid extent from model bbox or default
    const b = bbox || this._lastBBox || null;
    let gridSize = 20000;
    let gridCenter = { x: 0, z: 0 };
    let majorSpacing = 1000;  // mm

    if (b) {
      this._lastBBox = b;
      const spanX = (b.max.x - b.min.x) || 1000;
      const spanZ = (b.max.z - b.min.z) || 1000;
      const maxSpan = Math.max(spanX, spanZ, (b.max.y - b.min.y) || 1000);
      gridSize = maxSpan * 3;
      gridCenter = {
        x: (b.min.x + b.max.x) / 2,
        z: (b.min.z + b.max.z) / 2,
      };

      // Pick a nice major spacing
      const niceSteps = [100, 250, 500, 1000, 2000, 2500, 5000, 10000, 25000];
      const targetDivisions = 8;
      const idealStep = maxSpan / targetDivisions;
      majorSpacing = niceSteps.find(s => s >= idealStep) || niceSteps[niceSteps.length - 1];
    }

    const divisions = Math.round(gridSize / majorSpacing);
    const minorDivisions = Math.min(divisions * 4, 200);

    // Ground plane (subtle)
    const groundGeo = new THREE.PlaneGeometry(gridSize, gridSize);
    const groundMat = new THREE.MeshBasicMaterial({
      color: theme.groundPlane,
      transparent: true,
      opacity: 0.35,
      side: THREE.DoubleSide,
    });
    this._groundPlane = new THREE.Mesh(groundGeo, groundMat);
    this._groundPlane.rotation.x = -Math.PI / 2;
    this._groundPlane.position.set(gridCenter.x, -0.5, gridCenter.z);
    this.scene.add(this._groundPlane);

    // Grid lines
    const gridGroup = new THREE.Group();

    const minor = new THREE.GridHelper(gridSize, minorDivisions, theme.gridMinor, theme.gridMinor);
    minor.material.transparent = true;
    minor.material.opacity = 0.3;
    minor.position.set(gridCenter.x, 0, gridCenter.z);
    gridGroup.add(minor);

    const major = new THREE.GridHelper(gridSize, divisions, theme.gridMajor, theme.gridMajor);
    major.material.transparent = true;
    major.material.opacity = 0.6;
    major.position.set(gridCenter.x, 0.1, gridCenter.z);
    gridGroup.add(major);

    this._grid = gridGroup;
    this._grid.visible = this._visibility.grid;
    this._groundPlane.visible = this._visibility.grid;
    this.scene.add(gridGroup);
  }

  updateGrid(bbox) {
    const theme = THEMES[this._theme];
    this._buildGrid(theme, bbox);
  }

  // ── Axes with labels ─────────────────────────────────────────

  _buildAxes() {
    if (this._axes) {
      this._axes.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
      this.scene.remove(this._axes);
      this._axes = null;
    }

    const axesGroup = new THREE.Group();
    const axisLength = 2000;
    const headLength = axisLength * 0.08;
    const headWidth = axisLength * 0.03;

    const axes = [
      { dir: new THREE.Vector3(1, 0, 0), color: 0xff4444, label: "X" },
      { dir: new THREE.Vector3(0, 1, 0), color: 0x44cc44, label: "Y" },
      { dir: new THREE.Vector3(0, 0, 1), color: 0x4488ff, label: "Z" },
    ];

    for (const { dir, color, label } of axes) {
      // Arrow
      const arrow = new THREE.ArrowHelper(
        dir, new THREE.Vector3(0, 0, 0), axisLength,
        color, headLength, headWidth,
      );
      // Make the shaft thicker by replacing with a cylinder
      arrow.line.material.color.setHex(color);
      arrow.line.material.linewidth = 2;
      axesGroup.add(arrow);

      // CSS2D label at the tip
      if (CSS2DObject) {
        const div = document.createElement("div");
        div.textContent = label;
        div.style.cssText = `
          font-family: "JetBrains Mono", "SF Mono", monospace;
          font-size: 13px; font-weight: 800; color: #${color.toString(16).padStart(6, "0")};
          text-shadow: 0 0 6px rgba(0,0,0,0.8);
          pointer-events: none; user-select: none;
        `;
        const labelObj = new CSS2DObject(div);
        labelObj.position.copy(dir.clone().multiplyScalar(axisLength * 1.08));
        axesGroup.add(labelObj);
      }
    }

    this._axes = axesGroup;
    this._axes.visible = this._visibility.axes;
    this.scene.add(this._axes);
  }

  // ── Theme switching ────────────────────────────────────────────

  setTheme(themeName) {
    const theme = THEMES[themeName];
    if (!theme || !this._ready) return;
    this._theme = themeName;

    this.scene.background = new THREE.Color(theme.background);
    if (this._ambientLight) this._ambientLight.intensity = theme.ambient;
    this._buildGrid(theme, this._lastBBox);
  }

  getTheme() { return this._theme; }

  toggleTheme() {
    this.setTheme(this._theme === "dark" ? "light" : "dark");
    return this._theme;
  }

  // ── Visibility controls ────────────────────────────────────────

  setVisibility(layer, visible) {
    this._visibility[layer] = visible;

    switch (layer) {
      case "members":
        for (const m of this.members.values()) m.visible = visible;
        break;
      case "loads":
        for (const m of this.loadMeshes) m.visible = visible;
        break;
      case "supports":
        for (const m of this.supportMeshes) m.visible = visible;
        break;
      case "labels":
        for (const l of this.labelObjects) l.visible = visible;
        break;
      case "profiles":
        for (const b of this.profileBoxes) b.visible = visible;
        break;
      case "grid":
        if (this._grid) this._grid.visible = visible;
        if (this._groundPlane) this._groundPlane.visible = visible;
        break;
      case "axes":
        if (this._axes) this._axes.visible = visible;
        break;
    }
  }

  getVisibility(layer) { return this._visibility[layer]; }

  // ── Diagram scale ─────────────────────────────────────────────

  getDiagramScale() { return this._diagramScale; }

  setDiagramScale(scale) {
    this._diagramScale = Math.max(0.2, Math.min(5.0, scale));
  }

  adjustDiagramScale(delta) {
    this.setDiagramScale(this._diagramScale + delta);
    return this._diagramScale;
  }

  // Load scale follows diagram scale (single unified control)
  getLoadScale() { return this._diagramScale; }

  // ── Profile bounding boxes ─────────────────────────────────────

  addProfileBox(mesh) {
    mesh.visible = this._visibility.profiles;
    this.profileBoxes.push(mesh);
    this.scene.add(mesh);
  }

  clearProfileBoxes() {
    for (const m of this.profileBoxes) {
      this.scene.remove(m);
      if (m.geometry) m.geometry.dispose();
      if (m.material && m.material.dispose) m.material.dispose();
    }
    this.profileBoxes = [];
  }

  _animate() {
    if (!this._ready) return;
    this._animId = requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (this.labelRenderer) {
      this.labelRenderer.render(this.scene, this.camera);
    }
  }

  _onResize() {
    if (!this._ready) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    if (this.labelRenderer) this.labelRenderer.setSize(w, h);
  }

  // ── Camera controls ────────────────────────────────────────────

  fitToModel(bbox) {
    if (!bbox) return;
    const center = new THREE.Vector3(
      (bbox.min.x + bbox.max.x) / 2,
      (bbox.min.y + bbox.max.y) / 2,
      (bbox.min.z + bbox.max.z) / 2,
    );
    const size = new THREE.Vector3(
      bbox.max.x - bbox.min.x,
      bbox.max.y - bbox.min.y,
      bbox.max.z - bbox.min.z,
    );
    const maxDim = Math.max(size.x, size.y, size.z) || 1000;
    const dist = maxDim * 2;

    this.camera.position.set(center.x + dist * 0.6, center.y + dist * 0.4, center.z + dist * 0.8);
    this.controls.target.copy(center);
    this.controls.update();
  }

  setView(preset) {
    const target = this.controls.target.clone();
    const dist = this.camera.position.distanceTo(target) || 5000;
    const is2D = preset === "front" || preset === "top" || preset === "side";

    switch (preset) {
      case "front":
        this.camera.position.set(target.x, target.y, target.z + dist);
        break;
      case "top":
        this.camera.position.set(target.x, target.y + dist, target.z + 0.01);
        break;
      case "side":
        this.camera.position.set(target.x + dist, target.y, target.z);
        break;
      case "iso":
      default:
        this.camera.position.set(
          target.x + dist * 0.6,
          target.y + dist * 0.4,
          target.z + dist * 0.8,
        );
        break;
    }
    this.camera.lookAt(target);
    this.controls.update();

    // Show rulers in 2D views only
    this._currentViewPreset = preset;
    this._updateRulers(is2D ? preset : null);
  }

  _updateRulers(preset) {
    const hRuler = this.container.querySelector(".fea-ruler-h");
    const vRuler = this.container.querySelector(".fea-ruler-v");
    if (!hRuler || !vRuler) return;

    if (!preset) {
      hRuler.classList.add("hidden");
      vRuler.classList.add("hidden");
      return;
    }

    // Determine which world axes map to screen H and V
    // front (XY plane, camera on +Z): H = X, V = Y
    // side  (ZY plane, camera on +X): H = Z, V = Y
    // top   (XZ plane, camera on +Y): H = X, V = Z (inverted)
    let hAxis, vAxis;
    if (preset === "front") { hAxis = "x"; vAxis = "y"; }
    else if (preset === "side") { hAxis = "z"; vAxis = "y"; }
    else { hAxis = "x"; vAxis = "z"; }

    const w = this.container.clientWidth || 600;
    const h = this.container.clientHeight || 400;

    // Get world coords at screen corners
    const topLeft3 = new THREE.Vector3(-1, 1, 0.5).unproject(this.camera);
    const topRight3 = new THREE.Vector3(1, 1, 0.5).unproject(this.camera);
    const botLeft3 = new THREE.Vector3(-1, -1, 0.5).unproject(this.camera);

    const hWorldMin = topLeft3[hAxis];
    const hWorldMax = topRight3[hAxis];
    const vWorldMax = topLeft3[vAxis];
    const vWorldMin = botLeft3[vAxis];

    const hWorldPerPx = (hWorldMax - hWorldMin) / w;
    const vWorldPerPx = (vWorldMin - vWorldMax) / h;  // may be negative for top view

    const fmt = (mm) => {
      const v = Math.round(mm);
      if (Math.abs(v) >= 1000) return (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + "m";
      return v + "";
    };

    const niceStep = (range) => {
      const absRange = Math.abs(range);
      const niceSteps = [10, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 25000];
      const target = absRange / 8;
      return niceSteps.find(s => s >= target) || niceSteps[niceSteps.length - 1];
    };

    // Build horizontal ruler ticks
    const hRange = hWorldMax - hWorldMin;
    const hStep = niceStep(hRange);
    const hStart = Math.ceil(hWorldMin / hStep) * hStep;
    let hHTML = "";
    for (let val = hStart; val <= hWorldMax; val += hStep) {
      const px = (val - hWorldMin) / hWorldPerPx;
      if (px < 0 || px > w) continue;
      hHTML += `<div class="fea-ruler-tick" style="left:${px.toFixed(1)}px">${fmt(val)}</div>`;
    }
    hRuler.innerHTML = hHTML;
    hRuler.classList.remove("hidden");

    // Build vertical ruler ticks
    const vRange = vWorldMin - vWorldMax;
    const vStep = niceStep(vRange);
    const vHigh = Math.max(vWorldMin, vWorldMax);
    const vLow = Math.min(vWorldMin, vWorldMax);
    const vStartVal = Math.ceil(vLow / vStep) * vStep;
    let vHTML = "";
    for (let val = vStartVal; val <= vHigh; val += vStep) {
      // Map world value to pixel: top of viewport = vWorldMax
      const py = (val - vWorldMax) / vWorldPerPx;
      if (py < 0 || py > h) continue;
      vHTML += `<div class="fea-ruler-tick" style="top:${py.toFixed(1)}px">${fmt(val)}</div>`;
    }
    vRuler.innerHTML = vHTML;
    vRuler.classList.remove("hidden");
  }

  // ── Member management ──────────────────────────────────────────

  addMember(elemId, mesh) {
    mesh.visible = this._visibility.members;
    this.members.set(elemId, mesh);
    this.scene.add(mesh);
  }

  removeMember(elemId) {
    const mesh = this.members.get(elemId);
    if (mesh) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material.dispose) mesh.material.dispose();
      this.members.delete(elemId);
    }
  }

  clearMembers() {
    for (const [id] of this.members) {
      this.removeMember(id);
    }
  }

  // ── Support/Load visualization management ──────────────────────

  addSupportMesh(mesh) {
    mesh.visible = this._visibility.supports;
    this.supportMeshes.push(mesh);
    this.scene.add(mesh);
  }

  clearSupports() {
    for (const m of this.supportMeshes) {
      this.scene.remove(m);
      if (m.geometry) m.geometry.dispose();
      if (m.material && m.material.dispose) m.material.dispose();
    }
    this.supportMeshes = [];
  }

  addLoadMesh(mesh) {
    mesh.visible = this._visibility.loads;
    this.loadMeshes.push(mesh);
    this.scene.add(mesh);
  }

  clearLoads() {
    for (const m of this.loadMeshes) {
      this.scene.remove(m);
      m.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach(mat => mat.dispose());
          else child.material.dispose();
        }
      });
    }
    this.loadMeshes = [];
  }

  addResultMesh(mesh) {
    this.resultMeshes.push(mesh);
    this.scene.add(mesh);
  }

  clearResults() {
    for (const m of this.resultMeshes) {
      this.scene.remove(m);
      if (m.geometry) m.geometry.dispose();
      if (m.material && m.material.dispose) m.material.dispose();
    }
    this.resultMeshes = [];
  }

  // ── Label management ───────────────────────────────────────────

  addLabel(text, position, className = "fea-label-3d") {
    if (!CSS2DObject) return null;
    const div = document.createElement("div");
    div.className = className;
    div.textContent = text;
    div.style.fontSize = "10px";
    div.style.color = "#ccc";
    div.style.background = "rgba(0,0,0,0.5)";
    div.style.padding = "1px 4px";
    div.style.borderRadius = "3px";
    div.style.pointerEvents = "none";
    const label = new CSS2DObject(div);
    label.position.set(position.x, position.y, position.z);
    label.visible = this._visibility.labels;
    this.scene.add(label);
    this.labelObjects.push(label);
    return label;
  }

  clearLabels() {
    for (const l of this.labelObjects) {
      this.scene.remove(l);
    }
    this.labelObjects = [];
  }

  // ── Cleanup ────────────────────────────────────────────────────

  dispose() {
    this._ready = false;
    if (this._animId) cancelAnimationFrame(this._animId);
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this.clearMembers();
    this.clearSupports();
    this.clearLoads();
    this.clearResults();
    this.clearLabels();
    this.clearProfileBoxes();
    if (this._grid) {
      this._grid.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
      this.scene.remove(this._grid);
      this._grid = null;
    }
    if (this._groundPlane) {
      this._groundPlane.geometry?.dispose();
      this._groundPlane.material?.dispose();
      this.scene.remove(this._groundPlane);
      this._groundPlane = null;
    }
    if (this._axes) {
      this._axes.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
      this.scene.remove(this._axes);
      this._axes = null;
    }
    if (this.renderer) this.renderer.dispose();
    if (this.controls) this.controls.dispose();
  }
}
