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
    this._animId = null;
    this._ready = false;
  }

  async init() {
    await loadThree();
    const width = this.container.clientWidth || 600;
    const height = this.container.clientHeight || 400;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);

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
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5000, 10000, 7000);
    this.scene.add(dirLight);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dirLight2.position.set(-3000, -2000, -5000);
    this.scene.add(dirLight2);

    // Grid
    const grid = new THREE.GridHelper(20000, 40, 0x333355, 0x222244);
    grid.rotation.x = 0; // XZ plane
    this.scene.add(grid);
    this._grid = grid;

    // Axes
    const axes = new THREE.AxesHelper(1000);
    this.scene.add(axes);

    // Resize observer
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.container);

    this._ready = true;
    this._animate();
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
  }

  // ── Member management ──────────────────────────────────────────

  addMember(elemId, mesh) {
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
    this.loadMeshes.push(mesh);
    this.scene.add(mesh);
  }

  clearLoads() {
    for (const m of this.loadMeshes) {
      this.scene.remove(m);
      if (m.geometry) m.geometry.dispose();
      if (m.material && m.material.dispose) m.material.dispose();
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
    if (this.renderer) this.renderer.dispose();
    if (this.controls) this.controls.dispose();
  }
}
