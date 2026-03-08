/**
 * Support/restraint visualisation — creates 3D symbols at restrained nodes.
 */

import { getThree } from "./scene.js";

const SUPPORT_COLOR = 0x22cc66;
const SUPPORT_SCALE = 150; // base size in mm

/**
 * Create a pin support symbol (triangle).
 */
export function createPinSupport(position) {
  const THREE = getThree();
  if (!THREE) return null;
  const s = SUPPORT_SCALE;

  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.lineTo(-s * 0.6, -s);
  shape.lineTo(s * 0.6, -s);
  shape.closePath();

  const geometry = new THREE.ShapeGeometry(shape);
  const material = new THREE.MeshBasicMaterial({
    color: SUPPORT_COLOR,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.7,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(position.x, position.y, position.z);
  mesh.userData = { type: "support", kind: "pin" };

  // Also add hatch lines below
  const group = new THREE.Group();
  group.add(mesh);

  const hatch = _createHatchLines(THREE, s);
  hatch.position.set(0, -s, 0);
  group.add(hatch);
  group.position.set(position.x, position.y, position.z);
  mesh.position.set(0, 0, 0);

  return group;
}

/**
 * Create a roller support symbol (triangle + circle).
 */
export function createRollerSupport(position) {
  const THREE = getThree();
  if (!THREE) return null;
  const s = SUPPORT_SCALE;

  const group = new THREE.Group();

  // Triangle
  const triShape = new THREE.Shape();
  triShape.moveTo(0, 0);
  triShape.lineTo(-s * 0.6, -s * 0.8);
  triShape.lineTo(s * 0.6, -s * 0.8);
  triShape.closePath();

  const triGeo = new THREE.ShapeGeometry(triShape);
  const material = new THREE.MeshBasicMaterial({
    color: SUPPORT_COLOR,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.7,
  });
  group.add(new THREE.Mesh(triGeo, material));

  // Circle (roller)
  const circleGeo = new THREE.RingGeometry(s * 0.12, s * 0.18, 16);
  const circleMesh = new THREE.Mesh(circleGeo, material.clone());
  circleMesh.position.set(-s * 0.3, -s * 0.95, 0);
  group.add(circleMesh);

  const circleMesh2 = new THREE.Mesh(circleGeo.clone(), material.clone());
  circleMesh2.position.set(s * 0.3, -s * 0.95, 0);
  group.add(circleMesh2);

  // Hatch line
  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-s * 0.7, -s * 1.15, 0),
    new THREE.Vector3(s * 0.7, -s * 1.15, 0),
  ]);
  const lineMat = new THREE.LineBasicMaterial({ color: SUPPORT_COLOR });
  group.add(new THREE.Line(lineGeo, lineMat));

  group.position.set(position.x, position.y, position.z);
  group.userData = { type: "support", kind: "roller" };
  return group;
}

/**
 * Create a fixed support symbol (hatched block).
 */
export function createFixedSupport(position) {
  const THREE = getThree();
  if (!THREE) return null;
  const s = SUPPORT_SCALE;

  const group = new THREE.Group();

  // Block
  const blockGeo = new THREE.PlaneGeometry(s * 1.2, s * 0.5);
  const material = new THREE.MeshBasicMaterial({
    color: SUPPORT_COLOR,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.5,
  });
  const block = new THREE.Mesh(blockGeo, material);
  block.position.set(0, -s * 0.25, 0);
  group.add(block);

  // Hatch lines
  const hatch = _createHatchLines(THREE, s * 0.8);
  hatch.position.set(0, -s * 0.5, 0);
  group.add(hatch);

  group.position.set(position.x, position.y, position.z);
  group.userData = { type: "support", kind: "fixed" };
  return group;
}

/**
 * Auto-detect support type and create appropriate symbol.
 */
export function createSupportSymbol(position, restraint) {
  const { dx, dy, dz, rx, ry, rz } = restraint;
  const transCount = [dx, dy, dz].filter(Boolean).length;
  const rotCount = [rx, ry, rz].filter(Boolean).length;

  if (transCount >= 3 && rotCount >= 2) {
    return createFixedSupport(position);
  }
  if (transCount >= 2) {
    return createPinSupport(position);
  }
  return createRollerSupport(position);
}

// ── Internal helpers ─────────────────────────────────────────────

function _createHatchLines(THREE, width) {
  const group = new THREE.Group();
  const lineMat = new THREE.LineBasicMaterial({ color: SUPPORT_COLOR, transparent: true, opacity: 0.5 });
  const count = 5;
  const spacing = width / count;
  for (let i = 0; i <= count; i++) {
    const x = -width / 2 + i * spacing;
    const pts = [
      new THREE.Vector3(x, 0, 0),
      new THREE.Vector3(x - spacing * 0.6, -spacing * 0.8, 0),
    ];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    group.add(new THREE.Line(geo, lineMat));
  }
  return group;
}
