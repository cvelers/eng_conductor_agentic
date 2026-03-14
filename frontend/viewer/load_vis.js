/**
 * Load visualisation — creates arrows and symbols for applied loads.
 */

import { getThree } from "./scene.js";
import { getSelfWeightDistributedLoad } from "../fea/load_helpers.js";

const FORCE_COLOR = 0xff4444;    // red for forces
const MOMENT_COLOR = 0x4488ff;   // blue for moments
const DIST_COLOR = 0xff6644;     // orange for distributed
const ARROW_SCALE = 0.3;         // relative to max model dimension

/**
 * Create an arrow for a nodal point load.
 */
export function createNodalForceArrow(position, fx, fy, fz, maxDim) {
  const THREE = getThree();
  if (!THREE) return null;

  const group = new THREE.Group();
  const arrowLen = maxDim * ARROW_SCALE;

  const components = [
    { val: fx, dir: new THREE.Vector3(1, 0, 0) },
    { val: fy, dir: new THREE.Vector3(0, 1, 0) },
    { val: fz, dir: new THREE.Vector3(0, 0, 1) },
  ];

  for (const { val, dir } of components) {
    if (Math.abs(val) < 1e-6) continue;
    const len = arrowLen * 0.5;
    // Arrow points toward the node (applied force direction)
    const arrowDir = dir.clone().multiplyScalar(val > 0 ? 1 : -1);
    const origin = new THREE.Vector3(
      position.x - arrowDir.x * len,
      position.y - arrowDir.y * len,
      position.z - arrowDir.z * len,
    );
    const arrow = new THREE.ArrowHelper(
      arrowDir, origin, len,
      FORCE_COLOR, len * 0.2, len * 0.12,
    );
    group.add(arrow);
  }

  group.userData = { type: "load", kind: "nodal_force" };
  return group;
}

/**
 * Create a moment arc indicator at a node.
 */
export function createNodalMomentArc(position, mx, my, mz, maxDim) {
  const THREE = getThree();
  if (!THREE) return null;

  const group = new THREE.Group();
  const arcRadius = maxDim * ARROW_SCALE * 0.3;

  const components = [
    { val: mx, axis: "x" },
    { val: my, axis: "y" },
    { val: mz, axis: "z" },
  ];

  for (const { val, axis } of components) {
    if (Math.abs(val) < 1e-6) continue;

    const curve = new THREE.EllipseCurve(0, 0, arcRadius, arcRadius, 0, Math.PI * 1.5, false);
    const points = curve.getPoints(24);
    const geo = new THREE.BufferGeometry().setFromPoints(
      points.map(p => new THREE.Vector3(p.x, p.y, 0)),
    );
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: MOMENT_COLOR, linewidth: 2 }));

    // Rotate arc to correct plane
    if (axis === "x") line.rotation.y = Math.PI / 2;
    else if (axis === "y") line.rotation.x = -Math.PI / 2;

    line.position.set(position.x, position.y, position.z);
    group.add(line);
  }

  group.userData = { type: "load", kind: "nodal_moment" };
  return group;
}

/**
 * Create distributed load arrows along a member.
 */
export function createDistributedLoadVis(node1, node2, qx, qy, qz, maxDim) {
  const THREE = getThree();
  if (!THREE) return null;

  const group = new THREE.Group();

  const p1 = new THREE.Vector3(node1.x, node1.y, node1.z);
  const p2 = new THREE.Vector3(node2.x, node2.y, node2.z);
  const memberLen = p1.distanceTo(p2);
  if (memberLen < 1) return null;

  // Load direction in global coordinates
  const qMag = Math.sqrt(qx * qx + qy * qy + qz * qz);
  if (qMag < 1e-6) return null;

  const loadDir = new THREE.Vector3(qx, qy, qz).normalize();
  const arrowLen = maxDim * ARROW_SCALE * 0.4;

  // Number of arrows along member
  const nArrows = Math.max(5, Math.min(15, Math.round(memberLen / (maxDim * 0.05))));

  const arrowOrigins = [];
  for (let i = 0; i <= nArrows; i++) {
    const t = i / nArrows;
    const pt = new THREE.Vector3().lerpVectors(p1, p2, t);
    arrowOrigins.push(pt);

    // Arrow at this station
    const origin = pt.clone().sub(loadDir.clone().multiplyScalar(arrowLen));
    const arrow = new THREE.ArrowHelper(
      loadDir, origin, arrowLen,
      DIST_COLOR, arrowLen * 0.15, arrowLen * 0.08,
    );
    group.add(arrow);
  }

  // Connect arrow tips with a line
  const tipPoints = arrowOrigins.map(p => p.clone());
  const lineGeo = new THREE.BufferGeometry().setFromPoints(tipPoints);
  const lineMat = new THREE.LineBasicMaterial({ color: DIST_COLOR });
  group.add(new THREE.Line(lineGeo, lineMat));

  // Connect arrow bases with a line
  const basePoints = arrowOrigins.map(p =>
    p.clone().sub(loadDir.clone().multiplyScalar(arrowLen)),
  );
  const baseGeo = new THREE.BufferGeometry().setFromPoints(basePoints);
  group.add(new THREE.Line(baseGeo, lineMat.clone()));

  group.userData = { type: "load", kind: "distributed" };
  return group;
}

function createSelfWeightVis(model, element, load, maxDim) {
  const node1 = model.nodes[element.nodeIds[0]];
  const node2 = model.nodes[element.nodeIds[1]];
  if (!node1 || !node2) return null;

  const gravity = getSelfWeightDistributedLoad(model, element, load);
  if (!gravity) return null;
  return createDistributedLoadVis(
    node1,
    node2,
    gravity.qx,
    gravity.qy,
    gravity.qz,
    maxDim,
  );
}

/**
 * Build all load visualizations for a model and load case.
 */
export function buildAllLoads(model, loadCaseId, maxDim) {
  const meshes = [];
  const lc = model.loadCases[loadCaseId];
  if (!lc) return meshes;

  for (const load of lc.loads) {
    if (load.type === "nodal") {
      const node = model.nodes[load.node_id || load.nodeId];
      if (!node) continue;

      const fx = load.fx || 0, fy = load.fy || 0, fz = load.fz || 0;
      const mx = load.mx || 0, my = load.my || 0, mz = load.mz || 0;

      if (Math.abs(fx) + Math.abs(fy) + Math.abs(fz) > 1e-6) {
        const arrow = createNodalForceArrow(node, fx, fy, fz, maxDim);
        if (arrow) meshes.push(arrow);
      }
      if (Math.abs(mx) + Math.abs(my) + Math.abs(mz) > 1e-6) {
        const arc = createNodalMomentArc(node, mx, my, mz, maxDim);
        if (arc) meshes.push(arc);
      }
    }

    if (load.type === "distributed") {
      const elem = model.elements[load.element_id || load.elementId];
      if (!elem || elem.nodeIds.length < 2) continue;
      const n1 = model.nodes[elem.nodeIds[0]];
      const n2 = model.nodes[elem.nodeIds[1]];
      if (!n1 || !n2) continue;

      const qx = load.qx || 0, qy = load.qy || 0, qz = load.qz || 0;
      const vis = createDistributedLoadVis(n1, n2, qx, qy, qz, maxDim);
      if (vis) meshes.push(vis);
    }

    if (load.type === "self_weight") {
      for (const elem of Object.values(model.elements)) {
        if (!elem || !Array.isArray(elem.nodeIds) || elem.nodeIds.length < 2) continue;
        const vis = createSelfWeightVis(model, elem, load, maxDim);
        if (vis) meshes.push(vis);
      }
    }
  }

  return meshes;
}
