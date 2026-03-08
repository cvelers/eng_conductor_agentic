/**
 * Result visualisation — deformed shapes, color contours, force diagrams.
 */

import { getThree } from "./scene.js";

// Color gradient: blue (low) → green → yellow → red (high)
const CONTOUR_COLORS = [
  [0, 0, 1],      // blue
  [0, 0.5, 1],
  [0, 1, 0.5],    // cyan-green
  [0, 1, 0],      // green
  [0.5, 1, 0],
  [1, 1, 0],      // yellow
  [1, 0.5, 0],    // orange
  [1, 0, 0],      // red
];

/**
 * Interpolate color from gradient based on normalized value [0, 1].
 */
function getContourColor(t) {
  t = Math.max(0, Math.min(1, t));
  const idx = t * (CONTOUR_COLORS.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, CONTOUR_COLORS.length - 1);
  const frac = idx - lo;
  return [
    CONTOUR_COLORS[lo][0] + (CONTOUR_COLORS[hi][0] - CONTOUR_COLORS[lo][0]) * frac,
    CONTOUR_COLORS[lo][1] + (CONTOUR_COLORS[hi][1] - CONTOUR_COLORS[lo][1]) * frac,
    CONTOUR_COLORS[lo][2] + (CONTOUR_COLORS[hi][2] - CONTOUR_COLORS[lo][2]) * frac,
  ];
}

/**
 * Create deformed shape visualization for beam/frame structures.
 * Draws deformed member lines with scaled displacements.
 *
 * @param {Object} model - FEA model
 * @param {Object} displacements - { nodeId: { dx, dy, dz, ... } }
 * @param {number} scaleFactor - displacement magnification
 * @returns {THREE.Group}
 */
export function createDeformedShape(model, displacements, scaleFactor = 50) {
  const THREE = getThree();
  if (!THREE) return null;

  const group = new THREE.Group();
  const defColor = 0x4f9cf7;
  const origColor = 0x444466;

  for (const [elemId, elem] of Object.entries(model.elements)) {
    if (elem.nodeIds.length < 2) continue;
    const nid1 = elem.nodeIds[0];
    const nid2 = elem.nodeIds[1];
    const n1 = model.nodes[nid1];
    const n2 = model.nodes[nid2];
    if (!n1 || !n2) continue;

    const d1 = displacements[nid1] || { dx: 0, dy: 0, dz: 0 };
    const d2 = displacements[nid2] || { dx: 0, dy: 0, dz: 0 };

    // Original shape (dashed)
    const origPts = [
      new THREE.Vector3(n1.x, n1.y, n1.z),
      new THREE.Vector3(n2.x, n2.y, n2.z),
    ];
    const origGeo = new THREE.BufferGeometry().setFromPoints(origPts);
    const origLine = new THREE.Line(origGeo,
      new THREE.LineDashedMaterial({ color: origColor, dashSize: 50, gapSize: 30 }),
    );
    origLine.computeLineDistances();
    group.add(origLine);

    // Deformed shape (solid, color-coded by displacement magnitude)
    // Interpolate along member with multiple stations for curved appearance
    const nStations = 11;
    const defPts = [];
    for (let i = 0; i <= nStations; i++) {
      const t = i / nStations;
      // Linear interpolation of displacement (simplified — proper beam shape would use shape functions)
      const px = n1.x + (n2.x - n1.x) * t + (d1.dx + (d2.dx - d1.dx) * t) * scaleFactor;
      const py = n1.y + (n2.y - n1.y) * t + (d1.dy + (d2.dy - d1.dy) * t) * scaleFactor;
      const pz = n1.z + (n2.z - n1.z) * t + (d1.dz + (d2.dz - d1.dz) * t) * scaleFactor;
      defPts.push(new THREE.Vector3(px, py, pz));
    }

    const defGeo = new THREE.BufferGeometry().setFromPoints(defPts);
    const defLine = new THREE.Line(defGeo,
      new THREE.LineBasicMaterial({ color: defColor, linewidth: 2 }),
    );
    group.add(defLine);
  }

  group.userData = { type: "result", kind: "deformed" };
  return group;
}

/**
 * Create internal force diagram along beam elements.
 *
 * @param {Object} model - FEA model
 * @param {Object} elementForces - { elemId: { stations, M, V, N } }
 * @param {string} forceType - "M" | "V" | "N" (bending moment, shear, axial)
 * @param {number} scaleFactor - diagram magnification
 * @returns {THREE.Group}
 */
export function createForceDiagram(model, elementForces, forceType = "M", scaleFactor = 1) {
  const THREE = getThree();
  if (!THREE) return null;

  const group = new THREE.Group();

  // Find max value for scaling
  let maxVal = 0;
  for (const ef of Object.values(elementForces)) {
    const vals = ef[forceType] || ef.M || [];
    for (const v of vals) {
      if (Math.abs(v) > maxVal) maxVal = Math.abs(v);
    }
  }
  if (maxVal < 1e-10) return group;

  // Get model size for auto-scaling the diagram height
  const bbox = model.getBoundingBox ? model.getBoundingBox() : { min: { x: 0, y: 0, z: 0 }, max: { x: 1000, y: 1000, z: 1000 } };
  const modelSize = Math.max(
    bbox.max.x - bbox.min.x,
    bbox.max.y - bbox.min.y,
    bbox.max.z - bbox.min.z,
  ) || 1000;
  const diagramHeight = modelSize * 0.3 * scaleFactor;

  const colorMap = {
    M: 0xff4466,  // red-pink for moment
    V: 0x44aaff,  // blue for shear
    N: 0x44ff88,  // green for axial
  };
  const color = colorMap[forceType] || 0xffffff;

  for (const [elemId, elem] of Object.entries(model.elements)) {
    const ef = elementForces[elemId];
    if (!ef) continue;

    const vals = ef[forceType] || [];
    const stations = ef.stations || [];
    if (vals.length === 0 || stations.length === 0) continue;

    if (elem.nodeIds.length < 2) continue;
    const n1 = model.nodes[elem.nodeIds[0]];
    const n2 = model.nodes[elem.nodeIds[1]];
    if (!n1 || !n2) continue;

    // Member direction
    const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-6) continue;

    // Perpendicular direction for diagram offset
    // For 2D (XY plane): perpendicular is (-dy/L, dx/L, 0)
    // For 3D: use cross product with up vector
    let perpX, perpY, perpZ;
    if (Math.abs(dz) < 1e-6 && Math.abs(dx) > 1e-6 || Math.abs(dy) > 1e-6) {
      // 2D case (XY plane)
      perpX = -dy / L;
      perpY = dx / L;
      perpZ = 0;
    } else {
      // 3D case: cross with global Z
      perpX = dy / L;
      perpY = -dx / L;
      perpZ = 0;
      const pLen = Math.sqrt(perpX * perpX + perpY * perpY);
      if (pLen > 1e-6) { perpX /= pLen; perpY /= pLen; }
      else { perpX = 0; perpY = 1; perpZ = 0; }
    }

    // Build diagram outline
    const basePoints = [];
    const diagPoints = [];

    for (let i = 0; i < stations.length; i++) {
      const t = stations[i];
      const bx = n1.x + dx * t;
      const by = n1.y + dy * t;
      const bz = n1.z + dz * t;
      basePoints.push(new THREE.Vector3(bx, by, bz));

      const offset = (vals[i] / maxVal) * diagramHeight;
      diagPoints.push(new THREE.Vector3(
        bx + perpX * offset,
        by + perpY * offset,
        bz + perpZ * offset,
      ));
    }

    // Diagram line
    const diagGeo = new THREE.BufferGeometry().setFromPoints(diagPoints);
    group.add(new THREE.Line(diagGeo, new THREE.LineBasicMaterial({ color })));

    // Closing lines (connect diagram ends to member)
    if (basePoints.length > 0) {
      const close1 = new THREE.BufferGeometry().setFromPoints([basePoints[0], diagPoints[0]]);
      group.add(new THREE.Line(close1, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 })));

      const last = basePoints.length - 1;
      const close2 = new THREE.BufferGeometry().setFromPoints([basePoints[last], diagPoints[last]]);
      group.add(new THREE.Line(close2, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 })));
    }

    // Fill (semi-transparent shape between member and diagram)
    if (basePoints.length >= 2) {
      const vertices = [];
      for (let i = 0; i < basePoints.length - 1; i++) {
        // Two triangles per segment
        vertices.push(
          basePoints[i].x, basePoints[i].y, basePoints[i].z,
          diagPoints[i].x, diagPoints[i].y, diagPoints[i].z,
          basePoints[i + 1].x, basePoints[i + 1].y, basePoints[i + 1].z,

          diagPoints[i].x, diagPoints[i].y, diagPoints[i].z,
          diagPoints[i + 1].x, diagPoints[i + 1].y, diagPoints[i + 1].z,
          basePoints[i + 1].x, basePoints[i + 1].y, basePoints[i + 1].z,
        );
      }
      const fillGeo = new THREE.BufferGeometry();
      fillGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      fillGeo.computeVertexNormals();

      const fillMat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide,
      });
      group.add(new THREE.Mesh(fillGeo, fillMat));
    }
  }

  group.userData = { type: "result", kind: `diagram_${forceType}` };
  return group;
}

/**
 * Create a color bar legend as an HTML element.
 */
export function createColorBarHTML(minVal, maxVal, label, unit) {
  const container = document.createElement("div");
  container.className = "fea-color-bar-legend";
  container.innerHTML = `
    <div class="fea-cb-title">${label} (${unit})</div>
    <div class="fea-cb-gradient"></div>
    <div class="fea-cb-labels">
      <span>${minVal.toFixed(2)}</span>
      <span>${((minVal + maxVal) / 2).toFixed(2)}</span>
      <span>${maxVal.toFixed(2)}</span>
    </div>
  `;
  return container;
}

/**
 * Compute auto scale factor for deformed shape.
 */
export function autoDeformScale(model, displacements) {
  const bbox = model.getBoundingBox ? model.getBoundingBox() : null;
  if (!bbox) return 50;
  const modelSize = Math.max(
    bbox.max.x - bbox.min.x,
    bbox.max.y - bbox.min.y,
    bbox.max.z - bbox.min.z,
  ) || 1;

  let maxDisp = 0;
  for (const d of Object.values(displacements)) {
    const mag = Math.sqrt((d.dx || 0) ** 2 + (d.dy || 0) ** 2 + (d.dz || 0) ** 2);
    if (mag > maxDisp) maxDisp = mag;
  }

  if (maxDisp < 1e-10) return 1;
  // Target: deformed shape is ~10% of model size
  return (modelSize * 0.1) / maxDisp;
}
