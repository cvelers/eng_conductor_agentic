/**
 * Global load vector assembly.
 *
 * Processes a load case and assembles the global force vector.
 * Supports nodal loads, distributed loads, and self-weight loads.
 *
 * Model structure:
 *   model.loadCases: Map/object { id: { loads: [...] } }
 *   Each load object:
 *     - type: "nodal"
 *       nodeId, fx, fy, fz, mx, my, mz  (all optional, default 0)
 *     - type: "distributed"
 *       elementId, qx, qy, qz  (global direction, force/length)
 *       OR: q (perpendicular to member, for beam2d)
 *     - type: "self_weight"
 *       factor: multiplier (typically 1.0), direction: {x,y,z} (unit gravity vector)
 */

import { beam2dEquivLoads } from '../elements/beam2d.js';
import { frame3dEquivLoads } from '../elements/frame3d.js';
import { elementLength, elementAngle } from './assembler.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Get a value from a Map or plain object by key.
 */
function getEntry(mapOrObj, key) {
  if (mapOrObj instanceof Map) return mapOrObj.get(key);
  return mapOrObj[key];
}

/**
 * Iterate entries of a Map or plain object.
 */
function entries(mapOrObj) {
  if (mapOrObj instanceof Map) return mapOrObj.entries();
  return Object.entries(mapOrObj);
}

// ---------------------------------------------------------------------------
// DOF mapping helpers for load vector
// ---------------------------------------------------------------------------

/**
 * Map of DOF names to their offset within a node's DOF block,
 * keyed by analysis type.
 */
const DOF_OFFSETS = {
  beam2d: { fx: 0, fy: 1, mz: 2 },
  frame3d: { fx: 0, fy: 1, fz: 2, mx: 3, my: 4, mz: 5 },
  truss2d: { fx: 0, fy: 1 },
  truss3d: { fx: 0, fy: 1, fz: 2 },
};

// ---------------------------------------------------------------------------
// Nodal load processing
// ---------------------------------------------------------------------------

/**
 * Add a nodal load to the global force vector.
 *
 * @param {Float64Array} F - Global force vector
 * @param {Object} load - Load definition { nodeId, fx, fy, fz, mx, my, mz }
 * @param {Map} nodeMap - DOF map
 * @param {string} analysisType
 */
function addNodalLoad(F, load, nodeMap, analysisType) {
  const startDOF = nodeMap.get(load.nodeId);
  if (startDOF === undefined) {
    throw new Error(`load_vector: node "${load.nodeId}" not found in DOF map`);
  }

  const offsets = DOF_OFFSETS[analysisType];
  if (!offsets) {
    throw new Error(`load_vector: unsupported analysis type "${analysisType}"`);
  }

  // Add each force/moment component if it exists in the offset map
  for (const [name, offset] of Object.entries(offsets)) {
    const value = load[name];
    if (value !== undefined && value !== 0) {
      F[startDOF + offset] += value;
    }
  }
}

// ---------------------------------------------------------------------------
// Distributed load processing
// ---------------------------------------------------------------------------

/**
 * Add equivalent nodal forces from a distributed load on an element.
 *
 * @param {Float64Array} F - Global force vector
 * @param {Object} load - Load definition { elementId, qx, qy, qz } or { elementId, q }
 * @param {Object} model
 * @param {Map} nodeMap
 * @param {number} dofPerNode
 */
function addDistributedLoad(F, load, model, nodeMap, dofPerNode) {
  const element = getEntry(model.elements, load.elementId);
  if (!element) {
    throw new Error(
      `load_vector: element "${load.elementId}" not found for distributed load`
    );
  }

  const [nid1, nid2] = element.nodeIds;
  const startDOF1 = nodeMap.get(nid1);
  const startDOF2 = nodeMap.get(nid2);
  const L = elementLength(model, element);
  const type = element.type || model.analysisType;

  let fEquiv;

  switch (type) {
    case 'beam2d': {
      // load.q = perpendicular distributed load (local transverse)
      // OR load.qx/qy in global coords
      const angle = elementAngle(model, element);

      if (load.q !== undefined) {
        // q is perpendicular to member in local coords
        fEquiv = beam2dEquivLoads(load.q, L, angle);
      } else {
        // Global qx, qy: project onto local perpendicular direction
        // Local transverse direction: (-sin(angle), cos(angle))
        // q_perp = qx * (-sin(angle)) + qy * cos(angle)
        const qx = load.qx || 0;
        const qy = load.qy || 0;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const qPerp = -qx * s + qy * c;
        // Also handle axial component: q_axial = qx*c + qy*s
        const qAxial = qx * c + qy * s;

        fEquiv = beam2dEquivLoads(qPerp, L, angle);
        // Add axial component (equally distributed to both nodes)
        const fAxialLocal = (qAxial * L) / 2;
        // Transform axial contributions to global
        fEquiv[0] += fAxialLocal * c; // fx1
        fEquiv[1] += fAxialLocal * s; // fy1
        fEquiv[3] += fAxialLocal * c; // fx2
        fEquiv[4] += fAxialLocal * s; // fy2
      }

      // Scatter into global F
      for (let d = 0; d < dofPerNode; d++) {
        F[startDOF1 + d] += fEquiv[d];
        F[startDOF2 + d] += fEquiv[dofPerNode + d];
      }
      break;
    }

    case 'frame3d': {
      const qx = load.qx || 0;
      const qy = load.qy || 0;
      const qz = load.qz || 0;
      const node1 = getEntry(model.nodes, nid1);
      const node2 = getEntry(model.nodes, nid2);

      fEquiv = frame3dEquivLoads(qx, qy, qz, L, node1, node2);

      // Scatter into global F
      for (let d = 0; d < dofPerNode; d++) {
        F[startDOF1 + d] += fEquiv[d];
        F[startDOF2 + d] += fEquiv[dofPerNode + d];
      }
      break;
    }

    case 'truss2d':
    case 'truss3d': {
      // Truss elements only carry axial load.
      // Distributed loads are converted to equivalent nodal forces.
      // Simply split the total load equally between both nodes.
      const qx = load.qx || 0;
      const qy = load.qy || 0;
      const qz = load.qz || 0;

      if (type === 'truss2d') {
        F[startDOF1 + 0] += (qx * L) / 2;
        F[startDOF1 + 1] += (qy * L) / 2;
        F[startDOF2 + 0] += (qx * L) / 2;
        F[startDOF2 + 1] += (qy * L) / 2;
      } else {
        F[startDOF1 + 0] += (qx * L) / 2;
        F[startDOF1 + 1] += (qy * L) / 2;
        F[startDOF1 + 2] += (qz * L) / 2;
        F[startDOF2 + 0] += (qx * L) / 2;
        F[startDOF2 + 1] += (qy * L) / 2;
        F[startDOF2 + 2] += (qz * L) / 2;
      }
      break;
    }

    default:
      throw new Error(
        `load_vector: unsupported element type "${type}" for distributed load`
      );
  }
}

// ---------------------------------------------------------------------------
// Self-weight load processing
// ---------------------------------------------------------------------------

/**
 * Add self-weight loads to the global force vector.
 *
 * For each element, compute the weight as A * rho * L * g * factor,
 * then distribute as equivalent nodal loads in the gravity direction.
 *
 * @param {Float64Array} F - Global force vector
 * @param {Object} load - { factor, direction: {x, y, z} } or { factor, g }
 * @param {Object} model
 * @param {Map} nodeMap
 * @param {number} dofPerNode
 */
function addSelfWeight(F, load, model, nodeMap, dofPerNode) {
  const factor = load.factor !== undefined ? load.factor : 1.0;
  const g = load.g || 9.81; // gravitational acceleration

  // Gravity direction (defaults to -Y for 2D, -Z for 3D)
  let gx, gy, gz;
  if (load.direction) {
    gx = (load.direction.x || 0) * g * factor;
    gy = (load.direction.y || 0) * g * factor;
    gz = (load.direction.z || 0) * g * factor;
  } else {
    // Default: gravity in negative Y for 2D, negative Z for 3D
    gx = 0;
    if (model.analysisType === 'frame3d' || model.analysisType === 'truss3d') {
      gy = 0;
      gz = -g * factor;
    } else {
      gy = -g * factor;
      gz = 0;
    }
  }

  for (const [elemId, element] of entries(model.elements)) {
    const section = getEntry(model.sections, element.sectionId);
    const material = getEntry(model.materials, element.materialId);

    if (!material.rho && material.rho !== 0) {
      continue; // skip elements without density
    }

    const A = section.A;
    const rho = material.rho;
    const L = elementLength(model, element);

    // Total weight of element
    const mass = A * rho * L;

    // Distributed self-weight per unit length in global coords
    const qx = rho * A * gx;
    const qy = rho * A * gy;
    const qz = rho * A * gz;

    const [nid1, nid2] = element.nodeIds;
    const startDOF1 = nodeMap.get(nid1);
    const startDOF2 = nodeMap.get(nid2);
    const type = element.type || model.analysisType;

    switch (type) {
      case 'beam2d': {
        // Project gravity onto local perpendicular direction for bending FEFs
        const angle = elementAngle(model, element);
        const c = Math.cos(angle);
        const s = Math.sin(angle);

        // Perpendicular component (local v direction)
        const qPerp = -qx * s + qy * c;
        // Axial component (local u direction)
        const qAxial = qx * c + qy * s;

        // Equivalent nodal loads from transverse distributed load
        const fEquiv = beam2dEquivLoads(qPerp, L, angle);

        // Add axial component
        const fAxial = (qAxial * L) / 2;
        fEquiv[0] += fAxial * c;
        fEquiv[1] += fAxial * s;
        fEquiv[3] += fAxial * c;
        fEquiv[4] += fAxial * s;

        for (let d = 0; d < dofPerNode; d++) {
          F[startDOF1 + d] += fEquiv[d];
          F[startDOF2 + d] += fEquiv[dofPerNode + d];
        }
        break;
      }

      case 'frame3d': {
        const node1 = getEntry(model.nodes, nid1);
        const node2 = getEntry(model.nodes, nid2);
        const fEquiv = frame3dEquivLoads(qx, qy, qz, L, node1, node2);

        for (let d = 0; d < dofPerNode; d++) {
          F[startDOF1 + d] += fEquiv[d];
          F[startDOF2 + d] += fEquiv[dofPerNode + d];
        }
        break;
      }

      case 'truss2d': {
        // Simple lumped mass approach: half to each node
        F[startDOF1 + 0] += (qx * L) / 2;
        F[startDOF1 + 1] += (qy * L) / 2;
        F[startDOF2 + 0] += (qx * L) / 2;
        F[startDOF2 + 1] += (qy * L) / 2;
        break;
      }

      case 'truss3d': {
        F[startDOF1 + 0] += (qx * L) / 2;
        F[startDOF1 + 1] += (qy * L) / 2;
        F[startDOF1 + 2] += (qz * L) / 2;
        F[startDOF2 + 0] += (qx * L) / 2;
        F[startDOF2 + 1] += (qy * L) / 2;
        F[startDOF2 + 2] += (qz * L) / 2;
        break;
      }

      default:
        break;
    }
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Assemble the global load vector for a given load case.
 *
 * @param {Object} model - Structural model
 * @param {{ nodeMap: Map, totalDOF: number, dofPerNode: number }} dofMap
 * @param {string|number} loadCaseId
 * @returns {Float64Array} Global force vector of length totalDOF
 */
export function assembleLoadVector(model, dofMap, loadCaseId) {
  const { nodeMap, totalDOF, dofPerNode } = dofMap;
  const F = new Float64Array(totalDOF);

  const loadCase = getEntry(model.loadCases, loadCaseId);
  if (!loadCase) {
    throw new Error(`load_vector: load case "${loadCaseId}" not found`);
  }

  const loads = loadCase.loads || [];

  for (const load of loads) {
    switch (load.type) {
      case 'nodal':
        addNodalLoad(F, load, nodeMap, model.analysisType);
        break;

      case 'distributed':
        addDistributedLoad(F, load, model, nodeMap, dofPerNode);
        break;

      case 'self_weight':
        addSelfWeight(F, load, model, nodeMap, dofPerNode);
        break;

      default:
        console.warn(
          `load_vector: unknown load type "${load.type}", skipping`
        );
        break;
    }
  }

  return F;
}
