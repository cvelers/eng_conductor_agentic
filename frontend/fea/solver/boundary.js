/**
 * Boundary condition application for the FEA solver.
 *
 * Supports the penalty method for applying prescribed displacements and
 * fixed supports. Since CSC matrices are immutable after creation, the
 * penalty method rebuilds the matrix through COO.
 *
 * Model structure for supports:
 *   model.supports: Map/object { id: { nodeId, conditions: { dx, dy, dz, rx, ry, rz } } }
 *   Each condition is either:
 *     - true / "fixed"  => restrained (zero prescribed displacement)
 *     - a number         => prescribed displacement value
 *     - false / undefined => free (unrestrained)
 */

import { COOMatrix } from './sparse.js';

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

/**
 * DOF component names and their offsets per analysis type.
 * Each entry maps condition name -> DOF offset within the node.
 */
const DOF_COMPONENTS = {
  beam2d:  { dx: 0, dy: 1, rz: 2 },
  frame3d: { dx: 0, dy: 1, dz: 2, rx: 3, ry: 4, rz: 5 },
  truss2d: { dx: 0, dy: 1 },
  truss3d: { dx: 0, dy: 1, dz: 2 },
};

// ---------------------------------------------------------------------------
// Restrained / Free DOF identification
// ---------------------------------------------------------------------------

/**
 * Determine the list of restrained DOF indices and their prescribed values.
 *
 * @param {Object} model
 * @param {{ nodeMap: Map, totalDOF: number, dofPerNode: number }} dofMap
 * @returns {{ dofIndex: number, prescribedValue: number }[]}
 */
function getRestrainedDOFsDetailed(model, dofMap) {
  const { nodeMap } = dofMap;
  const components = DOF_COMPONENTS[model.analysisType];

  if (!components) {
    throw new Error(
      `boundary: unsupported analysis type "${model.analysisType}"`
    );
  }

  const restrained = [];

  if (!model.supports) return restrained;

  for (const [supportId, support] of entries(model.supports)) {
    const startDOF = nodeMap.get(support.nodeId);
    if (startDOF === undefined) {
      throw new Error(
        `boundary: support node "${support.nodeId}" not found in DOF map`
      );
    }

    const conditions = support.conditions || {};

    for (const [name, offset] of Object.entries(components)) {
      const cond = conditions[name];
      if (cond === undefined || cond === false || cond === null) {
        continue; // free DOF
      }

      const dofIndex = startDOF + offset;
      let prescribedValue = 0;

      if (typeof cond === 'number') {
        prescribedValue = cond;
      }
      // cond === true or cond === "fixed" => prescribedValue = 0

      restrained.push({ dofIndex, prescribedValue });
    }
  }

  return restrained;
}

/**
 * Returns list of restrained (constrained) DOF indices.
 *
 * @param {Object} model
 * @param {{ nodeMap: Map, totalDOF: number, dofPerNode: number }} dofMap
 * @returns {number[]} Sorted array of restrained DOF indices
 */
export function getRestrainedDOFs(model, dofMap) {
  const detailed = getRestrainedDOFsDetailed(model, dofMap);
  const indices = detailed.map((d) => d.dofIndex);
  indices.sort((a, b) => a - b);

  // Remove duplicates (a node could appear in multiple supports, though unusual)
  const unique = [];
  for (let i = 0; i < indices.length; i++) {
    if (i === 0 || indices[i] !== indices[i - 1]) {
      unique.push(indices[i]);
    }
  }

  return unique;
}

/**
 * Returns list of free (unconstrained) DOF indices.
 *
 * @param {Object} model
 * @param {{ nodeMap: Map, totalDOF: number, dofPerNode: number }} dofMap
 * @returns {number[]} Sorted array of free DOF indices
 */
export function getFreeDOFs(model, dofMap) {
  const { totalDOF } = dofMap;
  const restrainedSet = new Set(getRestrainedDOFs(model, dofMap));

  const free = [];
  for (let i = 0; i < totalDOF; i++) {
    if (!restrainedSet.has(i)) {
      free.push(i);
    }
  }

  return free;
}

// ---------------------------------------------------------------------------
// Penalty method
// ---------------------------------------------------------------------------

/**
 * Apply boundary conditions using the penalty method.
 *
 * For each restrained DOF i with prescribed displacement d_i:
 *   K[i, i] += penalty
 *   F[i]    = d_i * penalty
 *
 * Since CSC format is immutable, this creates a new COO matrix from the
 * existing CSC entries, adds penalty terms, then converts back to CSC.
 *
 * The force vector F is modified in-place.
 *
 * @param {CSCMatrix} K_csc - Global stiffness matrix in CSC format
 * @param {Float64Array} F - Global force vector (modified in-place)
 * @param {Object} model
 * @param {{ nodeMap: Map, totalDOF: number, dofPerNode: number }} dofMap
 * @param {number} [penalty=1e20] - Penalty stiffness value
 * @returns {{ K: CSCMatrix, F: Float64Array }}
 */
export function applyBoundaryConditionsPenalty(K_csc, F, model, dofMap, penalty = 1e20) {
  const n = K_csc.n;
  const restrained = getRestrainedDOFsDetailed(model, dofMap);

  if (restrained.length === 0) {
    console.warn('boundary: no boundary conditions found — system may be singular');
    return { K: K_csc, F };
  }

  // Build a set of restrained DOF indices for fast lookup
  const restrainedMap = new Map();
  for (const { dofIndex, prescribedValue } of restrained) {
    // If a DOF appears multiple times (unusual), use the last prescribed value
    restrainedMap.set(dofIndex, prescribedValue);
  }

  // Create new COO from existing CSC entries
  const coo = new COOMatrix(n);

  for (let j = 0; j < n; j++) {
    const start = K_csc.colPtr[j];
    const end = K_csc.colPtr[j + 1];
    for (let p = start; p < end; p++) {
      coo.addEntry(K_csc.rowIdx[p], j, K_csc.values[p]);
    }
  }

  // Add penalty terms and modify F
  for (const [dofIndex, prescribedValue] of restrainedMap) {
    coo.addEntry(dofIndex, dofIndex, penalty);
    F[dofIndex] = prescribedValue * penalty;
  }

  // Convert back to CSC
  const K_new = coo.toCSC();

  return { K: K_new, F };
}
