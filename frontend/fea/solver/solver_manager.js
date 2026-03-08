/**
 * FEA solver pipeline orchestrator.
 *
 * Coordinates the full solve sequence:
 *   1. Validate model
 *   2. Build DOF map
 *   3. Assemble global stiffness matrix K
 *   4. Assemble global load vector F
 *   5. Apply boundary conditions (penalty method)
 *   6. Solve K*u = F
 *   7. Compute reactions
 *   8. Recover element internal forces
 *   9. Return results
 *
 * All functions use ES module imports; no external dependencies.
 */

import { buildDOFMap, assembleGlobalK, elementLength, elementAngle } from './assembler.js';
import { assembleLoadVector } from './load_vector.js';
import { applyBoundaryConditionsPenalty, getRestrainedDOFs } from './boundary.js';
import { solve } from './cholesky.js';
import { beam2dInternalForces } from '../elements/beam2d.js';
import { frame3dInternalForces } from '../elements/frame3d.js';
import { trussInternalForce } from '../elements/truss.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getEntry(mapOrObj, key) {
  if (mapOrObj instanceof Map) return mapOrObj.get(key);
  return mapOrObj[key];
}

function entries(mapOrObj) {
  if (mapOrObj instanceof Map) return mapOrObj.entries();
  return Object.entries(mapOrObj);
}

function mapSize(mapOrObj) {
  if (mapOrObj instanceof Map) return mapOrObj.size;
  return Object.keys(mapOrObj).length;
}

// ---------------------------------------------------------------------------
// Model validation
// ---------------------------------------------------------------------------

/**
 * Validate the structural model for common errors.
 *
 * @param {Object} model
 * @throws {Error} If the model has critical issues
 */
function validateModel(model) {
  if (!model) throw new Error('solver: model is null or undefined');
  if (!model.analysisType) throw new Error('solver: model.analysisType is not set');
  if (!model.nodes || mapSize(model.nodes) === 0) {
    throw new Error('solver: model has no nodes');
  }
  if (!model.elements || mapSize(model.elements) === 0) {
    throw new Error('solver: model has no elements');
  }
  if (!model.materials || mapSize(model.materials) === 0) {
    throw new Error('solver: model has no materials');
  }
  if (!model.sections || mapSize(model.sections) === 0) {
    throw new Error('solver: model has no sections');
  }
  if (!model.supports || mapSize(model.supports) === 0) {
    throw new Error(
      'solver: model has no supports — the system will be singular. ' +
      'Add at least enough supports to prevent rigid-body motion.'
    );
  }

  // Check that all element node references are valid
  for (const [elemId, elem] of entries(model.elements)) {
    if (!elem.nodeIds || elem.nodeIds.length < 2) {
      throw new Error(`solver: element "${elemId}" has insufficient node references`);
    }
    for (const nid of elem.nodeIds) {
      const node = getEntry(model.nodes, nid);
      if (!node) {
        throw new Error(`solver: element "${elemId}" references unknown node "${nid}"`);
      }
    }
    // Check section and material references
    if (elem.sectionId !== undefined) {
      const sec = getEntry(model.sections, elem.sectionId);
      if (!sec) {
        throw new Error(`solver: element "${elemId}" references unknown section "${elem.sectionId}"`);
      }
    }
    if (elem.materialId !== undefined) {
      const mat = getEntry(model.materials, elem.materialId);
      if (!mat) {
        throw new Error(`solver: element "${elemId}" references unknown material "${elem.materialId}"`);
      }
    }
  }

  // Check that support node references are valid
  for (const [supId, sup] of entries(model.supports)) {
    const node = getEntry(model.nodes, sup.nodeId);
    if (!node) {
      throw new Error(`solver: support "${supId}" references unknown node "${sup.nodeId}"`);
    }
  }
}

// ---------------------------------------------------------------------------
// DOF name helpers
// ---------------------------------------------------------------------------

/**
 * DOF component names for each analysis type, in order.
 */
const DOF_NAMES = {
  beam2d:  ['dx', 'dy', 'rz'],
  frame3d: ['dx', 'dy', 'dz', 'rx', 'ry', 'rz'],
  truss2d: ['dx', 'dy'],
  truss3d: ['dx', 'dy', 'dz'],
};

/**
 * Reaction force component names for each analysis type, in order.
 */
const REACTION_NAMES = {
  beam2d:  ['fx', 'fy', 'mz'],
  frame3d: ['fx', 'fy', 'fz', 'mx', 'my', 'mz'],
  truss2d: ['fx', 'fy'],
  truss3d: ['fx', 'fy', 'fz'],
};

// ---------------------------------------------------------------------------
// Internal force recovery with intermediate stations
// ---------------------------------------------------------------------------

/**
 * Recover internal forces for a beam2d element at multiple stations.
 *
 * Computes N, V, M at equally spaced stations along the element.
 * For elements with distributed loads, the intermediate values account
 * for the load distribution (linear shear, parabolic moment).
 *
 * @param {Object} model
 * @param {Object} element
 * @param {Float64Array} u_elem - Element global displacements (length 6)
 * @param {Object|null} loadCase - Load case for distributed load effects
 * @param {string} elemId
 * @param {number} numStations - Number of stations (default 11)
 * @returns {{ stations: number[], N: number[], V: number[], M: number[] }}
 */
function recoverBeam2dForces(model, element, u_elem, loadCase, elemId, numStations = 11) {
  const section = getEntry(model.sections, element.sectionId);
  const material = getEntry(model.materials, element.materialId);
  const E = material.E;
  const A = section.A;
  const I = section.Iy || section.I || 0;
  const L = elementLength(model, element);
  const angle = elementAngle(model, element);

  // Get end forces from stiffness * displacements
  const endForces = beam2dInternalForces(E, A, I, L, angle, u_elem);

  // Determine distributed load on this element (if any)
  let qPerp = 0; // perpendicular distributed load in local coords
  let qAxial = 0; // axial distributed load in local coords

  if (loadCase && loadCase.loads) {
    for (const load of loadCase.loads) {
      if (load.type === 'distributed' && String(load.elementId) === String(elemId)) {
        if (load.q !== undefined) {
          qPerp = load.q;
        } else {
          const qx = load.qx || 0;
          const qy = load.qy || 0;
          const c = Math.cos(angle);
          const s = Math.sin(angle);
          qPerp = -qx * s + qy * c;
          qAxial = qx * c + qy * s;
        }
      }
    }
  }

  // Internal forces at stations
  // At node 1 (x=0): N1, V1, M1 from endForces
  // Along the element, the distributed load modifies shear and moment linearly/parabolically:
  //   V(x) = V1 - q * x   (for uniform transverse load)
  //   M(x) = M1 + V1 * x - q * x^2 / 2
  //   N(x) = N1 - qAxial * x  (for uniform axial load)
  const N1 = endForces.N[0];
  const V1 = endForces.V[0];
  const M1 = endForces.M[0];

  const stations = new Float64Array(numStations);
  const N = new Float64Array(numStations);
  const V = new Float64Array(numStations);
  const M = new Float64Array(numStations);

  for (let s = 0; s < numStations; s++) {
    const xi = s / (numStations - 1); // 0 to 1
    const x = xi * L;

    stations[s] = xi;
    N[s] = N1 - qAxial * x;
    V[s] = V1 - qPerp * x;
    M[s] = M1 + V1 * x - (qPerp * x * x) / 2;
  }

  return { stations: Array.from(stations), N: Array.from(N), V: Array.from(V), M: Array.from(M) };
}

/**
 * Recover internal forces for a frame3d element at multiple stations.
 *
 * @param {Object} model
 * @param {Object} element
 * @param {Float64Array} u_elem - Element global displacements (length 12)
 * @param {Object|null} loadCase
 * @param {string} elemId
 * @param {number} numStations
 * @returns {{ stations: number[], N: number[], Vy: number[], Vz: number[],
 *             Mx: number[], My: number[], Mz: number[] }}
 */
function recoverFrame3dForces(model, element, u_elem, loadCase, elemId, numStations = 11) {
  const section = getEntry(model.sections, element.sectionId);
  const material = getEntry(model.materials, element.materialId);
  const E = material.E;
  const nu = material.nu !== undefined ? material.nu : 0.3;
  const G = E / (2 * (1 + nu));
  const A = section.A;
  const Iy = section.Iy || 0;
  const Iz = section.Iz || 0;
  const J = section.J || 0;
  const L = elementLength(model, element);

  const [nid1, nid2] = element.nodeIds;
  const node1 = getEntry(model.nodes, nid1);
  const node2 = getEntry(model.nodes, nid2);

  const endForces = frame3dInternalForces(E, G, A, Iy, Iz, J, L, node1, node2, u_elem);

  // Determine distributed loads on this element (global qx, qy, qz)
  let qxG = 0, qyG = 0, qzG = 0;

  if (loadCase && loadCase.loads) {
    for (const load of loadCase.loads) {
      if (load.type === 'distributed' && String(load.elementId) === String(elemId)) {
        qxG += load.qx || 0;
        qyG += load.qy || 0;
        qzG += load.qz || 0;
      }
    }
  }

  // Project global distributed load onto local axes
  // We need the transformation matrix's rotation block
  const dx = (node2.x || 0) - (node1.x || 0);
  const dy = (node2.y || 0) - (node1.y || 0);
  const dz = (node2.z || 0) - (node1.z || 0);

  // Local x-axis direction cosines
  const cx = dx / L;
  const cy = dy / L;
  const cz = dz / L;

  // Local y-axis (same logic as frame3d.js localAxes)
  const dotZ = Math.abs(cz);
  const ref = dotZ > 0.995 ? [1, 0, 0] : [0, 0, 1];
  let yx = ref[1] * cz - ref[2] * cy;
  let yy = ref[2] * cx - ref[0] * cz;
  let yz = ref[0] * cy - ref[1] * cx;
  const yLen = Math.sqrt(yx * yx + yy * yy + yz * yz);
  if (yLen > 1e-14) { yx /= yLen; yy /= yLen; yz /= yLen; }

  // Local z-axis
  const zx = cy * yz - cz * yy;
  const zy = cz * yx - cx * yz;
  const zz_l = cx * yy - cy * yx;

  // Project global distributed load onto local axes
  const qxL = qxG * cx + qyG * cy + qzG * cz;      // axial
  const qyL = qxG * yx + qyG * yy + qzG * yz;      // transverse y
  const qzL = qxG * zx + qyG * zy + qzG * zz_l;    // transverse z

  // Compute internal forces at stations
  const N1 = endForces.N[0];
  const Vy1 = endForces.Vy[0];
  const Vz1 = endForces.Vz[0];
  const Mx1 = endForces.Mx[0];
  const My1 = endForces.My[0];
  const Mz1 = endForces.Mz[0];

  const stations = new Float64Array(numStations);
  const N_arr = new Float64Array(numStations);
  const Vy_arr = new Float64Array(numStations);
  const Vz_arr = new Float64Array(numStations);
  const Mx_arr = new Float64Array(numStations);
  const My_arr = new Float64Array(numStations);
  const Mz_arr = new Float64Array(numStations);

  for (let s = 0; s < numStations; s++) {
    const xi = s / (numStations - 1);
    const x = xi * L;

    stations[s] = xi;
    N_arr[s] = N1 - qxL * x;
    Vy_arr[s] = Vy1 - qyL * x;
    Vz_arr[s] = Vz1 - qzL * x;
    Mx_arr[s] = Mx1; // Torsion is constant for uniform distributed load (no distributed torque)
    // My(x) = My1 + Vz1*x - qzL*x^2/2 (note sign convention for bending about y)
    My_arr[s] = My1 + Vz1 * x - (qzL * x * x) / 2;
    // Mz(x) = Mz1 + Vy1*x - qyL*x^2/2
    Mz_arr[s] = Mz1 + Vy1 * x - (qyL * x * x) / 2;
  }

  return {
    stations: Array.from(stations),
    N:  Array.from(N_arr),
    Vy: Array.from(Vy_arr),
    Vz: Array.from(Vz_arr),
    Mx: Array.from(Mx_arr),
    My: Array.from(My_arr),
    Mz: Array.from(Mz_arr),
  };
}

/**
 * Recover internal forces for a truss element.
 *
 * @param {Object} model
 * @param {Object} element
 * @param {Float64Array} u_elem
 * @param {number} dofPerNode
 * @returns {{ stations: number[], N: number[] }}
 */
function recoverTrussForces(model, element, u_elem, dofPerNode) {
  const section = getEntry(model.sections, element.sectionId);
  const material = getEntry(model.materials, element.materialId);
  const E = material.E;
  const A = section.A;
  const L = elementLength(model, element);

  const [nid1, nid2] = element.nodeIds;
  const node1 = getEntry(model.nodes, nid1);
  const node2 = getEntry(model.nodes, nid2);

  const dim = dofPerNode; // 2 for truss2d, 3 for truss3d
  const N = trussInternalForce(E, A, L, node1, node2, u_elem, dim);

  // Truss: constant axial force along the element
  return {
    stations: [0, 1],
    N: [N, N],
  };
}

// ---------------------------------------------------------------------------
// Extract element displacements from global vector
// ---------------------------------------------------------------------------

/**
 * Extract the displacement sub-vector for an element from the global
 * displacement vector.
 *
 * @param {Float64Array} u_global - Full global displacement vector
 * @param {Object} element
 * @param {Map} nodeMap
 * @param {number} dofPerNode
 * @returns {Float64Array}
 */
function extractElementDisplacements(u_global, element, nodeMap, dofPerNode) {
  const elemDOF = dofPerNode * 2;
  const u_elem = new Float64Array(elemDOF);

  const [nid1, nid2] = element.nodeIds;
  const start1 = nodeMap.get(nid1);
  const start2 = nodeMap.get(nid2);

  for (let d = 0; d < dofPerNode; d++) {
    u_elem[d] = u_global[start1 + d];
    u_elem[dofPerNode + d] = u_global[start2 + d];
  }

  return u_elem;
}

// ---------------------------------------------------------------------------
// Main solve function
// ---------------------------------------------------------------------------

/**
 * Run the full FEA solver pipeline.
 *
 * @param {Object} model - Structural model
 * @param {string|number} loadCaseId - Load case to solve
 * @param {Function} [progressCallback] - Optional callback: ({phase, percent, message}) => void
 * @returns {{
 *   displacements: Object,
 *   reactions: Object,
 *   elementForces: Object,
 *   maxValues: Object,
 *   solverInfo: Object
 * }}
 */
export function solveFEA(model, loadCaseId, progressCallback) {
  const progress = progressCallback || (() => {});
  const t0 = performance.now();

  // --- Phase 1: Validate ---
  progress({ phase: 'validate', percent: 0, message: 'Validating model...' });
  validateModel(model);

  // --- Phase 2: Build DOF map ---
  progress({ phase: 'dofMap', percent: 10, message: 'Building DOF map...' });
  const dofMap = buildDOFMap(model);
  const { nodeMap, totalDOF, dofPerNode } = dofMap;

  if (totalDOF === 0) {
    throw new Error('solver: model has zero degrees of freedom');
  }

  // --- Phase 3: Assemble global K ---
  progress({ phase: 'assembleK', percent: 20, message: `Assembling stiffness matrix (${totalDOF} DOF)...` });
  const K_coo = assembleGlobalK(model, dofMap);
  const K_csc_original = K_coo.toCSC();

  // --- Phase 4: Assemble global F ---
  progress({ phase: 'assembleF', percent: 40, message: 'Assembling load vector...' });
  const F_original = assembleLoadVector(model, dofMap, loadCaseId);
  const F = new Float64Array(F_original); // copy for modification

  // --- Phase 5: Apply boundary conditions ---
  progress({ phase: 'applyBC', percent: 50, message: 'Applying boundary conditions...' });
  const { K: K_bc } = applyBoundaryConditionsPenalty(K_csc_original, F, model, dofMap);

  // --- Phase 6: Solve ---
  progress({ phase: 'solve', percent: 60, message: 'Solving system of equations...' });
  const tSolveStart = performance.now();
  const u = solve(K_bc, F);
  const tSolveEnd = performance.now();

  // --- Phase 7: Compute reactions ---
  progress({ phase: 'reactions', percent: 80, message: 'Computing reactions...' });
  // R = K_original * u - F_original
  const Ku = K_csc_original.multiplyVector(u);
  const R = new Float64Array(totalDOF);
  for (let i = 0; i < totalDOF; i++) {
    R[i] = Ku[i] - F_original[i];
  }

  // --- Phase 8: Recover element internal forces ---
  progress({ phase: 'internalForces', percent: 85, message: 'Recovering element internal forces...' });
  const loadCase = getEntry(model.loadCases, loadCaseId);

  // --- Build results ---

  // 8a. Displacement results
  const dofNames = DOF_NAMES[model.analysisType];
  const reactionNames = REACTION_NAMES[model.analysisType];
  const displacements = {};

  for (const [nodeId, startDOF] of nodeMap) {
    const disp = {};
    for (let d = 0; d < dofPerNode; d++) {
      disp[dofNames[d]] = u[startDOF + d];
    }
    displacements[nodeId] = disp;
  }

  // 8b. Reaction results (only at restrained DOFs)
  const restrainedDOFs = new Set(getRestrainedDOFs(model, dofMap));
  const reactions = {};

  for (const [supId, support] of entries(model.supports)) {
    const startDOF = nodeMap.get(support.nodeId);
    const reac = {};
    let hasReaction = false;

    for (let d = 0; d < dofPerNode; d++) {
      const dofIdx = startDOF + d;
      if (restrainedDOFs.has(dofIdx)) {
        reac[reactionNames[d]] = R[dofIdx];
        hasReaction = true;
      } else {
        reac[reactionNames[d]] = 0;
      }
    }

    if (hasReaction) {
      reactions[support.nodeId] = reac;
    }
  }

  // 8c. Element internal forces
  const elementForces = {};

  for (const [elemId, element] of entries(model.elements)) {
    const u_elem = extractElementDisplacements(u, element, nodeMap, dofPerNode);
    const type = element.type || model.analysisType;

    switch (type) {
      case 'beam2d': {
        elementForces[elemId] = recoverBeam2dForces(
          model, element, u_elem, loadCase, elemId
        );
        break;
      }

      case 'frame3d': {
        elementForces[elemId] = recoverFrame3dForces(
          model, element, u_elem, loadCase, elemId
        );
        break;
      }

      case 'truss2d':
      case 'truss3d': {
        elementForces[elemId] = recoverTrussForces(
          model, element, u_elem, dofPerNode
        );
        break;
      }

      default:
        break;
    }
  }

  // --- Phase 9: Compute max values ---
  progress({ phase: 'maxValues', percent: 95, message: 'Computing max values...' });

  let maxDisplacement = { nodeId: null, value: 0, direction: null };
  let maxRotation = { nodeId: null, value: 0, direction: null };

  for (const [nodeId, disp] of Object.entries(displacements)) {
    for (const [dir, val] of Object.entries(disp)) {
      const absVal = Math.abs(val);
      if (dir.startsWith('d')) {
        // Translational DOF
        if (absVal > Math.abs(maxDisplacement.value)) {
          maxDisplacement = { nodeId, value: val, direction: dir };
        }
      } else {
        // Rotational DOF
        if (absVal > Math.abs(maxRotation.value)) {
          maxRotation = { nodeId, value: val, direction: dir };
        }
      }
    }
  }

  // Max internal forces
  let maxAxialForce = { elemId: null, value: 0 };
  let maxShearForce = { elemId: null, value: 0, direction: null };
  let maxMoment = { elemId: null, value: 0, direction: null };

  for (const [elemId, forces] of Object.entries(elementForces)) {
    // Axial force
    if (forces.N) {
      for (const val of forces.N) {
        if (Math.abs(val) > Math.abs(maxAxialForce.value)) {
          maxAxialForce = { elemId, value: val };
        }
      }
    }

    // Shear forces
    if (forces.V) {
      for (const val of forces.V) {
        if (Math.abs(val) > Math.abs(maxShearForce.value)) {
          maxShearForce = { elemId, value: val, direction: 'V' };
        }
      }
    }
    if (forces.Vy) {
      for (const val of forces.Vy) {
        if (Math.abs(val) > Math.abs(maxShearForce.value)) {
          maxShearForce = { elemId, value: val, direction: 'Vy' };
        }
      }
    }
    if (forces.Vz) {
      for (const val of forces.Vz) {
        if (Math.abs(val) > Math.abs(maxShearForce.value)) {
          maxShearForce = { elemId, value: val, direction: 'Vz' };
        }
      }
    }

    // Moments
    if (forces.M) {
      for (const val of forces.M) {
        if (Math.abs(val) > Math.abs(maxMoment.value)) {
          maxMoment = { elemId, value: val, direction: 'M' };
        }
      }
    }
    if (forces.Mz) {
      for (const val of forces.Mz) {
        if (Math.abs(val) > Math.abs(maxMoment.value)) {
          maxMoment = { elemId, value: val, direction: 'Mz' };
        }
      }
    }
    if (forces.My) {
      for (const val of forces.My) {
        if (Math.abs(val) > Math.abs(maxMoment.value)) {
          maxMoment = { elemId, value: val, direction: 'My' };
        }
      }
    }
    if (forces.Mx) {
      for (const val of forces.Mx) {
        if (Math.abs(val) > Math.abs(maxMoment.value)) {
          maxMoment = { elemId, value: val, direction: 'Mx' };
        }
      }
    }
  }

  // Max reaction
  let maxReaction = { nodeId: null, value: 0, direction: null };
  for (const [nodeId, reac] of Object.entries(reactions)) {
    for (const [dir, val] of Object.entries(reac)) {
      if (Math.abs(val) > Math.abs(maxReaction.value)) {
        maxReaction = { nodeId, value: val, direction: dir };
      }
    }
  }

  const tEnd = performance.now();

  progress({ phase: 'done', percent: 100, message: 'Solve complete.' });

  return {
    displacements,
    reactions,
    elementForces,
    maxValues: {
      maxDisplacement,
      maxRotation,
      maxAxialForce,
      maxShearForce,
      maxMoment,
      maxReaction,
    },
    solverInfo: {
      analysisType: model.analysisType,
      nodeCount: mapSize(model.nodes),
      elementCount: mapSize(model.elements),
      dofCount: totalDOF,
      restrainedDOFCount: restrainedDOFs.size,
      freeDOFCount: totalDOF - restrainedDOFs.size,
      nnzK: K_csc_original.nnz,
      solveTimeMs: Math.round(tSolveEnd - tSolveStart),
      totalTimeMs: Math.round(tEnd - t0),
      solver: totalDOF < 2000 ? 'dense_cholesky' : 'sparse_cholesky',
    },
  };
}
