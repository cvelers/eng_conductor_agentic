/**
 * Global stiffness matrix assembler.
 *
 * Builds the DOF map from the structural model, then assembles the global
 * stiffness matrix in COO format by scattering element contributions.
 *
 * Supported analysis types: beam2d, frame3d, truss2d, truss3d.
 *
 * Model structure:
 *   model.analysisType: "beam2d" | "frame3d" | "truss2d" | "truss3d"
 *   model.nodes:     Map/object { id: { x, y, z } }
 *   model.elements:  Map/object { id: { type, nodeIds: [id1, id2], sectionId, materialId } }
 *   model.sections:  Map/object { id: { A, Iy, Iz, J, ... } }
 *   model.materials: Map/object { id: { E, nu, rho, ... } }
 */

import { COOMatrix } from './sparse.js';
import { beam2dStiffness, BEAM2D_DOF_PER_NODE } from '../elements/beam2d.js';
import { frame3dStiffness, FRAME3D_DOF_PER_NODE } from '../elements/frame3d.js';
import { truss2dStiffness, TRUSS2D_DOF_PER_NODE, truss3dStiffness, TRUSS3D_DOF_PER_NODE } from '../elements/truss.js';

// ---------------------------------------------------------------------------
// Helpers for accessing model data (supports both Map and plain objects)
// ---------------------------------------------------------------------------

/**
 * Get a value from a Map or plain object by key.
 * @param {Map|Object} mapOrObj
 * @param {string|number} key
 * @returns {*}
 */
function getEntry(mapOrObj, key) {
  if (mapOrObj instanceof Map) return mapOrObj.get(key);
  return mapOrObj[key];
}

/**
 * Iterate entries of a Map or plain object.
 * @param {Map|Object} mapOrObj
 * @returns {Iterable<[string|number, *]>}
 */
function entries(mapOrObj) {
  if (mapOrObj instanceof Map) return mapOrObj.entries();
  return Object.entries(mapOrObj);
}

// ---------------------------------------------------------------------------
// DOF Map
// ---------------------------------------------------------------------------

/**
 * Determine the number of degrees of freedom per node for a given analysis type.
 *
 * @param {string} analysisType
 * @returns {number}
 */
function dofPerNodeForType(analysisType) {
  switch (analysisType) {
    case 'beam2d':  return BEAM2D_DOF_PER_NODE;   // 3
    case 'frame3d': return FRAME3D_DOF_PER_NODE;   // 6
    case 'truss2d': return TRUSS2D_DOF_PER_NODE;   // 2
    case 'truss3d': return TRUSS3D_DOF_PER_NODE;   // 3
    default:
      throw new Error(`assembler: unsupported analysis type "${analysisType}"`);
  }
}

/**
 * Build the DOF numbering map from the model.
 *
 * Each node is assigned a contiguous block of DOFs starting from 0.
 * The order follows the insertion order of model.nodes.
 *
 * @param {Object} model
 * @returns {{
 *   nodeMap: Map<string|number, number>,
 *   totalDOF: number,
 *   dofPerNode: number
 * }}
 *   nodeMap: maps nodeId -> first DOF index for that node.
 *   totalDOF: total number of DOFs in the system.
 *   dofPerNode: DOFs per node for this analysis type.
 */
export function buildDOFMap(model) {
  const dofPerNode = dofPerNodeForType(model.analysisType);
  const nodeMap = new Map();
  let nextDOF = 0;

  for (const [nodeId] of entries(model.nodes)) {
    nodeMap.set(nodeId, nextDOF);
    nextDOF += dofPerNode;
  }

  return {
    nodeMap,
    totalDOF: nextDOF,
    dofPerNode,
  };
}

// ---------------------------------------------------------------------------
// Element geometry helpers
// ---------------------------------------------------------------------------

/**
 * Compute the length of an element.
 *
 * @param {Object} model
 * @param {Object} element - Element object with nodeIds: [id1, id2]
 * @returns {number} Element length
 */
export function elementLength(model, element) {
  const n1 = getEntry(model.nodes, element.nodeIds[0]);
  const n2 = getEntry(model.nodes, element.nodeIds[1]);

  const dx = (n2.x || 0) - (n1.x || 0);
  const dy = (n2.y || 0) - (n1.y || 0);
  const dz = (n2.z || 0) - (n1.z || 0);

  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Compute the angle of a 2D element from the horizontal (radians).
 * Measured from the positive x-axis, counter-clockwise positive.
 *
 * @param {Object} model
 * @param {Object} element
 * @returns {number} Angle in radians
 */
export function elementAngle(model, element) {
  const n1 = getEntry(model.nodes, element.nodeIds[0]);
  const n2 = getEntry(model.nodes, element.nodeIds[1]);

  const dx = (n2.x || 0) - (n1.x || 0);
  const dy = (n2.y || 0) - (n1.y || 0);

  return Math.atan2(dy, dx);
}

// ---------------------------------------------------------------------------
// Element stiffness dispatch
// ---------------------------------------------------------------------------

/**
 * Compute the global stiffness matrix for a single element and return it
 * along with the global DOF indices for scatter.
 *
 * @param {Object} model
 * @param {Object} element
 * @param {Map<string|number, number>} nodeMap - DOF map
 * @param {number} dofPerNode
 * @returns {{ Ke: Float64Array, dofs: number[] }}
 *   Ke: row-major element global stiffness matrix (nDOF_elem x nDOF_elem)
 *   dofs: global DOF indices corresponding to rows/cols of Ke
 */
function elementStiffnessAndDOFs(model, element, nodeMap, dofPerNode) {
  const [nid1, nid2] = element.nodeIds;
  const node1 = getEntry(model.nodes, nid1);
  const node2 = getEntry(model.nodes, nid2);
  const section = getEntry(model.sections, element.sectionId);
  const material = getEntry(model.materials, element.materialId);

  const E = material.E;
  const nu = material.nu !== undefined ? material.nu : 0.3;
  const A = section.A;
  const L = elementLength(model, element);

  if (L < 1e-14) {
    throw new Error(`assembler: element has zero length (nodes ${nid1}, ${nid2})`);
  }

  // Build the global DOF index array for this element
  const startDOF1 = nodeMap.get(nid1);
  const startDOF2 = nodeMap.get(nid2);

  if (startDOF1 === undefined || startDOF2 === undefined) {
    throw new Error(
      `assembler: node not found in DOF map (node ${nid1} or ${nid2})`
    );
  }

  const dofs = [];
  for (let d = 0; d < dofPerNode; d++) dofs.push(startDOF1 + d);
  for (let d = 0; d < dofPerNode; d++) dofs.push(startDOF2 + d);

  let Ke;
  const type = element.type || model.analysisType;

  switch (type) {
    case 'beam2d': {
      const I = section.Iy || section.I || 0;
      const angle = elementAngle(model, element);
      Ke = beam2dStiffness(E, A, I, L, angle);
      break;
    }

    case 'frame3d': {
      const G = E / (2 * (1 + nu));
      const Iy = section.Iy || 0;
      const Iz = section.Iz || 0;
      const J = section.J || 0;
      Ke = frame3dStiffness(E, G, A, Iy, Iz, J, L, node1, node2);
      break;
    }

    case 'truss2d': {
      const angle = elementAngle(model, element);
      Ke = truss2dStiffness(E, A, L, angle);
      break;
    }

    case 'truss3d': {
      Ke = truss3dStiffness(E, A, L, node1, node2);
      break;
    }

    default:
      throw new Error(`assembler: unsupported element type "${type}"`);
  }

  return { Ke, dofs };
}

// ---------------------------------------------------------------------------
// Global assembly
// ---------------------------------------------------------------------------

/**
 * Assemble the global stiffness matrix in COO format.
 *
 * For each element:
 *   1. Compute element global stiffness matrix Ke.
 *   2. Determine global DOF indices.
 *   3. Scatter Ke entries into the global COO matrix.
 *
 * @param {Object} model - Structural model
 * @param {{ nodeMap: Map, totalDOF: number, dofPerNode: number }} dofMap
 * @returns {COOMatrix} Global stiffness matrix in COO format
 */
export function assembleGlobalK(model, dofMap) {
  const { nodeMap, totalDOF, dofPerNode } = dofMap;
  const K = new COOMatrix(totalDOF);
  const elemDOF = dofPerNode * 2; // 2-node elements

  for (const [elemId, element] of entries(model.elements)) {
    const { Ke, dofs } = elementStiffnessAndDOFs(
      model, element, nodeMap, dofPerNode
    );

    // Scatter element stiffness into global matrix
    for (let i = 0; i < elemDOF; i++) {
      for (let j = 0; j < elemDOF; j++) {
        const val = Ke[i * elemDOF + j];
        if (val !== 0) {
          K.addEntry(dofs[i], dofs[j], val);
        }
      }
    }
  }

  return K;
}
