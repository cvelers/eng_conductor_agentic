/**
 * Truss (bar / axial-only) element — 2D and 3D variants.
 *
 * 2D: 2 nodes, 2 DOF per node (u, v) → 4×4 K
 * 3D: 2 nodes, 3 DOF per node (u, v, w) → 6×6 K
 *
 * K_local = (EA/L) * [[1, -1], [-1, 1]]  (1D axial stiffness)
 * Transformed to global coordinates via direction cosines.
 *
 * All matrices are returned as row-major Float64Array.
 */

export const TRUSS2D_DOF_PER_NODE = 2;
export const TRUSS3D_DOF_PER_NODE = 3;

// ---------------------------------------------------------------------------
// 2D truss stiffness
// ---------------------------------------------------------------------------

/**
 * Compute the 4×4 global stiffness matrix for a 2D truss element.
 *
 * The global stiffness is assembled directly from direction cosines:
 *
 *   K = (EA/L) * [ c²   cs  -c²  -cs ]
 *                 [ cs   s²  -cs  -s² ]
 *                 [-c²  -cs   c²   cs ]
 *                 [-cs  -s²   cs   s² ]
 *
 * where c = cos(angle), s = sin(angle).
 *
 * @param {number} E     - Young's modulus
 * @param {number} A     - Cross-section area
 * @param {number} L     - Element length
 * @param {number} angle - Member angle from horizontal (radians)
 * @returns {Float64Array} row-major 4×4 global stiffness matrix
 */
export function truss2dStiffness(E, A, L, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const k = (E * A) / L;

  const cc = c * c;
  const ss = s * s;
  const cs = c * s;

  // Row-major 4x4
  return new Float64Array([
     k * cc,   k * cs,  -k * cc,  -k * cs,
     k * cs,   k * ss,  -k * cs,  -k * ss,
    -k * cc,  -k * cs,   k * cc,   k * cs,
    -k * cs,  -k * ss,   k * cs,   k * ss,
  ]);
}

// ---------------------------------------------------------------------------
// 3D truss stiffness
// ---------------------------------------------------------------------------

/**
 * Compute the 6×6 global stiffness matrix for a 3D truss element.
 *
 * Direction cosines:
 *   cx = (xJ - xI) / L
 *   cy = (yJ - yI) / L
 *   cz = (zJ - zI) / L
 *
 * The global stiffness is:
 *   K = (EA/L) * [ D  -D ]
 *                 [-D   D ]
 *
 * where D is the 3×3 outer product of the direction cosine vector:
 *   D[i][j] = c[i] * c[j]
 *
 * @param {number} E  - Young's modulus
 * @param {number} A  - Cross-section area
 * @param {number} L  - Element length
 * @param {{x:number,y:number,z:number}} nodeI - Start node
 * @param {{x:number,y:number,z:number}} nodeJ - End node
 * @returns {Float64Array} row-major 6×6 global stiffness matrix
 */
export function truss3dStiffness(E, A, L, nodeI, nodeJ) {
  const cx = (nodeJ.x - nodeI.x) / L;
  const cy = (nodeJ.y - nodeI.y) / L;
  const cz = (nodeJ.z - nodeI.z) / L;
  const k = (E * A) / L;

  const cv = [cx, cy, cz];

  // Build the 6x6 matrix
  const K = new Float64Array(36);

  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const d = k * cv[i] * cv[j];

      // Top-left block (node I, node I)
      K[i * 6 + j] = d;

      // Top-right block (node I, node J)
      K[i * 6 + (j + 3)] = -d;

      // Bottom-left block (node J, node I)
      K[(i + 3) * 6 + j] = -d;

      // Bottom-right block (node J, node J)
      K[(i + 3) * 6 + (j + 3)] = d;
    }
  }

  return K;
}

// ---------------------------------------------------------------------------
// Internal force recovery
// ---------------------------------------------------------------------------

/**
 * Compute the internal axial force for a truss element.
 *
 * The axial force is obtained by projecting the displacement difference
 * onto the member axis:
 *
 *   N = (EA/L) * (u_J - u_I) · e_x
 *
 * where e_x is the unit vector along the member and · is the dot product.
 * Positive value means tension.
 *
 * @param {number} E  - Young's modulus
 * @param {number} A  - Cross-section area
 * @param {number} L  - Element length
 * @param {{x:number,y:number,z:number}|{x:number,y:number}} nodeI - Start node
 * @param {{x:number,y:number,z:number}|{x:number,y:number}} nodeJ - End node
 * @param {Float64Array|number[]} u_global - Global displacement vector
 *   (length 4 for 2D, length 6 for 3D)
 * @param {number} dim - Spatial dimension (2 or 3)
 * @returns {number} Axial force (positive = tension)
 */
export function trussInternalForce(E, A, L, nodeI, nodeJ, u_global, dim) {
  const k = (E * A) / L;

  if (dim === 2) {
    const cx = (nodeJ.x - nodeI.x) / L;
    const cy = (nodeJ.y - nodeI.y) / L;

    // Displacement of node I in global coords
    const uI = u_global[0];
    const vI = u_global[1];
    // Displacement of node J in global coords
    const uJ = u_global[2];
    const vJ = u_global[3];

    // Elongation = (u_J - u_I) projected onto member axis
    const elongation = (uJ - uI) * cx + (vJ - vI) * cy;

    return k * elongation;
  }

  if (dim === 3) {
    const cx = (nodeJ.x - nodeI.x) / L;
    const cy = (nodeJ.y - nodeI.y) / L;
    const cz = (nodeJ.z - nodeI.z) / L;

    const uI = u_global[0];
    const vI = u_global[1];
    const wI = u_global[2];
    const uJ = u_global[3];
    const vJ = u_global[4];
    const wJ = u_global[5];

    const elongation = (uJ - uI) * cx + (vJ - vI) * cy + (wJ - wI) * cz;

    return k * elongation;
  }

  throw new Error(`trussInternalForce: unsupported dimension ${dim}, must be 2 or 3`);
}
