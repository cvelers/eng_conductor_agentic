/**
 * 3D frame (beam-column) element.
 *
 * 2 nodes, 6 DOF per node: (u, v, w, θx, θy, θz)
 *   u, v, w     – translational displacements along global X, Y, Z
 *   θx, θy, θz  – rotations about global X, Y, Z
 *
 * Local DOF ordering per node: [u, v, w, θx, θy, θz]
 * Local x-axis is along the member from node I to node J.
 *
 * All matrices are returned as row-major Float64Array.
 */

export const FRAME3D_DOF_PER_NODE = 6;
export const FRAME3D_NODES = 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Multiply two matrices stored as row-major Float64Array.
 * A is (rA x cA), B is (cA x cB), result is (rA x cB).
 */
function matMulRect(A, rA, cA, B, cB) {
  const C = new Float64Array(rA * cB);
  for (let i = 0; i < rA; i++) {
    for (let j = 0; j < cB; j++) {
      let sum = 0;
      for (let k = 0; k < cA; k++) {
        sum += A[i * cA + k] * B[k * cB + j];
      }
      C[i * cB + j] = sum;
    }
  }
  return C;
}

/**
 * Multiply two square n x n matrices.
 */
function matMul(A, B, n) {
  return matMulRect(A, n, n, B, n);
}

/**
 * Transpose a square matrix.
 */
function matTranspose(M, n) {
  const T = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      T[j * n + i] = M[i * n + j];
    }
  }
  return T;
}

/**
 * Multiply a square matrix by a vector: y = M * x.
 */
function matVecMul(M, x, n) {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += M[i * n + j] * x[j];
    }
    y[i] = sum;
  }
  return y;
}

/**
 * Euclidean norm of a 3-element array-like.
 */
function norm3(v) {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/**
 * Cross product of two 3-vectors.
 */
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Normalise a 3-vector in place and return it.
 */
function normalise(v) {
  const len = norm3(v);
  if (len < 1e-14) {
    throw new Error("frame3d: zero-length vector cannot be normalised");
  }
  v[0] /= len;
  v[1] /= len;
  v[2] /= len;
  return v;
}

// ---------------------------------------------------------------------------
// Local coordinate system
// ---------------------------------------------------------------------------

/**
 * Compute local axis directions for a 3D frame element using the
 * "vertical reference" approach.
 *
 * Local x-axis: along the member (I -> J).
 * Reference vector:
 *   - If the member is nearly vertical (|xLocal dot globalZ| > 0.995),
 *     use global X [1,0,0] as the reference.
 *   - Otherwise use global Z [0,0,1].
 * Local y-axis: ref x xLocal (then normalised)
 * Local z-axis: xLocal x yLocal
 *
 * @param {{x:number,y:number,z:number}} nodeI
 * @param {{x:number,y:number,z:number}} nodeJ
 * @returns {{ xLocal: number[], yLocal: number[], zLocal: number[] }}
 */
export function frame3dLocalAxes(nodeI, nodeJ) {
  const dx = nodeJ.x - nodeI.x;
  const dy = nodeJ.y - nodeI.y;
  const dz = nodeJ.z - nodeI.z;

  const xLocal = normalise([dx, dy, dz]);

  // Choose reference vector
  const dotZ = Math.abs(xLocal[2]); // |xLocal · globalZ|
  const ref = dotZ > 0.995 ? [1, 0, 0] : [0, 0, 1];

  // yLocal = ref x xLocal  (then normalise)
  let yLocal = cross(ref, xLocal);
  normalise(yLocal);

  // zLocal = xLocal x yLocal
  let zLocal = cross(xLocal, yLocal);
  normalise(zLocal);

  return { xLocal, yLocal, zLocal };
}

// ---------------------------------------------------------------------------
// Transformation matrix (12x12)
// ---------------------------------------------------------------------------

/**
 * Build the 12x12 transformation matrix T.
 *
 * The 3x3 rotation block R has rows [xLocal, yLocal, zLocal].
 * T is block-diagonal: diag(R, R, R, R) — four copies for the four groups
 * of 3 DOFs (translations node 1, rotations node 1, translations node 2,
 * rotations node 2).
 *
 * @param {{x:number,y:number,z:number}} nodeI
 * @param {{x:number,y:number,z:number}} nodeJ
 * @returns {Float64Array} row-major 12x12
 */
function transformationMatrix(nodeI, nodeJ) {
  const { xLocal, yLocal, zLocal } = frame3dLocalAxes(nodeI, nodeJ);

  // 3x3 rotation matrix (rows are local axes expressed in global coords)
  const R = [
    xLocal[0], xLocal[1], xLocal[2],
    yLocal[0], yLocal[1], yLocal[2],
    zLocal[0], zLocal[1], zLocal[2],
  ];

  const T = new Float64Array(144); // 12x12, initialised to 0

  // Place R in four 3x3 diagonal blocks
  for (let block = 0; block < 4; block++) {
    const offset = block * 3;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        T[(offset + i) * 12 + (offset + j)] = R[i * 3 + j];
      }
    }
  }

  return T;
}

// ---------------------------------------------------------------------------
// 12x12 local stiffness matrix
// ---------------------------------------------------------------------------

/**
 * Build the 12x12 local stiffness matrix for a 3D frame element.
 *
 * DOF order per node: [u, v, w, θx, θy, θz]
 *   u   – axial (along local x)
 *   v   – shear along local y (bending about local z, uses Iz)
 *   w   – shear along local z (bending about local y, uses Iy)
 *   θx  – torsion
 *   θy  – rotation about local y
 *   θz  – rotation about local z
 *
 * Full DOF vector: [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2]
 *                   0  1  2   3   4   5    6  7  8   9  10  11
 */
function localStiffness(E, G, A, Iy, Iz, J, L) {
  const K = new Float64Array(144);
  const L2 = L * L;
  const L3 = L * L * L;

  const eaL = (E * A) / L;
  const gjL = (G * J) / L;

  // Bending about local z (uses Iz, affects v DOFs 1,5,7,11)
  const eiz = E * Iz;
  const k_vv = (12 * eiz) / L3;
  const k_vt = (6 * eiz) / L2;
  const k_tt = (4 * eiz) / L;
  const k_tt2 = (2 * eiz) / L;

  // Bending about local y (uses Iy, affects w DOFs 2,4,8,10)
  const eiy = E * Iy;
  const k_ww = (12 * eiy) / L3;
  const k_wr = (6 * eiy) / L2;
  const k_rr = (4 * eiy) / L;
  const k_rr2 = (2 * eiy) / L;

  // Helper to set K[i][j]
  const s = (i, j, val) => { K[i * 12 + j] = val; };

  // --- Axial: DOFs 0, 6 ---
  s(0, 0, eaL);    s(0, 6, -eaL);
  s(6, 0, -eaL);   s(6, 6, eaL);

  // --- Torsion: DOFs 3, 9 ---
  s(3, 3, gjL);    s(3, 9, -gjL);
  s(9, 3, -gjL);   s(9, 9, gjL);

  // --- Bending about local z-axis (shear in v, rotation θz): DOFs 1,5,7,11 ---
  s(1, 1, k_vv);    s(1, 5, k_vt);    s(1, 7, -k_vv);   s(1, 11, k_vt);
  s(5, 1, k_vt);    s(5, 5, k_tt);    s(5, 7, -k_vt);   s(5, 11, k_tt2);
  s(7, 1, -k_vv);   s(7, 5, -k_vt);   s(7, 7, k_vv);    s(7, 11, -k_vt);
  s(11, 1, k_vt);   s(11, 5, k_tt2);  s(11, 7, -k_vt);  s(11, 11, k_tt);

  // --- Bending about local y-axis (shear in w, rotation θy): DOFs 2,4,8,10 ---
  // Note: the coupling between w and θy follows the opposite sign convention
  // compared to v and θz because of the right-hand rule.
  s(2, 2, k_ww);    s(2, 4, -k_wr);   s(2, 8, -k_ww);   s(2, 10, -k_wr);
  s(4, 2, -k_wr);   s(4, 4, k_rr);    s(4, 8, k_wr);    s(4, 10, k_rr2);
  s(8, 2, -k_ww);   s(8, 4, k_wr);    s(8, 8, k_ww);    s(8, 10, k_wr);
  s(10, 2, -k_wr);  s(10, 4, k_rr2);  s(10, 8, k_wr);   s(10, 10, k_rr);

  return K;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Compute the 12x12 global stiffness matrix for a 3D frame element.
 *
 * @param {number} E   - Young's modulus
 * @param {number} G   - Shear modulus
 * @param {number} A   - Cross-section area
 * @param {number} Iy  - Second moment of area about local y-axis
 * @param {number} Iz  - Second moment of area about local z-axis
 * @param {number} J   - Torsion constant
 * @param {number} L   - Element length
 * @param {{x:number,y:number,z:number}} nodeI - Start node coordinates
 * @param {{x:number,y:number,z:number}} nodeJ - End node coordinates
 * @returns {Float64Array} row-major 12x12 global stiffness matrix
 */
export function frame3dStiffness(E, G, A, Iy, Iz, J, L, nodeI, nodeJ) {
  const Kl = localStiffness(E, G, A, Iy, Iz, J, L);
  const T = transformationMatrix(nodeI, nodeJ);
  const Tt = matTranspose(T, 12);

  // K_global = Tᵀ · K_local · T
  const KlT = matMul(Kl, T, 12);
  return matMul(Tt, KlT, 12);
}

/**
 * Compute equivalent nodal forces for uniform distributed loads applied in
 * global directions (qx, qy, qz) on a 3D frame element.
 *
 * The loads are first projected onto local axes, then fixed-end forces are
 * computed for each local component:
 *   - Local x (axial): f = [qx_L * L/2, ..., qx_L * L/2, ...]
 *   - Local y (bending about z): uses standard beam FE forces
 *   - Local z (bending about y): uses standard beam FE forces
 *
 * The local force vector is then transformed to global coords: f_global = Tᵀ · f_local.
 *
 * @param {number} qx - Distributed load in global X (force/length)
 * @param {number} qy - Distributed load in global Y (force/length)
 * @param {number} qz - Distributed load in global Z (force/length)
 * @param {number} L  - Element length
 * @param {{x:number,y:number,z:number}} nodeI
 * @param {{x:number,y:number,z:number}} nodeJ
 * @returns {Float64Array} length-12 global equivalent nodal force vector
 */
export function frame3dEquivLoads(qx, qy, qz, L, nodeI, nodeJ) {
  const T = transformationMatrix(nodeI, nodeJ);

  // Project global load vector onto local axes using rotation matrix (first 3x3 block of T)
  // q_local = R · q_global  where R is the 3x3 block
  const R = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      R[i * 3 + j] = T[i * 12 + j];
    }
  }

  const qGlobal = [qx, qy, qz];
  const qLocal = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      qLocal[i] += R[i * 3 + j] * qGlobal[j];
    }
  }

  const qxL = qLocal[0]; // axial
  const qyL = qLocal[1]; // transverse in local y
  const qzL = qLocal[2]; // transverse in local z

  const L2 = L * L;

  // Build local fixed-end force vector
  // DOF: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
  const fLocal = new Float64Array(12);

  // Axial (local x distributed load)
  fLocal[0] = (qxL * L) / 2;   // u1
  fLocal[6] = (qxL * L) / 2;   // u2

  // Bending about local z (transverse load in local y, affects v and θz)
  fLocal[1] = (qyL * L) / 2;          // v1
  fLocal[5] = (qyL * L2) / 12;        // θz1
  fLocal[7] = (qyL * L) / 2;          // v2
  fLocal[11] = -(qyL * L2) / 12;      // θz2

  // Bending about local y (transverse load in local z, affects w and θy)
  fLocal[2] = (qzL * L) / 2;          // w1
  fLocal[4] = -(qzL * L2) / 12;       // θy1
  fLocal[8] = (qzL * L) / 2;          // w2
  fLocal[10] = (qzL * L2) / 12;       // θy2

  // Transform to global: f_global = Tᵀ · f_local
  const Tt = matTranspose(T, 12);
  return matVecMul(Tt, fLocal, 12);
}

/**
 * Recover internal forces at the two element nodes of a 3D frame element.
 *
 * Steps:
 *   1. Transform global displacements to local: u_local = T · u_global
 *   2. Local forces: f_local = K_local · u_local
 *   3. Extract internal forces at each end.
 *
 * @param {number} E   - Young's modulus
 * @param {number} G   - Shear modulus
 * @param {number} A   - Cross-section area
 * @param {number} Iy  - Second moment of area about local y-axis
 * @param {number} Iz  - Second moment of area about local z-axis
 * @param {number} J   - Torsion constant
 * @param {number} L   - Element length
 * @param {{x:number,y:number,z:number}} nodeI
 * @param {{x:number,y:number,z:number}} nodeJ
 * @param {Float64Array|number[]} u_global - length-12 global displacement vector
 * @returns {{ N: number[], Vy: number[], Vz: number[], Mx: number[], My: number[], Mz: number[] }}
 *   Each array has two entries [node1, node2].
 */
export function frame3dInternalForces(E, G, A, Iy, Iz, J, L, nodeI, nodeJ, u_global) {
  const T = transformationMatrix(nodeI, nodeJ);
  const Kl = localStiffness(E, G, A, Iy, Iz, J, L);

  // u_local = T · u_global
  const uLocal = matVecMul(T, new Float64Array(u_global), 12);

  // f_local = K_local · u_local
  const fLocal = matVecMul(Kl, uLocal, 12);

  // DOF: [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
  //        0   1   2    3    4    5    6   7   8    9   10   11
  //
  // Internal forces: at node 1 the element reaction is negative of the nodal
  // force; at node 2 it equals the nodal force.
  return {
    N:  [-fLocal[0], fLocal[6]],
    Vy: [-fLocal[1], fLocal[7]],
    Vz: [-fLocal[2], fLocal[8]],
    Mx: [-fLocal[3], fLocal[9]],
    My: [-fLocal[4], fLocal[10]],
    Mz: [-fLocal[5], fLocal[11]],
  };
}
