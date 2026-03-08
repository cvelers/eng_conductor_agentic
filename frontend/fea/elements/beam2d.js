/**
 * 2D Euler-Bernoulli beam element.
 *
 * 2 nodes, 3 DOF per node: (u, v, theta)
 *   u     – axial displacement along member axis
 *   v     – transverse displacement perpendicular to member
 *   theta – rotation about the out-of-plane axis
 *
 * Local DOF ordering: [u1, v1, theta1, u2, v2, theta2]
 *
 * All matrices are returned as row-major Float64Array.
 */

export const BEAM2D_DOF_PER_NODE = 3;
export const BEAM2D_NODES = 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Multiply two square matrices stored as row-major Float64Array.
 * @param {Float64Array} A - left matrix (n x n)
 * @param {Float64Array} B - right matrix (n x n)
 * @param {number} n       - dimension
 * @returns {Float64Array}  A * B
 */
function matMul(A, B, n) {
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }
  return C;
}

/**
 * Transpose a square matrix (row-major Float64Array).
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

// ---------------------------------------------------------------------------
// Transformation matrix (6x6)
// ---------------------------------------------------------------------------

/**
 * Build the 6x6 transformation matrix T that maps global DOFs to local DOFs.
 *
 * The 2x2 rotation block R is:
 *   [ c  s ]
 *   [-s  c ]
 *
 * T is block-diagonal with blocks [R, 1, R, 1] where 1 stands for a pass-through
 * for the rotational DOF (theta is unchanged).
 *
 * @param {number} angle - member angle from horizontal (radians)
 * @returns {Float64Array} row-major 6x6
 */
function transformationMatrix(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const T = new Float64Array(36); // initialised to 0

  // Node 1 block: rows 0-2, cols 0-2
  T[0 * 6 + 0] = c;
  T[0 * 6 + 1] = s;
  T[1 * 6 + 0] = -s;
  T[1 * 6 + 1] = c;
  T[2 * 6 + 2] = 1;

  // Node 2 block: rows 3-5, cols 3-5
  T[3 * 6 + 3] = c;
  T[3 * 6 + 4] = s;
  T[4 * 6 + 3] = -s;
  T[4 * 6 + 4] = c;
  T[5 * 6 + 5] = 1;

  return T;
}

// ---------------------------------------------------------------------------
// Local stiffness matrix (6x6)
// ---------------------------------------------------------------------------

/**
 * Build the 6x6 local stiffness matrix combining axial bar and Euler-Bernoulli
 * bending stiffness.
 *
 * DOF order: [u1, v1, theta1, u2, v2, theta2]
 *
 * Axial terms (EA/L) at DOFs 0 and 3:
 *   K[0,0] = EA/L    K[0,3] = -EA/L
 *   K[3,0] = -EA/L   K[3,3] =  EA/L
 *
 * Bending terms (EI/L^3) at DOFs 1,2,4,5 (Hermitian shape functions):
 *   K[1,1] =  12EI/L^3   K[1,2] =  6EI/L^2   K[1,4] = -12EI/L^3   K[1,5] =  6EI/L^2
 *   K[2,1] =  6EI/L^2    K[2,2] =  4EI/L      K[2,4] = -6EI/L^2    K[2,5] =  2EI/L
 *   K[4,1] = -12EI/L^3   K[4,2] = -6EI/L^2    K[4,4] =  12EI/L^3   K[4,5] = -6EI/L^2
 *   K[5,1] =  6EI/L^2    K[5,2] =  2EI/L      K[5,4] = -6EI/L^2    K[5,5] =  4EI/L
 */
function localStiffness(E, A, I, L) {
  const K = new Float64Array(36);
  const eaL = (E * A) / L;
  const eiL3 = (E * I) / (L * L * L);
  const eiL2 = (E * I) / (L * L);
  const eiL = (E * I) / L;

  // Axial
  K[0 * 6 + 0] = eaL;
  K[0 * 6 + 3] = -eaL;
  K[3 * 6 + 0] = -eaL;
  K[3 * 6 + 3] = eaL;

  // Bending
  K[1 * 6 + 1] = 12 * eiL3;
  K[1 * 6 + 2] = 6 * eiL2;
  K[1 * 6 + 4] = -12 * eiL3;
  K[1 * 6 + 5] = 6 * eiL2;

  K[2 * 6 + 1] = 6 * eiL2;
  K[2 * 6 + 2] = 4 * eiL;
  K[2 * 6 + 4] = -6 * eiL2;
  K[2 * 6 + 5] = 2 * eiL;

  K[4 * 6 + 1] = -12 * eiL3;
  K[4 * 6 + 2] = -6 * eiL2;
  K[4 * 6 + 4] = 12 * eiL3;
  K[4 * 6 + 5] = -6 * eiL2;

  K[5 * 6 + 1] = 6 * eiL2;
  K[5 * 6 + 2] = 2 * eiL;
  K[5 * 6 + 4] = -6 * eiL2;
  K[5 * 6 + 5] = 4 * eiL;

  return K;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Compute the 6x6 global stiffness matrix for a 2D beam element.
 *
 * @param {number} E     - Young's modulus
 * @param {number} A     - Cross-section area
 * @param {number} I     - Second moment of area (about out-of-plane axis)
 * @param {number} L     - Element length
 * @param {number} angle - Member angle from horizontal (radians)
 * @returns {Float64Array} row-major 6x6 global stiffness matrix
 */
export function beam2dStiffness(E, A, I, L, angle) {
  const Kl = localStiffness(E, A, I, L);
  const T = transformationMatrix(angle);
  const Tt = matTranspose(T, 6);

  // K_global = Tᵀ · K_local · T
  const KlT = matMul(Kl, T, 6);
  return matMul(Tt, KlT, 6);
}

/**
 * Compute equivalent nodal forces for a uniform distributed load acting
 * perpendicular to the member in local coordinates.
 *
 * Fixed-end forces in local coords:
 *   f_local = [0, qL/2, qL²/12, 0, qL/2, -qL²/12]
 *
 * Returned in global coordinates: f_global = Tᵀ · f_local.
 *
 * @param {number} q     - Uniform distributed load intensity (force/length,
 *                          perpendicular to member, positive in local +v direction)
 * @param {number} L     - Element length
 * @param {number} angle - Member angle from horizontal (radians)
 * @returns {Float64Array} length-6 global equivalent nodal force vector
 */
export function beam2dEquivLoads(q, L, angle) {
  const fLocal = new Float64Array(6);
  fLocal[0] = 0;
  fLocal[1] = (q * L) / 2;
  fLocal[2] = (q * L * L) / 12;
  fLocal[3] = 0;
  fLocal[4] = (q * L) / 2;
  fLocal[5] = -(q * L * L) / 12;

  const T = transformationMatrix(angle);
  const Tt = matTranspose(T, 6);
  return matVecMul(Tt, fLocal, 6);
}

/**
 * Recover internal forces at the two element nodes.
 *
 * Steps:
 *   1. Transform global displacements to local: u_local = T · u_global
 *   2. Local forces: f_local = K_local · u_local
 *   3. Extract axial force N, shear V, and moment M at each node.
 *
 * Sign conventions (local coords):
 *   - N positive = tension
 *   - V and M follow beam sign convention (positive per Euler-Bernoulli)
 *
 * @param {number} E        - Young's modulus
 * @param {number} A        - Cross-section area
 * @param {number} I        - Second moment of area
 * @param {number} L        - Element length
 * @param {number} angle    - Member angle from horizontal (radians)
 * @param {Float64Array|number[]} u_global - length-6 global displacement vector
 * @returns {{ N: number[], V: number[], M: number[] }}
 */
export function beam2dInternalForces(E, A, I, L, angle, u_global) {
  const T = transformationMatrix(angle);
  const Kl = localStiffness(E, A, I, L);

  // u_local = T · u_global
  const uLocal = matVecMul(T, new Float64Array(u_global), 6);

  // f_local = K_local · u_local
  const fLocal = matVecMul(Kl, uLocal, 6);

  // Internal forces at node 1 are opposite to the applied nodal forces.
  // At node 1: N1 = -f[0], V1 = -f[1], M1 = -f[2]  (equilibrium of left cut)
  // At node 2: N2 =  f[3], V2 =  f[4], M2 =  f[5]
  //
  // We return the element-end forces in the conventional sense:
  //   N positive = tension => N1 = -fLocal[0], N2 = fLocal[3]
  return {
    N: [-fLocal[0], fLocal[3]],
    V: [-fLocal[1], fLocal[4]],
    M: [-fLocal[2], fLocal[5]],
  };
}
