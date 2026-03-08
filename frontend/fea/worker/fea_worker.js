/**
 * FEA Web Worker — runs the solver pipeline off the main thread.
 *
 * Communication protocol:
 *   IN:  { type: "SOLVE", payload: { model, loadCaseId } }
 *   IN:  { type: "CANCEL" }
 *   OUT: { type: "PROGRESS", payload: { phase, percent, message } }
 *   OUT: { type: "RESULT", payload: { displacements, reactions, elementForces, maxValues, solverInfo } }
 *   OUT: { type: "ERROR", payload: { message, phase } }
 */

// Import solver modules (relative to worker location)
// Note: Web Workers use importScripts or dynamic import for ES modules
// We inline the solver logic here to avoid import complexity in workers

/* global self */

self.onmessage = async function (e) {
  const { type, payload } = e.data;

  if (type === "CANCEL") {
    // Signal cancellation (handled via flag)
    self._cancelled = true;
    return;
  }

  if (type === "SOLVE") {
    self._cancelled = false;
    try {
      const result = await solveFEA(payload.model, payload.loadCaseId);
      if (!self._cancelled) {
        self.postMessage({ type: "RESULT", payload: result });
      }
    } catch (err) {
      self.postMessage({ type: "ERROR", payload: { message: err.message, phase: err.phase || "unknown" } });
    }
  }
};

function progress(phase, percent, message) {
  self.postMessage({ type: "PROGRESS", payload: { phase, percent, message } });
}

// ══════════════════════════════════════════════════════════════════
// Inline solver implementation (avoids ES module import issues in workers)
// ══════════════════════════════════════════════════════════════════

async function solveFEA(modelData, loadCaseId) {
  const t0 = performance.now();
  progress("init", 0, "Initializing solver...");

  const model = modelData;
  const nodes = model.nodes;
  const elements = model.elements;
  const materials = model.materials;
  const sections = model.sections;
  const restraints = model.restraints;
  const loadCases = model.loadCases;
  const analysisType = model.analysisType || "frame3d";

  // ── Step 1: Determine DOF per node ─────────────────────────────
  let dofPerNode;
  if (analysisType === "beam2d") dofPerNode = 3;
  else if (analysisType === "truss2d") dofPerNode = 2;
  else if (analysisType === "truss3d") dofPerNode = 3;
  else dofPerNode = 6; // frame3d

  // ── Step 2: Build DOF map ──────────────────────────────────────
  progress("assembly", 5, "Building DOF map...");
  const nodeIds = Object.keys(nodes);
  const nodeMap = {};
  let totalDOF = 0;
  for (const nid of nodeIds) {
    nodeMap[nid] = totalDOF;
    totalDOF += dofPerNode;
  }

  // ── Step 3: Assemble global K (dense for simplicity up to ~3000 DOF) ──
  progress("assembly", 10, `Assembling stiffness matrix (${totalDOF} DOF)...`);
  const K = new Float64Array(totalDOF * totalDOF); // dense
  const F = new Float64Array(totalDOF);

  const elemIds = Object.keys(elements);
  for (let ei = 0; ei < elemIds.length; ei++) {
    if (self._cancelled) throw Object.assign(new Error("Cancelled"), { phase: "assembly" });
    const elemId = elemIds[ei];
    const elem = elements[elemId];
    const nids = elem.nodeIds;
    if (nids.length < 2) continue;

    const n1 = nodes[nids[0]];
    const n2 = nodes[nids[1]];
    if (!n1 || !n2) continue;

    const sec = sections[elem.sectionId] || {};
    const mat = materials[elem.materialId] || { E: 210000, nu: 0.3 };
    const E = mat.E || 210000;
    const nu = mat.nu || 0.3;
    const G = E / (2 * (1 + nu));
    const A = sec.A || 1000;
    const Iy = sec.Iy || 1e6;
    const Iz = sec.Iz || 1e5;
    const J = sec.It || sec.J || 1e4;

    const dx = n2.x - n1.x;
    const dy = n2.y - n1.y;
    const dz = n2.z - n1.z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-6) continue;

    let Ke;
    if (analysisType === "beam2d") {
      const angle = Math.atan2(dy, dx);
      Ke = beam2dStiffnessMatrix(E, A, Iy, L, angle);
    } else if (analysisType === "truss2d") {
      const angle = Math.atan2(dy, dx);
      Ke = truss2dStiffnessMatrix(E, A, L, angle);
    } else if (analysisType === "truss3d") {
      Ke = truss3dStiffnessMatrix(E, A, L, n1, n2);
    } else {
      Ke = frame3dStiffnessMatrix(E, G, A, Iy, Iz, J, L, n1, n2);
    }

    // Scatter into global K
    const dofs = [];
    for (const nid of nids) {
      const start = nodeMap[nid];
      for (let d = 0; d < dofPerNode; d++) dofs.push(start + d);
    }

    const nDof = dofs.length;
    for (let i = 0; i < nDof; i++) {
      for (let j = 0; j < nDof; j++) {
        K[dofs[i] * totalDOF + dofs[j]] += Ke[i * nDof + j];
      }
    }

    if (ei % 50 === 0) {
      progress("assembly", 10 + (ei / elemIds.length) * 30, `Assembled ${ei + 1}/${elemIds.length} elements`);
    }
  }

  // ── Step 4: Assemble load vector ───────────────────────────────
  progress("assembly", 40, "Assembling load vector...");
  const lc = loadCases[loadCaseId] || loadCases[Object.keys(loadCases)[0]];
  if (lc) {
    for (const load of lc.loads) {
      if (load.type === "nodal") {
        const nid = load.node_id || load.nodeId;
        const start = nodeMap[nid];
        if (start === undefined) continue;
        if (dofPerNode >= 1 && load.fx) F[start] += load.fx;
        if (dofPerNode >= 2 && load.fy) F[start + 1] += load.fy;
        if (dofPerNode >= 3 && (analysisType !== "beam2d")) {
          if (load.fz) F[start + 2] += load.fz;
        }
        if (dofPerNode === 3 && analysisType === "beam2d") {
          if (load.mz) F[start + 2] += load.mz;
        }
        if (dofPerNode === 6) {
          if (load.mx) F[start + 3] += load.mx;
          if (load.my) F[start + 4] += load.my;
          if (load.mz) F[start + 5] += load.mz;
        }
      }
      if (load.type === "distributed") {
        const elemId = load.element_id || load.elementId;
        const elem = elements[elemId];
        if (!elem) continue;
        const n1 = nodes[elem.nodeIds[0]];
        const n2 = nodes[elem.nodeIds[1]];
        if (!n1 || !n2) continue;
        const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (L < 1e-6) continue;

        const qx = load.qx || 0, qy = load.qy || 0, qz = load.qz || 0;

        if (analysisType === "beam2d") {
          const angle = Math.atan2(dy, dx);
          const feq = beam2dEquivLoadVector(qy, L, angle);
          const dofs = [];
          for (const nid of elem.nodeIds) {
            const s = nodeMap[nid];
            for (let d = 0; d < dofPerNode; d++) dofs.push(s + d);
          }
          for (let i = 0; i < feq.length; i++) F[dofs[i]] += feq[i];
        } else if (analysisType === "frame3d") {
          const feq = frame3dEquivLoadVector(qx, qy, qz, L, n1, n2);
          const dofs = [];
          for (const nid of elem.nodeIds) {
            const s = nodeMap[nid];
            for (let d = 0; d < dofPerNode; d++) dofs.push(s + d);
          }
          for (let i = 0; i < feq.length; i++) F[dofs[i]] += feq[i];
        } else {
          // Truss: no transverse loads; only axial
          // Distribute axial component equally
          const cx = dx / L, cy = dy / L, cz = dz / L;
          const qAxial = qx * cx + qy * cy + qz * cz;
          const fAxial = qAxial * L / 2;
          for (const nid of elem.nodeIds) {
            const s = nodeMap[nid];
            if (dofPerNode >= 1) F[s] += fAxial * cx;
            if (dofPerNode >= 2) F[s + 1] += fAxial * cy;
            if (dofPerNode >= 3) F[s + 2] += fAxial * cz;
          }
        }
      }
    }
  }

  // ── Step 5: Apply boundary conditions (penalty method) ─────────
  progress("boundary", 45, "Applying boundary conditions...");
  const PENALTY = 1e20;
  for (const [nid, rest] of Object.entries(restraints)) {
    const start = nodeMap[nid];
    if (start === undefined) continue;
    const dofFlags = analysisType === "beam2d"
      ? [rest.dx, rest.dy, rest.rz]
      : analysisType === "truss2d"
        ? [rest.dx, rest.dy]
        : analysisType === "truss3d"
          ? [rest.dx, rest.dy, rest.dz]
          : [rest.dx, rest.dy, rest.dz, rest.rx, rest.ry, rest.rz];

    for (let d = 0; d < dofFlags.length; d++) {
      if (dofFlags[d]) {
        const idx = start + d;
        K[idx * totalDOF + idx] += PENALTY;
        F[idx] = 0; // prescribed displacement = 0
      }
    }
  }

  // ── Step 5b: Pre-solve checks ────────────────────────────────
  progress("solve", 48, "Checking structural stability...");

  // Check connectivity: all nodes must be reachable from restrained nodes via elements
  const restrainedNodes = new Set(Object.keys(restraints).filter(nid => nodeMap[nid] !== undefined));
  if (restrainedNodes.size === 0) {
    throw Object.assign(new Error("No boundary conditions applied. Add supports (restraints) to the model."), { phase: "boundary" });
  }
  const adjacency = {};
  for (const nid of nodeIds) adjacency[nid] = new Set();
  for (const [, elem] of Object.entries(elements)) {
    const nids = elem.nodeIds;
    if (nids && nids.length >= 2) {
      adjacency[nids[0]]?.add(nids[1]);
      adjacency[nids[1]]?.add(nids[0]);
    }
  }
  const visited = new Set();
  const queue = [...restrainedNodes];
  while (queue.length > 0) {
    const n = queue.pop();
    if (visited.has(n)) continue;
    visited.add(n);
    for (const nb of (adjacency[n] || [])) {
      if (!visited.has(nb)) queue.push(nb);
    }
  }
  const disconnected = nodeIds.filter(nid => !visited.has(nid));
  if (disconnected.length > 0) {
    throw Object.assign(
      new Error(`Disconnected nodes: ${disconnected.join(", ")}. These nodes have no path to any support. Check element connectivity.`),
      { phase: "boundary" },
    );
  }

  // Check for zero-stiffness DOFs (excluding penalized DOFs)
  const dofLabels = analysisType === "beam2d" ? ["dx", "dy", "rz"]
    : analysisType === "truss2d" ? ["dx", "dy"]
    : analysisType === "truss3d" ? ["dx", "dy", "dz"]
    : ["dx", "dy", "dz", "rx", "ry", "rz"];
  for (let i = 0; i < totalDOF; i++) {
    const diag = K[i * totalDOF + i];
    if (diag < 1e-10) {
      const nodeIdx = Math.floor(i / dofPerNode);
      const dofIdx = i % dofPerNode;
      const nid = nodeIds[nodeIdx] || "?";
      const dir = dofLabels[dofIdx] || `DOF${dofIdx}`;
      throw Object.assign(
        new Error(`Node ${nid} has zero stiffness in ${dir} direction. No element provides stiffness at this DOF. Check element connectivity and boundary conditions.`),
        { phase: "boundary" },
      );
    }
  }

  // ── Step 6: Solve K*u = F (dense Cholesky) ─────────────────────
  progress("solve", 50, "Factorizing stiffness matrix...");
  const u = denseCholeskySolve(K, F, totalDOF, nodeIds, dofPerNode, dofLabels);
  progress("solve", 75, "Solution complete.");

  // ── Step 7: Extract results ────────────────────────────────────
  progress("postprocess", 80, "Extracting displacements...");

  // Displacements
  const displacements = {};
  for (const nid of nodeIds) {
    const s = nodeMap[nid];
    if (dofPerNode === 3 && analysisType === "beam2d") {
      displacements[nid] = { dx: u[s], dy: u[s + 1], dz: 0, rx: 0, ry: 0, rz: u[s + 2] };
    } else if (dofPerNode === 2) {
      displacements[nid] = { dx: u[s], dy: u[s + 1], dz: 0, rx: 0, ry: 0, rz: 0 };
    } else if (dofPerNode === 3 && analysisType === "truss3d") {
      displacements[nid] = { dx: u[s], dy: u[s + 1], dz: u[s + 2], rx: 0, ry: 0, rz: 0 };
    } else {
      displacements[nid] = { dx: u[s], dy: u[s + 1], dz: u[s + 2], rx: u[s + 3], ry: u[s + 4], rz: u[s + 5] };
    }
  }

  // Reactions
  progress("postprocess", 85, "Computing reactions...");
  const reactions = {};
  for (const [nid, rest] of Object.entries(restraints)) {
    const s = nodeMap[nid];
    if (s === undefined) continue;
    // R = K_row * u (from original K, but penalty is dominant)
    const r = {};
    const labels = analysisType === "beam2d" ? ["fx", "fy", "mz"]
      : analysisType === "truss2d" ? ["fx", "fy"]
        : analysisType === "truss3d" ? ["fx", "fy", "fz"]
          : ["fx", "fy", "fz", "mx", "my", "mz"];

    for (let d = 0; d < dofPerNode; d++) {
      const idx = s + d;
      let rVal = -F[idx]; // Start with applied load (negative because R + F = Ku)
      // Ku for this DOF
      let ku = 0;
      for (let j = 0; j < totalDOF; j++) {
        ku += (K[idx * totalDOF + j] - (idx === j ? PENALTY : 0)) * u[j];
      }
      r[labels[d]] = ku;
    }
    reactions[nid] = r;
  }

  // Element internal forces
  progress("postprocess", 90, "Recovering element forces...");
  const elementForces = {};
  for (const [elemId, elem] of Object.entries(elements)) {
    if (elem.nodeIds.length < 2) continue;
    const n1 = nodes[elem.nodeIds[0]];
    const n2 = nodes[elem.nodeIds[1]];
    if (!n1 || !n2) continue;

    const sec = sections[elem.sectionId] || {};
    const mat = materials[elem.materialId] || { E: 210000, nu: 0.3 };
    const E = mat.E || 210000;
    const A = sec.A || 1000;
    const Iy = sec.Iy || 1e6;

    const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
    const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (L < 1e-6) continue;

    // Get element DOF displacements
    const ue = [];
    for (const nid of elem.nodeIds) {
      const s = nodeMap[nid];
      for (let d = 0; d < dofPerNode; d++) ue.push(u[s + d]);
    }

    if (analysisType === "beam2d") {
      const angle = Math.atan2(dy, dx);
      const forces = beam2dRecoverForces(E, A, Iy, L, angle, ue);

      // Get distributed loads for this element
      const qLoads = getDistributedLoadsForElement(lc, elemId);
      const q = qLoads.qy || 0; // load intensity in local y (= global y for horizontal beams)

      // Fixed-end force correction:
      // beam2dRecoverForces returns K_local * u_local (forces from displacements only).
      // Total end forces = K*u - equivalent_nodal_loads  (where equiv. loads = [0, qL/2, qL²/12, 0, qL/2, -qL²/12])
      // This gives the actual internal forces at the element ends.
      const V1 = forces.V[0] - q * L / 2;
      const M1 = forces.M[0] - q * L * L / 12;

      const stations = [];
      const Mvals = [], Vvals = [], Nvals = [];
      const nSt = 11;
      for (let i = 0; i <= nSt - 1; i++) {
        const t = i / (nSt - 1);
        stations.push(t);
        const x = t * L;
        // Internal forces along beam from equilibrium of left segment:
        // V(x) = V1 + q*x  (shear decreases for downward load since q is negative)
        // M(x) = M1 + V1*x + q*x²/2  (moment from left-end forces + distributed load)
        Nvals.push(forces.N[0]);
        Vvals.push(V1 + q * x);
        Mvals.push(M1 + V1 * x + q * x * x / 2);
      }
      elementForces[elemId] = { stations, N: Nvals, V: Vvals, M: Mvals, Vy: Vvals, Vz: [], Mx: [], My: [], Mz: Mvals };
    } else if (analysisType === "truss2d" || analysisType === "truss3d") {
      const dim = analysisType === "truss2d" ? 2 : 3;
      const cx = dx / L, cy = dy / L, cz = dz / L;
      let elongation = 0;
      if (dim === 2) elongation = (ue[2] - ue[0]) * cx + (ue[3] - ue[1]) * cy;
      else elongation = (ue[3] - ue[0]) * cx + (ue[4] - ue[1]) * cy + (ue[5] - ue[2]) * cz;
      const N = E * A / L * elongation;
      elementForces[elemId] = { stations: [0, 1], N: [N, N], V: [0, 0], M: [0, 0], Vy: [0, 0], Vz: [0, 0], Mx: [0, 0], My: [0, 0], Mz: [0, 0] };
    } else {
      // frame3d — recover forces with intermediate stations and fixed-end corrections
      const nu = mat.nu || 0.3;
      const G = E / (2 * (1 + nu));
      const Iz = sec.Iz || 1e5;
      const J = sec.It || sec.J || 1e4;
      const forces = frame3dRecoverForces(E, G, A, Iy, Iz, J, L, n1, n2, ue);

      // Get distributed loads and project to local coords
      const qLoads = getDistributedLoadsForElement(lc, elemId);
      const qx = qLoads.qx || 0, qy = qLoads.qy || 0, qz = qLoads.qz || 0;
      const xL = [dx / L, dy / L, dz / L];
      let ref;
      if (Math.abs(xL[0]) < 0.01 && Math.abs(xL[2]) < 0.01) ref = [1, 0, 0];
      else ref = [0, 1, 0];
      const yL = cross(ref, xL);
      const yLen = Math.sqrt(yL[0] ** 2 + yL[1] ** 2 + yL[2] ** 2);
      if (yLen > 1e-10) { yL[0] /= yLen; yL[1] /= yLen; yL[2] /= yLen; }
      const zL = cross(xL, yL);
      const qlx = qx * xL[0] + qy * xL[1] + qz * xL[2];
      const qly = qx * yL[0] + qy * yL[1] + qz * yL[2];
      const qlz = qx * zL[0] + qy * zL[1] + qz * zL[2];

      // Fixed-end corrections: subtract equivalent nodal loads from K*u forces
      const N1  = forces.N[0]  - qlx * L / 2;
      const Vy1 = forces.Vy[0] - qly * L / 2;
      const Vz1 = forces.Vz[0] - qlz * L / 2;
      const Mx1 = forces.Mx[0]; // torsion unaffected by transverse UDL
      const My1 = forces.My[0] - (-qlz * L * L / 12); // fl[4] = -qlz*L²/12
      const Mz1 = forces.Mz[0] - (qly * L * L / 12);  // fl[5] = qly*L²/12

      // Intermediate stations
      const nSt = 11;
      const stations = [], Nvals = [], Vyvals = [], Vzvals = [], Mxvals = [], Myvals = [], Mzvals = [];
      for (let i = 0; i < nSt; i++) {
        const t = i / (nSt - 1);
        stations.push(t);
        const x = t * L;
        Nvals.push(N1 + qlx * x);
        Vyvals.push(Vy1 + qly * x);
        Vzvals.push(Vz1 + qlz * x);
        Mxvals.push(Mx1); // constant torsion
        Myvals.push(My1 + Vz1 * x + qlz * x * x / 2); // Note: My sign convention
        Mzvals.push(Mz1 + Vy1 * x + qly * x * x / 2);
      }
      elementForces[elemId] = {
        stations, N: Nvals,
        V: Vyvals, M: Mzvals,
        Vy: Vyvals, Vz: Vzvals,
        Mx: Mxvals, My: Myvals, Mz: Mzvals,
      };
    }
  }

  // Max values
  progress("postprocess", 95, "Computing summary...");
  let maxDisp = { nodeId: "", value: 0, direction: "" };
  for (const [nid, d] of Object.entries(displacements)) {
    const mag = Math.sqrt(d.dx * d.dx + d.dy * d.dy + d.dz * d.dz);
    if (mag > maxDisp.value) {
      maxDisp = { nodeId: nid, value: mag, direction: Math.abs(d.dy) > Math.abs(d.dx) && Math.abs(d.dy) > Math.abs(d.dz) ? "y" : Math.abs(d.dx) > Math.abs(d.dz) ? "x" : "z" };
    }
  }

  let maxMoment = { elementId: "", value: 0, unit: "N·mm" };
  let maxShear = { elementId: "", value: 0, unit: "N" };
  for (const [eid, ef] of Object.entries(elementForces)) {
    for (const m of (ef.M || [])) {
      if (Math.abs(m) > Math.abs(maxMoment.value)) maxMoment = { elementId: eid, value: m, unit: "N·mm" };
    }
    for (const v of (ef.V || [])) {
      if (Math.abs(v) > Math.abs(maxShear.value)) maxShear = { elementId: eid, value: v, unit: "N" };
    }
  }

  const solveTime = performance.now() - t0;
  progress("done", 100, `Solved in ${solveTime.toFixed(0)}ms`);

  return {
    displacements,
    reactions,
    elementForces,
    maxValues: { maxDisplacement: maxDisp, maxMoment, maxShear },
    solverInfo: { dofCount: totalDOF, solveTimeMs: Math.round(solveTime), elementCount: elemIds.length },
  };
}

// ══════════════════════════════════════════════════════════════════
// Inline element stiffness functions
// ══════════════════════════════════════════════════════════════════

function beam2dStiffnessMatrix(E, A, I, L, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const L2 = L * L, L3 = L2 * L;
  const ea = E * A / L;
  const k1 = 12 * E * I / L3;
  const k2 = 6 * E * I / L2;
  const k3 = 4 * E * I / L;
  const k4 = 2 * E * I / L;

  // Local stiffness
  const Kl = [
    ea, 0, 0, -ea, 0, 0,
    0, k1, k2, 0, -k1, k2,
    0, k2, k3, 0, -k2, k4,
    -ea, 0, 0, ea, 0, 0,
    0, -k1, -k2, 0, k1, -k2,
    0, k2, k4, 0, -k2, k3,
  ];

  // Transformation: T = [R 0; 0 R] where R is 3x3 block
  const T = new Float64Array(36);
  // First 3x3 block (rows 0-2, cols 0-2)
  T[0] = c;  T[1] = s;       // T[0][0]=c, T[0][1]=s
  T[6] = -s; T[7] = c;       // T[1][0]=-s, T[1][1]=c
  T[14] = 1;                   // T[2][2]=1
  // Second 3x3 block (rows 3-5, cols 3-5)
  T[21] = c; T[22] = s;      // T[3][3]=c, T[3][4]=s
  T[27] = -s; T[28] = c;     // T[4][3]=-s, T[4][4]=c
  T[35] = 1;                   // T[5][5]=1

  // K_global = Tᵀ * Kl * T
  return transformMatrix(Kl, T, 6);
}

function beam2dEquivLoadVector(q, L, angle) {
  // Fixed-end forces for UDL perpendicular to beam (in local coords, positive = downward)
  const fl = [0, q * L / 2, q * L * L / 12, 0, q * L / 2, -q * L * L / 12];
  const c = Math.cos(angle), s = Math.sin(angle);

  // Transform to global
  return new Float64Array([
    fl[0] * c - fl[1] * s,
    fl[0] * s + fl[1] * c,
    fl[2],
    fl[3] * c - fl[4] * s,
    fl[3] * s + fl[4] * c,
    fl[5],
  ]);
}

function beam2dRecoverForces(E, A, I, L, angle, ue) {
  const c = Math.cos(angle), s = Math.sin(angle);
  // Transform to local
  const ul = [
    ue[0] * c + ue[1] * s,
    -ue[0] * s + ue[1] * c,
    ue[2],
    ue[3] * c + ue[4] * s,
    -ue[3] * s + ue[4] * c,
    ue[5],
  ];
  const L2 = L * L, L3 = L2 * L;
  const N1 = E * A / L * (ul[3] - ul[0]);
  const V1 = E * I / L3 * (12 * (ul[1] - ul[4]) + 6 * L * (ul[2] + ul[5]));
  const M1 = E * I / L2 * (6 * (ul[1] - ul[4]) + L * (4 * ul[2] + 2 * ul[5]));
  const M2 = E * I / L2 * (6 * (ul[1] - ul[4]) + L * (2 * ul[2] + 4 * ul[5]));
  return { N: [N1, -N1], V: [V1, -V1], M: [M1, M2] };
}

function truss2dStiffnessMatrix(E, A, L, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const k = E * A / L;
  const cc = c * c, ss = s * s, cs = c * s;
  return new Float64Array([
    k * cc, k * cs, -k * cc, -k * cs,
    k * cs, k * ss, -k * cs, -k * ss,
    -k * cc, -k * cs, k * cc, k * cs,
    -k * cs, -k * ss, k * cs, k * ss,
  ]);
}

function truss3dStiffnessMatrix(E, A, L, n1, n2) {
  const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
  const cx = dx / L, cy = dy / L, cz = dz / L;
  const k = E * A / L;
  const d = [cx * cx, cx * cy, cx * cz, cy * cy, cy * cz, cz * cz];
  return new Float64Array([
    k * d[0], k * d[1], k * d[2], -k * d[0], -k * d[1], -k * d[2],
    k * d[1], k * d[3], k * d[4], -k * d[1], -k * d[3], -k * d[4],
    k * d[2], k * d[4], k * d[5], -k * d[2], -k * d[4], -k * d[5],
    -k * d[0], -k * d[1], -k * d[2], k * d[0], k * d[1], k * d[2],
    -k * d[1], -k * d[3], -k * d[4], k * d[1], k * d[3], k * d[4],
    -k * d[2], -k * d[4], -k * d[5], k * d[2], k * d[4], k * d[5],
  ]);
}

function frame3dStiffnessMatrix(E, G, A, Iy, Iz, J, L, n1, n2) {
  const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
  const L2 = L * L, L3 = L2 * L;

  // Local axis
  const xL = [dx / L, dy / L, dz / L];
  let ref;
  if (Math.abs(xL[0]) < 0.01 && Math.abs(xL[2]) < 0.01) ref = [1, 0, 0];
  else ref = [0, 1, 0];

  const yL = cross(ref, xL);
  const yLen = Math.sqrt(yL[0] * yL[0] + yL[1] * yL[1] + yL[2] * yL[2]);
  yL[0] /= yLen; yL[1] /= yLen; yL[2] /= yLen;

  const zL = cross(xL, yL);

  // 12x12 local stiffness
  const Kl = new Float64Array(144);
  const ea = E * A / L, gj = G * J / L;
  const eiy = [12 * E * Iy / L3, 6 * E * Iy / L2, 4 * E * Iy / L, 2 * E * Iy / L];
  const eiz = [12 * E * Iz / L3, 6 * E * Iz / L2, 4 * E * Iz / L, 2 * E * Iz / L];

  // Axial: DOFs 0, 6
  Kl[0 * 12 + 0] = ea;  Kl[0 * 12 + 6] = -ea;
  Kl[6 * 12 + 0] = -ea; Kl[6 * 12 + 6] = ea;
  // Torsion: DOFs 3, 9
  Kl[3 * 12 + 3] = gj;  Kl[3 * 12 + 9] = -gj;
  Kl[9 * 12 + 3] = -gj; Kl[9 * 12 + 9] = gj;
  // Bending about local z (v, θz): DOFs 1,5,7,11
  Kl[1 * 12 + 1] = eiz[0]; Kl[1 * 12 + 5] = eiz[1]; Kl[1 * 12 + 7] = -eiz[0]; Kl[1 * 12 + 11] = eiz[1];
  Kl[5 * 12 + 1] = eiz[1]; Kl[5 * 12 + 5] = eiz[2]; Kl[5 * 12 + 7] = -eiz[1]; Kl[5 * 12 + 11] = eiz[3];
  Kl[7 * 12 + 1] = -eiz[0]; Kl[7 * 12 + 5] = -eiz[1]; Kl[7 * 12 + 7] = eiz[0]; Kl[7 * 12 + 11] = -eiz[1];
  Kl[11 * 12 + 1] = eiz[1]; Kl[11 * 12 + 5] = eiz[3]; Kl[11 * 12 + 7] = -eiz[1]; Kl[11 * 12 + 11] = eiz[2];
  // Bending about local y (w, θy): DOFs 2,4,8,10
  Kl[2 * 12 + 2] = eiy[0]; Kl[2 * 12 + 4] = -eiy[1]; Kl[2 * 12 + 8] = -eiy[0]; Kl[2 * 12 + 10] = -eiy[1];
  Kl[4 * 12 + 2] = -eiy[1]; Kl[4 * 12 + 4] = eiy[2]; Kl[4 * 12 + 8] = eiy[1]; Kl[4 * 12 + 10] = eiy[3];
  Kl[8 * 12 + 2] = -eiy[0]; Kl[8 * 12 + 4] = eiy[1]; Kl[8 * 12 + 8] = eiy[0]; Kl[8 * 12 + 10] = eiy[1];
  Kl[10 * 12 + 2] = -eiy[1]; Kl[10 * 12 + 4] = eiy[3]; Kl[10 * 12 + 8] = eiy[1]; Kl[10 * 12 + 10] = eiy[2];

  // Build 12x12 transformation
  const T = new Float64Array(144);
  for (let b = 0; b < 4; b++) {
    const off = b * 3;
    T[(off + 0) * 12 + off + 0] = xL[0]; T[(off + 0) * 12 + off + 1] = xL[1]; T[(off + 0) * 12 + off + 2] = xL[2];
    T[(off + 1) * 12 + off + 0] = yL[0]; T[(off + 1) * 12 + off + 1] = yL[1]; T[(off + 1) * 12 + off + 2] = yL[2];
    T[(off + 2) * 12 + off + 0] = zL[0]; T[(off + 2) * 12 + off + 1] = zL[1]; T[(off + 2) * 12 + off + 2] = zL[2];
  }

  return transformMatrix(Kl, T, 12);
}

function frame3dEquivLoadVector(qx, qy, qz, L, n1, n2) {
  const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
  const xL = [dx / L, dy / L, dz / L];
  let ref;
  if (Math.abs(xL[0]) < 0.01 && Math.abs(xL[2]) < 0.01) ref = [1, 0, 0];
  else ref = [0, 1, 0];
  const yL = cross(ref, xL);
  const yLen = Math.sqrt(yL[0] ** 2 + yL[1] ** 2 + yL[2] ** 2);
  yL[0] /= yLen; yL[1] /= yLen; yL[2] /= yLen;
  const zL = cross(xL, yL);

  // Project global loads to local
  const qlx = qx * xL[0] + qy * xL[1] + qz * xL[2];
  const qly = qx * yL[0] + qy * yL[1] + qz * yL[2];
  const qlz = qx * zL[0] + qy * zL[1] + qz * zL[2];

  const L2 = L * L;
  // Local equivalent nodal loads
  const fl = new Float64Array(12);
  fl[0] = qlx * L / 2; fl[6] = qlx * L / 2;   // axial
  fl[1] = qly * L / 2; fl[7] = qly * L / 2;   // transverse y
  fl[5] = qly * L2 / 12; fl[11] = -qly * L2 / 12; // moments from y load
  fl[2] = qlz * L / 2; fl[8] = qlz * L / 2;   // transverse z
  fl[4] = -qlz * L2 / 12; fl[10] = qlz * L2 / 12; // moments from z load

  // Transform to global
  const T = new Float64Array(144);
  for (let b = 0; b < 4; b++) {
    const off = b * 3;
    T[(off + 0) * 12 + off + 0] = xL[0]; T[(off + 0) * 12 + off + 1] = xL[1]; T[(off + 0) * 12 + off + 2] = xL[2];
    T[(off + 1) * 12 + off + 0] = yL[0]; T[(off + 1) * 12 + off + 1] = yL[1]; T[(off + 1) * 12 + off + 2] = yL[2];
    T[(off + 2) * 12 + off + 0] = zL[0]; T[(off + 2) * 12 + off + 1] = zL[1]; T[(off + 2) * 12 + off + 2] = zL[2];
  }

  // f_global = Tᵀ * fl
  const fg = new Float64Array(12);
  for (let i = 0; i < 12; i++) {
    for (let j = 0; j < 12; j++) {
      fg[i] += T[j * 12 + i] * fl[j]; // Tᵀ[i][j] = T[j][i]
    }
  }
  return fg;
}

function frame3dRecoverForces(E, G, A, Iy, Iz, J, L, n1, n2, ue) {
  const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
  const xL = [dx / L, dy / L, dz / L];
  let ref;
  if (Math.abs(xL[0]) < 0.01 && Math.abs(xL[2]) < 0.01) ref = [1, 0, 0];
  else ref = [0, 1, 0];
  const yL = cross(ref, xL);
  const yLen = Math.sqrt(yL[0] ** 2 + yL[1] ** 2 + yL[2] ** 2);
  yL[0] /= yLen; yL[1] /= yLen; yL[2] /= yLen;
  const zL = cross(xL, yL);

  const T = new Float64Array(144);
  for (let b = 0; b < 4; b++) {
    const off = b * 3;
    T[(off + 0) * 12 + off + 0] = xL[0]; T[(off + 0) * 12 + off + 1] = xL[1]; T[(off + 0) * 12 + off + 2] = xL[2];
    T[(off + 1) * 12 + off + 0] = yL[0]; T[(off + 1) * 12 + off + 1] = yL[1]; T[(off + 1) * 12 + off + 2] = yL[2];
    T[(off + 2) * 12 + off + 0] = zL[0]; T[(off + 2) * 12 + off + 1] = zL[1]; T[(off + 2) * 12 + off + 2] = zL[2];
  }

  // Local displacements
  const ul = new Float64Array(12);
  for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) ul[i] += T[i * 12 + j] * ue[j];

  const L2 = L * L, L3 = L2 * L;
  const N1 = E * A / L * (ul[6] - ul[0]);
  const Vy1 = E * Iz / L3 * (12 * (ul[1] - ul[7]) + 6 * L * (ul[5] + ul[11]));
  const Vz1 = E * Iy / L3 * (12 * (ul[2] - ul[8]) - 6 * L * (ul[4] + ul[10]));
  const Mx1 = G * J / L * (ul[9] - ul[3]);
  const My1 = E * Iy / L2 * (-6 * (ul[2] - ul[8]) + L * (4 * ul[4] + 2 * ul[10]));
  const Mz1 = E * Iz / L2 * (6 * (ul[1] - ul[7]) + L * (4 * ul[5] + 2 * ul[11]));

  return {
    N: [N1, -N1],
    Vy: [Vy1, -Vy1],
    Vz: [Vz1, -Vz1],
    Mx: [Mx1, -Mx1],
    My: [My1, E * Iy / L2 * (-6 * (ul[2] - ul[8]) + L * (2 * ul[4] + 4 * ul[10]))],
    Mz: [Mz1, E * Iz / L2 * (6 * (ul[1] - ul[7]) + L * (2 * ul[5] + 4 * ul[11]))],
  };
}

function getDistributedLoadsForElement(lc, elemId) {
  if (!lc) return {};
  for (const load of lc.loads) {
    if (load.type === "distributed" && (load.element_id === elemId || load.elementId === elemId)) {
      return load;
    }
  }
  return {};
}

// ══════════════════════════════════════════════════════════════════
// Matrix helpers
// ══════════════════════════════════════════════════════════════════

function transformMatrix(Kl, T, n) {
  // K_global = Tᵀ * Kl * T
  // First: temp = Kl * T
  const temp = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += Kl[i * n + k] * T[k * n + j];
      temp[i * n + j] = s;
    }
  }
  // Then: result = Tᵀ * temp
  const result = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += T[k * n + i] * temp[k * n + j]; // Tᵀ[i][k] = T[k][i]
      result[i * n + j] = s;
    }
  }
  return result;
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function denseCholeskySolve(K, F, n, nodeIds, dofPerNode, dofLabels) {
  // In-place Cholesky factorization of dense symmetric matrix K
  // K is stored row-major in Float64Array of size n*n
  // Returns solution vector u
  const L = new Float64Array(n * n);

  // Copy K to L (lower triangle)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      L[i * n + j] = K[i * n + j];
    }
  }

  // Cholesky factorization
  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let k = 0; k < j; k++) sum += L[j * n + k] * L[j * n + k];
    const diag = L[j * n + j] - sum;
    if (diag <= 0) {
      // Build a helpful error message identifying the problematic node and DOF
      let detail = `DOF ${j}`;
      if (nodeIds && dofPerNode) {
        const nodeIdx = Math.floor(j / dofPerNode);
        const dofIdx = j % dofPerNode;
        const nid = nodeIds[nodeIdx] || "?";
        const dir = (dofLabels && dofLabels[dofIdx]) || `DOF${dofIdx}`;
        detail = `node ${nid}, ${dir} direction`;
      }
      throw Object.assign(
        new Error(`Structure is unstable at ${detail}. This usually means: (1) insufficient supports — add fixed or pinned restraints, (2) a mechanism — check that all members are properly connected, or (3) a node with no element attached.`),
        { phase: "solve" },
      );
    }
    L[j * n + j] = Math.sqrt(diag);

    for (let i = j + 1; i < n; i++) {
      let s = 0;
      for (let k = 0; k < j; k++) s += L[i * n + k] * L[j * n + k];
      L[i * n + j] = (L[i * n + j] - s) / L[j * n + j];
    }

    // Progress update every 100 rows
    if (j % 100 === 0 && n > 500) {
      progress("solve", 50 + (j / n) * 20, `Factorizing... ${Math.round(j / n * 100)}%`);
    }
  }

  // Forward substitution: L * y = F
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = F[i];
    for (let j = 0; j < i; j++) s -= L[i * n + j] * y[j];
    y[i] = s / L[i * n + i];
  }

  // Back substitution: Lᵀ * u = y
  const u = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) s -= L[j * n + i] * u[j]; // Lᵀ[i][j] = L[j][i]
    u[i] = s / L[i * n + i];
  }

  return u;
}
