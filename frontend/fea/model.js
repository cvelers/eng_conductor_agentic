/**
 * FEA Model — central data structure for the client-side FEA engine.
 *
 * Stores nodes, elements, materials, sections, restraints, and load cases.
 * Processes commands from the backend FEA analyst agent.
 * Serialises for Web Worker transfer.
 *
 * Internal units: mm, N, MPa (N/mm²), N·mm
 */

// ── Standard material presets (Eurocode steel grades) ─────────────
const MATERIAL_PRESETS = {
  S235: { E: 210000, nu: 0.3, rho: 7.85e-9, fy: 235, fu: 360, name: "S235" },
  S275: { E: 210000, nu: 0.3, rho: 7.85e-9, fy: 275, fu: 430, name: "S275" },
  S355: { E: 210000, nu: 0.3, rho: 7.85e-9, fy: 355, fu: 510, name: "S355" },
  S420: { E: 210000, nu: 0.3, rho: 7.85e-9, fy: 420, fu: 520, name: "S420" },
  S460: { E: 210000, nu: 0.3, rho: 7.85e-9, fy: 460, fu: 540, name: "S460" },
};

// ── Restraint presets ─────────────────────────────────────────────
const RESTRAINT_PRESETS = {
  pin:      { dx: true, dy: true, dz: true, rx: false, ry: false, rz: false },
  pinned:   { dx: true, dy: true, dz: true, rx: false, ry: false, rz: false },
  hinge:    { dx: true, dy: true, dz: true, rx: false, ry: false, rz: false },
  fixed:    { dx: true, dy: true, dz: true, rx: true,  ry: true,  rz: true  },
  fixed_support: { dx: true, dy: true, dz: true, rx: true,  ry: true,  rz: true  },
  encastre: { dx: true, dy: true, dz: true, rx: true,  ry: true,  rz: true  },
  roller:   { dx: false, dy: true, dz: true, rx: false, ry: false, rz: false },
  roller_x: { dx: false, dy: true, dz: true, rx: false, ry: false, rz: false },
  roller_y: { dx: true, dy: false, dz: true, rx: false, ry: false, rz: false },
  roller_z: { dx: true, dy: true, dz: false, rx: false, ry: false, rz: false },
  pin_2d:   { dx: true, dy: true, dz: false, rx: false, ry: false, rz: false },
  roller_2d:{ dx: false, dy: true, dz: false, rx: false, ry: false, rz: false },
  fixed_2d: { dx: true, dy: true, dz: false, rx: false, ry: false, rz: true  },
  simply_supported: { dx: true, dy: true, dz: false, rx: false, ry: false, rz: false },
};


export class FEAModel {
  constructor() {
    this.nodes = {};        // { id: { x, y, z } }
    this.elements = {};     // { id: { type, nodeIds, sectionId, materialId } }
    this.materials = {};    // { id: { E, nu, rho, fy, fu, name } }
    this.sections = {};     // { id: { profileName, A, Iy, Iz, J, Wply, Wely, h, b, tw, tf, r, ... } }
    this.restraints = {};   // { nodeId: { dx, dy, dz, rx, ry, rz } }
    this.loadCases = {};    // { id: { name, loads: [...] } }
    this.analysisType = "frame3d"; // "beam2d" | "frame3d" | "truss2d" | "truss3d"
    this.metadata = {};
    this._profileDb = null; // Lazy-loaded profile database
  }

  // ── Profile database setter (loaded externally) ────────────────
  setProfileDb(db) {
    this._profileDb = db;
  }

  // ── Command dispatcher — processes commands from FEA analyst ───
  applyCommand(cmd) {
    switch (cmd.action) {
      case "add_node":       return this._addNode(cmd);
      case "add_nodes":      return this._addNodes(cmd);
      case "add_element":    return this._addElement(cmd);
      case "add_elements":   return this._addElements(cmd);
      case "assign_section": return this._assignSection(cmd);
      case "assign_material":return this._assignMaterial(cmd);
      case "set_restraint":  return this._setRestraint(cmd);
      case "set_restraints": return this._setRestraints(cmd);
      case "add_load_case":  return this._addLoadCase(cmd);
      case "add_loads":      return this._addLoads(cmd);
      case "set_analysis_type": this.analysisType = cmd.type; return;
      case "clear":          return this._clear();
      case "solve":          return; // Handled via fea_solve_request, not model state
      default:
        console.warn(`FEAModel: unknown command action "${cmd.action}"`);
    }
  }

  applyCommands(commands) {
    for (const cmd of commands) {
      this.applyCommand(cmd);
    }
  }

  // ── Node operations ────────────────────────────────────────────
  _addNode(cmd) {
    this.nodes[cmd.id] = { x: cmd.x || 0, y: cmd.y || 0, z: cmd.z || 0 };
  }

  _addNodes(cmd) {
    for (const n of cmd.nodes) {
      this.nodes[n.id] = { x: n.x || 0, y: n.y || 0, z: n.z || 0 };
    }
  }

  // ── Element operations ─────────────────────────────────────────
  _addElement(cmd) {
    let nodeIds = cmd.node_ids || cmd.nodeIds || cmd.nodes || cmd.node_id || null;
    // Handle node_start/node_end, start_node/end_node, from/to, n1/n2 variants
    if (!nodeIds || (Array.isArray(nodeIds) && nodeIds.length === 0)) {
      const start = cmd.node_start || cmd.nodeStart || cmd.start_node || cmd.startNode || cmd.from_node || cmd.fromNode || cmd.from || cmd.node1 || cmd.n1 || cmd.start || cmd.i || cmd.node_i || cmd.nodeI;
      const end = cmd.node_end || cmd.nodeEnd || cmd.end_node || cmd.endNode || cmd.to_node || cmd.toNode || cmd.to || cmd.node2 || cmd.n2 || cmd.end || cmd.j || cmd.node_j || cmd.nodeJ;
      if (start != null && end != null) {
        nodeIds = [String(start), String(end)];
      }
    }
    if (!nodeIds) nodeIds = [];
    this.elements[cmd.id] = {
      type: cmd.type || "beam",
      nodeIds: Array.isArray(nodeIds) ? nodeIds.map(String) : [String(nodeIds)],
      sectionId: cmd.section_id || cmd.sectionId || null,
      materialId: cmd.material_id || cmd.materialId || null,
    };
  }

  _addElements(cmd) {
    for (const e of cmd.elements) {
      this._addElement({ ...e, action: "add_element" });
    }
  }

  // ── Section assignment ─────────────────────────────────────────
  _assignSection(cmd) {
    const elemIds = cmd.element_ids || cmd.elementIds || [];
    let secId, secProps;

    if (cmd.profile_name || cmd.profileName) {
      const profileName = cmd.profile_name || cmd.profileName;
      secId = `sec_${profileName}`;
      secProps = this._resolveProfile(profileName);
      if (!secProps) {
        // Use properties embedded by backend if available, otherwise defaults
        if (cmd.properties) {
          secProps = { profileName, ...cmd.properties };
        } else {
          console.warn(`FEAModel: profile "${profileName}" not found in database`);
          secProps = { profileName, A: 0, Iy: 0, Iz: 0, J: 0, h: 0, b: 0, tw: 0, tf: 0, r: 0 };
        }
      }
    } else if (cmd.properties) {
      secId = cmd.section_id || cmd.sectionId || `sec_custom_${Object.keys(this.sections).length}`;
      secProps = cmd.properties;
    } else {
      return;
    }

    this.sections[secId] = secProps;
    for (const eid of elemIds) {
      if (this.elements[eid]) this.elements[eid].sectionId = secId;
    }
  }

  _resolveProfile(profileName) {
    if (!this._profileDb) return null;
    const upper = profileName.toUpperCase();
    // Try each series
    for (const series of Object.values(this._profileDb)) {
      if (series[upper]) {
        return { profileName: upper, ...series[upper] };
      }
      // Try with original casing
      if (series[profileName]) {
        return { profileName, ...series[profileName] };
      }
    }
    return null;
  }

  // ── Material assignment ────────────────────────────────────────
  _assignMaterial(cmd) {
    const elemIds = cmd.element_ids || cmd.elementIds || [];
    let matId, matProps;

    if (cmd.grade) {
      const grade = cmd.grade.toUpperCase();
      matId = `mat_${grade}`;
      matProps = MATERIAL_PRESETS[grade] || { E: 210000, nu: 0.3, rho: 7.85e-9, fy: 355, fu: 510, name: grade };
    } else if (cmd.properties) {
      matId = cmd.material_id || cmd.materialId || `mat_custom_${Object.keys(this.materials).length}`;
      matProps = cmd.properties;
    } else {
      return;
    }

    this.materials[matId] = matProps;
    for (const eid of elemIds) {
      if (this.elements[eid]) this.elements[eid].materialId = matId;
    }
  }

  // ── Restraints ─────────────────────────────────────────────────
  _setRestraint(cmd) {
    const nodeId = cmd.node_id || cmd.nodeId;
    if (cmd.type) {
      // Normalize: lowercase, replace spaces/hyphens with underscores
      const normalized = cmd.type.toLowerCase().replace(/[\s-]/g, "_");
      const preset = RESTRAINT_PRESETS[normalized];
      if (preset) {
        this.restraints[nodeId] = { ...preset };
        return;
      }
      console.warn(`FEAModel: unknown restraint type "${cmd.type}" (normalized: "${normalized}"), falling back to DOF flags`);
    }
    this.restraints[nodeId] = {
      dx: !!cmd.dx, dy: !!cmd.dy, dz: !!cmd.dz,
      rx: !!cmd.rx, ry: !!cmd.ry, rz: !!cmd.rz,
    };
  }

  _setRestraints(cmd) {
    for (const r of cmd.restraints) {
      this._setRestraint({ ...r, action: "set_restraint" });
    }
  }

  // ── Load cases ─────────────────────────────────────────────────
  _addLoadCase(cmd) {
    const lcId = cmd.load_case_id || cmd.loadCaseId || "LC1";
    this.loadCases[lcId] = { name: cmd.name || lcId, loads: cmd.loads || [] };
  }

  _addLoads(cmd) {
    const lcId = cmd.load_case_id || cmd.loadCaseId || "LC1";
    if (!this.loadCases[lcId]) {
      this.loadCases[lcId] = { name: lcId, loads: [] };
    }
    const loads = cmd.loads || [];
    this.loadCases[lcId].loads.push(...loads);
  }

  // ── Clear ──────────────────────────────────────────────────────
  _clear() {
    this.nodes = {};
    this.elements = {};
    this.materials = {};
    this.sections = {};
    this.restraints = {};
    this.loadCases = {};
  }

  // ── Validation ─────────────────────────────────────────────────
  validate() {
    const warnings = [];
    const errors = [];

    // Check nodes exist
    if (Object.keys(this.nodes).length === 0) {
      errors.push("No nodes defined");
    }
    if (Object.keys(this.elements).length === 0) {
      errors.push("No elements defined");
    }

    // Check element connectivity
    for (const [eid, el] of Object.entries(this.elements)) {
      for (const nid of (el.nodeIds || [])) {
        if (!this.nodes[nid]) {
          errors.push(`Element ${eid} references undefined node ${nid}`);
        }
      }
      if (!el.sectionId || !this.sections[el.sectionId]) {
        warnings.push(`Element ${eid} has no section assigned`);
      }
      if (!el.materialId || !this.materials[el.materialId]) {
        warnings.push(`Element ${eid} has no material assigned`);
      }
    }

    // Check restraints
    if (Object.keys(this.restraints).length === 0) {
      errors.push("No restraints defined — structure is unstable");
    }

    // Check load cases
    if (Object.keys(this.loadCases).length === 0) {
      warnings.push("No load cases defined");
    }

    // Check for coincident nodes in elements (zero-length)
    for (const [eid, el] of Object.entries(this.elements)) {
      if ((el.nodeIds || []).length >= 2) {
        const n1 = this.nodes[el.nodeIds[0]];
        const n2 = this.nodes[el.nodeIds[1]];
        if (n1 && n2) {
          const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
          const L = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (L < 1e-6) {
            errors.push(`Element ${eid} has zero length (coincident nodes)`);
          }
        }
      }
    }

    // Connectivity check: all nodes must be reachable from supported nodes
    const allNodes = Object.keys(this.nodes);
    const supportedNodes = new Set(Object.keys(this.restraints));
    if (allNodes.length > 0 && supportedNodes.size > 0) {
      const adj = {};
      for (const nid of allNodes) adj[nid] = new Set();
      for (const [, el] of Object.entries(this.elements)) {
        const nids = el.nodeIds || [];
        if (nids.length >= 2 && adj[nids[0]] && adj[nids[1]]) {
          adj[nids[0]].add(nids[1]);
          adj[nids[1]].add(nids[0]);
        }
      }
      const visited = new Set();
      const queue = [...supportedNodes].filter(n => adj[n]);
      while (queue.length > 0) {
        const n = queue.pop();
        if (visited.has(n)) continue;
        visited.add(n);
        for (const nb of adj[n]) {
          if (!visited.has(nb)) queue.push(nb);
        }
      }
      const disconnected = allNodes.filter(n => !visited.has(n));
      if (disconnected.length > 0) {
        errors.push(`Disconnected nodes: ${disconnected.join(", ")} — not connected to any support via elements`);
      }
    }

    return { valid: errors.length === 0, errors, warnings };
  }

  // ── Serialization (for Web Worker transfer) ────────────────────
  serialize() {
    return {
      nodes: { ...this.nodes },
      elements: { ...this.elements },
      materials: { ...this.materials },
      sections: { ...this.sections },
      restraints: { ...this.restraints },
      loadCases: { ...this.loadCases },
      analysisType: this.analysisType,
    };
  }

  static deserialize(data) {
    const m = new FEAModel();
    m.nodes = data.nodes || {};
    m.elements = data.elements || {};
    m.materials = data.materials || {};
    m.sections = data.sections || {};
    m.restraints = data.restraints || {};
    m.loadCases = data.loadCases || {};
    m.analysisType = data.analysisType || "frame3d";
    return m;
  }

  // ── Helpers ────────────────────────────────────────────────────
  getNodeIds() { return Object.keys(this.nodes); }
  getElementIds() { return Object.keys(this.elements); }

  getBoundingBox() {
    const ids = this.getNodeIds();
    if (ids.length === 0) return { min: { x: 0, y: 0, z: 0 }, max: { x: 0, y: 0, z: 0 } };
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const id of ids) {
      const n = this.nodes[id];
      if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
      if (n.z < minZ) minZ = n.z; if (n.z > maxZ) maxZ = n.z;
    }
    return { min: { x: minX, y: minY, z: minZ }, max: { x: maxX, y: maxY, z: maxZ } };
  }

  getElementLength(elemId) {
    const el = this.elements[elemId];
    if (!el || el.nodeIds.length < 2) return 0;
    const n1 = this.nodes[el.nodeIds[0]];
    const n2 = this.nodes[el.nodeIds[1]];
    if (!n1 || !n2) return 0;
    const dx = n2.x - n1.x, dy = n2.y - n1.y, dz = n2.z - n1.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
}
