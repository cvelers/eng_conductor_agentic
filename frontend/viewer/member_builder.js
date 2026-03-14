/**
 * Member builder — creates extruded 3D meshes for structural members.
 * Takes FEA model data and generates Three.js geometry.
 */

import { getProfileShape } from "../fea/profiles/profile_geometry.js";
import { getThree } from "./scene.js";

const STEEL_COLOR = 0x8899aa;  // steel blue-gray
const STEEL_EMISSIVE = 0x0a0a14;

/**
 * Build a 3D mesh for a beam/column member by extruding its cross-section profile.
 *
 * @param {Object} node1 - Start node {x, y, z} in mm
 * @param {Object} node2 - End node {x, y, z} in mm
 * @param {Object} sectionProps - Section properties including profileName, h, b, tw, tf, r
 * @param {Object} options - Optional: { color, opacity, wireframe }
 * @returns {THREE.Mesh}
 */
export function buildMemberMesh(node1, node2, sectionProps, options = {}) {
  const THREE = getThree();
  if (!THREE) return null;

  // Scale factor: section props are in mm, but we render at 1:1 scale
  const shape = getProfileShape(THREE, sectionProps);
  if (!shape) return null;

  const p1 = new THREE.Vector3(node1.x, node1.y, node1.z);
  const p2 = new THREE.Vector3(node2.x, node2.y, node2.z);

  const direction = new THREE.Vector3().subVectors(p2, p1);
  const length = direction.length();
  if (length < 0.01) return null;

  // Create extrusion path along Z, then orient
  const extrudePath = new THREE.LineCurve3(
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, length),
  );

  const geometry = new THREE.ExtrudeGeometry(shape, {
    extrudePath,
    steps: 1,
    bevelEnabled: false,
  });

  geometry.computeVertexNormals();

  const material = new THREE.MeshLambertMaterial({
    color: options.color || STEEL_COLOR,
    emissive: STEEL_EMISSIVE,
    transparent: !!options.opacity,
    opacity: options.opacity || 1,
    wireframe: !!options.wireframe,
    side: THREE.FrontSide,
    // Push solid mesh slightly back so edge wireframes render cleanly on top
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });

  const mesh = new THREE.Mesh(geometry, material);

  // Orient the mesh: extrusion is along Z, we need to align Z with the member direction
  const up = new THREE.Vector3(0, 0, 1);
  const dir = direction.clone().normalize();

  // Quaternion to rotate from Z axis to member direction
  const quat = new THREE.Quaternion().setFromUnitVectors(up, dir);
  mesh.quaternion.copy(quat);

  // Position at start node
  mesh.position.copy(p1);

  mesh.userData = { type: "member", sectionProps };

  return mesh;
}

/**
 * Build a simple line representation of a member (for wireframe mode).
 */
export function buildMemberLine(node1, node2, color = 0x4f9cf7) {
  const THREE = getThree();
  if (!THREE) return null;

  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(node1.x, node1.y, node1.z),
    new THREE.Vector3(node2.x, node2.y, node2.z),
  ]);

  const material = new THREE.LineBasicMaterial({ color, linewidth: 2 });
  return new THREE.Line(geometry, material);
}

/**
 * Build all members from an FEA model.
 * Returns Map<elemId, THREE.Mesh>.
 */
export function buildAllMembers(model) {
  const meshes = new Map();

  for (const [elemId, elem] of Object.entries(model.elements)) {
    if (!elem.nodeIds || elem.nodeIds.length < 2) continue;

    const n1 = model.nodes[elem.nodeIds[0]];
    const n2 = model.nodes[elem.nodeIds[1]];
    if (!n1 || !n2) continue;

    const sec = elem.sectionId ? model.sections[elem.sectionId] : null;

    let mesh;
    if (sec && (sec.h || sec.profileName)) {
      mesh = buildMemberMesh(n1, n2, sec);
    } else {
      // Fallback: thin line
      mesh = buildMemberLine(n1, n2);
    }

    if (mesh) {
      mesh.userData.elemId = elemId;
      meshes.set(elemId, mesh);
    }
  }

  return meshes;
}

/**
 * Build edge wireframes from the actual 3D member meshes using EdgesGeometry.
 * This highlights the true 3D profile edges of the extruded cross-sections.
 *
 * @param {Map<string, THREE.Mesh>} memberMeshes - Map of elemId → THREE.Mesh from buildAllMembers
 * @returns {THREE.LineSegments[]}
 */
export function buildProfileEdges(memberMeshes) {
  const THREE = getThree();
  if (!THREE) return [];

  const edges = [];
  const EDGE_COLOR = 0xccddff;  // near-white blue — very visible

  for (const [elemId, mesh] of memberMeshes) {
    if (!mesh.geometry) continue;

    const edgesGeo = new THREE.EdgesGeometry(mesh.geometry, 15);
    const mat = new THREE.LineBasicMaterial({
      color: EDGE_COLOR,
      depthTest: false,   // always render on top
    });
    const lineSegments = new THREE.LineSegments(edgesGeo, mat);
    lineSegments.renderOrder = 1;  // draw after solid meshes

    // Copy the mesh's transform so edges align with the member
    lineSegments.position.copy(mesh.position);
    lineSegments.quaternion.copy(mesh.quaternion);
    lineSegments.scale.copy(mesh.scale);

    lineSegments.userData = { type: "profileBox", elemId };
    edges.push(lineSegments);
  }

  return edges;
}
