/**
 * Shared load helpers for the frontend FEA model, solver, and viewer.
 *
 * Internal units:
 * - length: mm
 * - force: N
 * - stress: MPa = N/mm^2
 * - density: kg/mm^3
 */

function getEntry(mapOrObj, key) {
  if (mapOrObj instanceof Map) return mapOrObj.get(key);
  return mapOrObj?.[key];
}

export function getSelfWeightAcceleration(load, analysisType) {
  const factor = load?.factor !== undefined ? load.factor : 1.0;
  const g = load?.g || 9.81;

  if (load?.direction && typeof load.direction === "object") {
    return {
      gx: (load.direction.x || 0) * g * factor,
      gy: (load.direction.y || 0) * g * factor,
      gz: (load.direction.z || 0) * g * factor,
    };
  }

  if (analysisType === "frame3d" || analysisType === "truss3d") {
    return { gx: 0, gy: 0, gz: -g * factor };
  }

  return { gx: 0, gy: -g * factor, gz: 0 };
}

export function getSelfWeightDistributedLoad(model, element, load) {
  const section = getEntry(model.sections, element?.sectionId);
  const material = getEntry(model.materials, element?.materialId);
  if (!section || !material) return null;

  const area = Number(section.A || 0);
  const density = Number(material.rho);
  if (!(area > 0) || !(density > 0)) return null;

  const { gx, gy, gz } = getSelfWeightAcceleration(load, model.analysisType);
  return {
    qx: density * area * gx,
    qy: density * area * gy,
    qz: density * area * gz,
  };
}
