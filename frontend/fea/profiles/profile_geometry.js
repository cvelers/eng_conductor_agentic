/**
 * Profile geometry generator — creates THREE.Shape outlines for section extrusion.
 * Lazy-loads Three.js; all functions accept THREE as a parameter.
 */

/**
 * Generate an I-section shape (IPE, HEB, HEA, HEM).
 * Origin at centroid. Dimensions in mm.
 */
export function iSectionShape(THREE, { h, b, tw, tf, r = 0 }) {
  const halfH = h / 2;
  const halfB = b / 2;
  const halfTw = tw / 2;

  const shape = new THREE.Shape();

  // Start bottom-left of bottom flange, go clockwise
  shape.moveTo(-halfB, -halfH);
  shape.lineTo(halfB, -halfH);
  shape.lineTo(halfB, -halfH + tf);

  // Bottom-right fillet (or sharp corner)
  if (r > 0) {
    shape.lineTo(halfTw + r, -halfH + tf);
    shape.absarc(halfTw + r, -halfH + tf + r, r, -Math.PI / 2, Math.PI, true);
  } else {
    shape.lineTo(halfTw, -halfH + tf);
  }

  // Right side of web up
  shape.lineTo(halfTw, halfH - tf);

  // Top-right fillet
  if (r > 0) {
    shape.absarc(halfTw + r, halfH - tf - r, r, Math.PI, Math.PI / 2, true);
    shape.lineTo(halfB, halfH - tf);
  } else {
    shape.lineTo(halfB, halfH - tf);
  }

  // Top flange
  shape.lineTo(halfB, halfH);
  shape.lineTo(-halfB, halfH);
  shape.lineTo(-halfB, halfH - tf);

  // Top-left fillet
  if (r > 0) {
    shape.lineTo(-halfTw - r, halfH - tf);
    shape.absarc(-halfTw - r, halfH - tf - r, r, Math.PI / 2, 0, true);
  } else {
    shape.lineTo(-halfTw, halfH - tf);
  }

  // Left side of web down
  shape.lineTo(-halfTw, -halfH + tf);

  // Bottom-left fillet
  if (r > 0) {
    shape.absarc(-halfTw - r, -halfH + tf + r, r, 0, -Math.PI / 2, true);
    shape.lineTo(-halfB, -halfH + tf);
  } else {
    shape.lineTo(-halfB, -halfH + tf);
  }

  shape.lineTo(-halfB, -halfH);

  return shape;
}

/**
 * Generate a rectangular hollow section (RHS/SHS) shape.
 */
export function rhsShape(THREE, { h, b, t, r = 0 }) {
  const halfH = h / 2;
  const halfB = b / 2;
  const ri = Math.max(0, r - t);

  // Outer rectangle
  const outer = new THREE.Shape();
  if (r > 0) {
    outer.moveTo(-halfB + r, -halfH);
    outer.lineTo(halfB - r, -halfH);
    outer.absarc(halfB - r, -halfH + r, r, -Math.PI / 2, 0, false);
    outer.lineTo(halfB, halfH - r);
    outer.absarc(halfB - r, halfH - r, r, 0, Math.PI / 2, false);
    outer.lineTo(-halfB + r, halfH);
    outer.absarc(-halfB + r, halfH - r, r, Math.PI / 2, Math.PI, false);
    outer.lineTo(-halfB, -halfH + r);
    outer.absarc(-halfB + r, -halfH + r, r, Math.PI, 3 * Math.PI / 2, false);
  } else {
    outer.moveTo(-halfB, -halfH);
    outer.lineTo(halfB, -halfH);
    outer.lineTo(halfB, halfH);
    outer.lineTo(-halfB, halfH);
    outer.lineTo(-halfB, -halfH);
  }

  // Inner hole
  const iH = halfH - t;
  const iB = halfB - t;
  const hole = new THREE.Path();
  if (ri > 0) {
    hole.moveTo(-iB + ri, -iH);
    hole.lineTo(iB - ri, -iH);
    hole.absarc(iB - ri, -iH + ri, ri, -Math.PI / 2, 0, false);
    hole.lineTo(iB, iH - ri);
    hole.absarc(iB - ri, iH - ri, ri, 0, Math.PI / 2, false);
    hole.lineTo(-iB + ri, iH);
    hole.absarc(-iB + ri, iH - ri, ri, Math.PI / 2, Math.PI, false);
    hole.lineTo(-iB, -iH + ri);
    hole.absarc(-iB + ri, -iH + ri, ri, Math.PI, 3 * Math.PI / 2, false);
  } else {
    hole.moveTo(-iB, -iH);
    hole.lineTo(iB, -iH);
    hole.lineTo(iB, iH);
    hole.lineTo(-iB, iH);
    hole.lineTo(-iB, -iH);
  }
  outer.holes.push(hole);

  return outer;
}

/**
 * Generate a circular hollow section (CHS) shape.
 */
export function chsShape(THREE, { d, t }) {
  const outerR = d / 2;
  const innerR = outerR - t;

  const outer = new THREE.Shape();
  outer.absarc(0, 0, outerR, 0, Math.PI * 2, false);

  const hole = new THREE.Path();
  hole.absarc(0, 0, innerR, 0, Math.PI * 2, true);
  outer.holes.push(hole);

  return outer;
}

/**
 * Generate a solid rectangular section.
 */
export function rectShape(THREE, { h, b }) {
  const shape = new THREE.Shape();
  shape.moveTo(-b / 2, -h / 2);
  shape.lineTo(b / 2, -h / 2);
  shape.lineTo(b / 2, h / 2);
  shape.lineTo(-b / 2, h / 2);
  shape.lineTo(-b / 2, -h / 2);
  return shape;
}

/**
 * Generate a solid circular section.
 */
export function circleShape(THREE, { d }) {
  const shape = new THREE.Shape();
  shape.absarc(0, 0, d / 2, 0, Math.PI * 2, false);
  return shape;
}

/**
 * Auto-detect section type and generate shape.
 */
export function getProfileShape(THREE, sectionProps) {
  const name = (sectionProps.profileName || "").toUpperCase();

  // I-sections (IPE, HEA, HEB, HEM)
  if (name.startsWith("IPE") || name.startsWith("HE") || sectionProps.tf) {
    return iSectionShape(THREE, sectionProps);
  }

  // CHS
  if (name.startsWith("CHS") || (sectionProps.d && sectionProps.t && !sectionProps.b)) {
    return chsShape(THREE, sectionProps);
  }

  // RHS / SHS
  if (name.startsWith("RHS") || name.startsWith("SHS") || (sectionProps.h && sectionProps.b && sectionProps.t)) {
    return rhsShape(THREE, sectionProps);
  }

  // Fallback: simple rectangle based on A
  const side = Math.sqrt(sectionProps.A || 1000);
  return rectShape(THREE, { h: side, b: side });
}
