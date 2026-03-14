from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_self_weight_reactions_and_moment_scale_correctly() -> None:
    script = """
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = process.cwd();
const { solveFEA } = await import(pathToFileURL(path.join(root, "frontend/fea/solver/solver_manager.js")).href);

const model = {
  analysisType: "beam2d",
  nodes: {
    N1: { x: 0, y: 0, z: 0 },
    N2: { x: 5000, y: 0, z: 0 },
  },
  elements: {
    E1: { type: "beam2d", nodeIds: ["N1", "N2"], sectionId: "SEC1", materialId: "MAT1" },
  },
  sections: {
    SEC1: { A: 10000, Iy: 8.0e7 },
  },
  materials: {
    MAT1: { E: 210000, nu: 0.3, rho: 7.85e-6 },
  },
  supports: {
    SUP_N1: { nodeId: "N1", conditions: { dx: true, dy: true, rz: false } },
    SUP_N2: { nodeId: "N2", conditions: { dx: false, dy: true, rz: false } },
  },
  loadCases: {
    LC1: {
      loads: [{ type: "self_weight", factor: 1.0, direction: { x: 0, y: -1, z: 0 } }],
    },
  },
};

const result = solveFEA(model, "LC1");
const totalFy = Object.values(result.reactions).reduce((sum, reaction) => sum + Math.abs(reaction.fy || 0), 0);
const maxMoment = Math.abs(result.maxValues.maxMoment.value || 0);

console.log(JSON.stringify({ totalFy, maxMoment }));
"""
    completed = subprocess.run(
        ["node", "--experimental-default-type=module", "--input-type=module", "-e", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(completed.stdout.strip())
    assert data["totalFy"] == pytest.approx(3850.425, rel=1e-3)
    assert data["maxMoment"] == pytest.approx(2406515.625, rel=1e-3)
