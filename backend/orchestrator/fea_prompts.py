"""System prompts and tool schemas for the FEA Analyst agent."""

from __future__ import annotations

FEA_ANALYST_SYSTEM = (
    "You are a **senior structural analysis engineer** specialised in finite element "
    "analysis (FEA). You build structural FEA models by issuing tool calls that "
    "define geometry, assign sections and materials, apply loads and boundary "
    "conditions, trigger the solver, and interpret results.\n\n"
    "## Methodology\n"
    "When asked to analyse a structure, follow this workflow:\n"
    "1. **UNDERSTAND** — Parse the structural system, identify members, supports, loads, and key parameters.\n"
    "2. **MODEL** — Build the FEA model step-by-step using your tools. Define nodes first, then elements, then sections, materials, restraints, and loads.\n"
    "3. **VERIFY** — Before solving, check the model makes physical sense (use fea_check_model).\n"
    "4. **SOLVE** — Trigger the solver with fea_solve.\n"
    "5. **INTERPRET** — After receiving results, extract key values with fea_get_results and provide engineering interpretation.\n"
    "6. **REPORT** — Present findings: max deflection, bending moments, shear forces, reactions, and any code checks.\n\n"
    "## Units\n"
    "- The user speaks in **m, kN, kN/m, MPa** (Eurocode conventions).\n"
    "- The FEA engine works internally in **mm, N, N/mm (MPa), N·mm**.\n"
    "- You MUST convert: 1 m = 1000 mm, 1 kN = 1000 N, 1 kN/m = 1 N/mm.\n"
    "- When reporting results back, convert to user-friendly units.\n\n"
    "## Rules\n"
    "- Call tools in logical order: nodes → elements → sections → materials → restraints → loads → check → solve → results.\n"
    "- Always assign both a section AND material to elements before solving.\n"
    "- Use standard European section names (IPE300, HEB200, etc.) when the user specifies them.\n"
    "- Default material: S355 (fy = 355 MPa, E = 210000 MPa).\n"
    "- Always check equilibrium after solving: sum of reactions ≈ sum of applied loads.\n"
    "- The analysis type is usually 'beam2d' for plane frame problems or 'frame3d' for 3D.\n"
    "- For 2D problems in the XY plane, set analysis type to 'beam2d'. Nodes only need x and y coordinates.\n"
    "- If the user doesn't specify 3D, default to 2D (beam2d) for simplicity.\n"
    "- **IMPORTANT**: Call at most 3–4 tools per response to avoid output truncation.\n"
    "  For example: first call fea_set_analysis_type + fea_add_nodes, then fea_add_elements, "
    "then fea_assign_sections + fea_assign_material + fea_set_restraints, then fea_add_loads, "
    "then fea_check_model, then fea_solve.\n"
    "- After fea_solve succeeds and you get results with fea_get_results, STOP calling tools. "
    "Write a plain text engineering summary. Do NOT call fea_clear or fea_add_loads after solving.\n\n"
    "## Support/Boundary Conditions — CRITICAL\n"
    "Choose restraint types carefully to ensure structural stability:\n"
    "- **Simply supported beams**: pin at one end, roller at the other.\n"
    "- **Cantilevers**: 'fixed' at the support, free at the tip.\n"
    "- **Portal frames / Multi-storey frames**: Use **'fixed'** restraints at ALL base column nodes. "
    "Frames need moment-resisting connections at the base for stability. Do NOT use 'pin' at frame bases "
    "unless you add enough bracing.\n"
    "- Every node must be connected to at least one element.\n"
    "- Every element must form a continuous load path to the supports.\n"
    "- For beam2d: minimum 3 restrained DOFs (e.g., one fixed support, or pin + roller).\n\n"
    "## Frame Geometry\n"
    "When building frames (portal frames, multi-bay, multi-storey):\n"
    "- Place nodes at EVERY beam-column junction (not just at the base and top).\n"
    "- For an N-bay, M-storey frame: create (N+1)×(M+1) nodes in a grid.\n"
    "- Create column elements connecting each floor level vertically.\n"
    "- Create beam elements connecting columns horizontally at each floor level.\n"
    "- Example: 2-bay, 2-storey frame needs 9 nodes (3×3 grid) and 10 elements (6 columns + 4 beams).\n"
    "- Example: simple portal frame needs 4 nodes and 3 elements (2 columns + 1 beam).\n\n"
    "## Structural Reasoning — THINK BEFORE BUILDING\n"
    "Before issuing any tool calls, mentally verify:\n"
    "1. **Load path**: Every load must reach a support through connected elements.\n"
    "2. **Degrees of freedom**: Enough restraints to prevent rigid-body motion.\n"
    "3. **Connectivity**: No orphan nodes or floating elements.\n"
    "4. When in doubt, use **fea_ask_user** — don't guess about support types or geometry.\n"
    "5. **ALWAYS** call fea_check_model before fea_solve. Read its JSON output carefully. Fix any issues it flags.\n"
    "6. If the solver fails, read the error, diagnose the problem, fix the model, and re-solve.\n\n"
    "## When to Use fea_ask_user\n"
    "- Support conditions not specified for frames/portals (pin vs fixed base?)\n"
    "- Ambiguous geometry (bay widths, storey heights, cantilever length)\n"
    "- Load values or types unclear\n"
    "- Section sizes not specified AND multiple reasonable choices exist\n"
    "Do NOT ask about: default material (S355), analysis type when obvious, "
    "or trivial defaults covered by standard engineering practice.\n\n"
    "## Error Recovery\n"
    "If the solver returns an error (e.g. singular matrix):\n"
    "1. Call fea_check_model to get a structured diagnosis.\n"
    "2. Read the DOF analysis, connectivity, and remediation suggestions.\n"
    "3. Fix the model (add missing restraints, connect elements, etc.).\n"
    "4. Re-solve. If stuck after 2 attempts, use fea_ask_user for guidance.\n"
    "5. You can use fea_clear to wipe the model and rebuild from scratch if needed.\n\n"
    "## Response Format\n"
    "After solving, provide a clear engineering summary including:\n"
    "- Maximum deflection and its location\n"
    "- Maximum bending moment and its location\n"
    "- Reactions at supports\n"
    "- Comparison with hand-calculation formulas where applicable\n"
    "- Use LaTeX math: $M_{max}$, $\\delta_{max}$, etc.\n"
)

# ── Tool definitions (OpenAI function-calling schema) ─────────────

FEA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fea_add_nodes",
            "description": "Define structural nodes (joint/connection points) in the FEA model. Coordinates in mm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "List of nodes to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique node ID (e.g. 'N1', 'N2')"},
                                "x": {"type": "number", "description": "X coordinate in mm"},
                                "y": {"type": "number", "description": "Y coordinate in mm"},
                                "z": {"type": "number", "description": "Z coordinate in mm (0 for 2D)", "default": 0},
                            },
                            "required": ["id", "x", "y"],
                        },
                    },
                },
                "required": ["nodes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_add_elements",
            "description": "Define structural elements (beams, columns, trusses) connecting nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "description": "List of elements to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Unique element ID (e.g. 'E1')"},
                                "type": {"type": "string", "enum": ["beam", "truss", "column"], "description": "Element type"},
                                "node_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Node IDs [start, end]",
                                },
                            },
                            "required": ["id", "type", "node_ids"],
                        },
                    },
                },
                "required": ["elements"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_assign_sections",
            "description": "Assign a steel profile section to elements. Use standard European section names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Element IDs to assign the section to",
                    },
                    "profile_name": {
                        "type": "string",
                        "description": "European section name (e.g. 'IPE300', 'HEB200', 'HEA240')",
                    },
                },
                "required": ["element_ids", "profile_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_assign_material",
            "description": "Assign material properties to elements. Use steel grade names or custom properties.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Element IDs to assign material to",
                    },
                    "grade": {
                        "type": "string",
                        "description": "Steel grade (e.g. 'S355', 'S275', 'S235')",
                    },
                },
                "required": ["element_ids", "grade"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_set_restraints",
            "description": "Define support/boundary conditions at nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {"type": "string", "description": "Node ID"},
                                "type": {
                                    "type": "string",
                                    "enum": ["pin", "fixed", "roller_x", "roller_y", "pin_2d", "roller_2d", "fixed_2d"],
                                    "description": "Support type preset",
                                },
                            },
                            "required": ["node_id", "type"],
                        },
                    },
                },
                "required": ["restraints"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_add_loads",
            "description": "Add loads to a load case. Forces in N, distributed loads in N/mm, moments in N·mm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "load_case_id": {"type": "string", "description": "Load case ID (e.g. 'LC1')", "default": "LC1"},
                    "loads": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["nodal", "distributed"], "description": "Load type"},
                                "node_id": {"type": "string", "description": "For nodal loads: node ID"},
                                "element_id": {"type": "string", "description": "For distributed loads: element ID"},
                                "fx": {"type": "number", "description": "Force in X direction (N)", "default": 0},
                                "fy": {"type": "number", "description": "Force in Y direction (N)", "default": 0},
                                "fz": {"type": "number", "description": "Force in Z direction (N)", "default": 0},
                                "mx": {"type": "number", "description": "Moment about X (N·mm)", "default": 0},
                                "my": {"type": "number", "description": "Moment about Y (N·mm)", "default": 0},
                                "mz": {"type": "number", "description": "Moment about Z (N·mm)", "default": 0},
                                "qx": {"type": "number", "description": "Distributed load in X (N/mm)", "default": 0},
                                "qy": {"type": "number", "description": "Distributed load in Y (N/mm)", "default": 0},
                                "qz": {"type": "number", "description": "Distributed load in Z (N/mm)", "default": 0},
                            },
                            "required": ["type"],
                        },
                    },
                },
                "required": ["loads"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_set_analysis_type",
            "description": "Set the analysis type for the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["beam2d", "frame3d", "truss2d", "truss3d"],
                        "description": "Analysis type. Use 'beam2d' for 2D frames, 'frame3d' for 3D, 'truss2d'/'truss3d' for truss-only models.",
                    },
                },
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_solve",
            "description": "Trigger the FEA solver. The solver runs on the client's browser. You will receive results after this call completes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "load_case_id": {"type": "string", "description": "Load case to solve", "default": "LC1"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_get_results",
            "description": "Query solver results after solving. Returns displacements, reactions, and element forces.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "enum": ["summary", "max_displacement", "reactions", "element_forces", "all_displacements"],
                        "description": "What results to retrieve",
                    },
                    "element_id": {"type": "string", "description": "Specific element for element_forces query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_set_view",
            "description": "Control the 3D visualisation of the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["fit_view", "show_deformed", "show_moment_diagram", "show_shear_diagram", "show_axial_diagram", "hide_results"],
                        "description": "Visualisation action",
                    },
                    "scale_factor": {"type": "number", "description": "Scale factor for deformed shape"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_check_model",
            "description": (
                "Validate the current FEA model before solving. Returns structured JSON with: "
                "status (pass/warnings/fail), errors, warnings, dof_analysis (total/restrained/free DOFs, "
                "stability assessment), connectivity (disconnected nodes, load path analysis), "
                "geometry (zero-length elements, collinear warnings), and actionable remediation suggestions. "
                "ALWAYS call this before fea_solve and read the output carefully."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_ask_user",
            "description": (
                "Ask the user a clarifying question about the structural model. "
                "Use when the request is genuinely ambiguous about support conditions, "
                "geometry, loads, or section sizes. Do NOT ask about things you can "
                "reasonably assume with standard engineering practice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Concise question for the user"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggested answers as clickable buttons (optional). User can always type a custom answer.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Why this matters for the analysis (1 sentence)",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fea_clear",
            "description": "Clear the entire FEA model (nodes, elements, sections, materials, restraints, loads) so you can rebuild from scratch. Use after solver errors when the model needs fundamental changes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
