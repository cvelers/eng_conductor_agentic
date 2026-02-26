from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

TOOL_MODULES = {
    "section_classification_ec3": "tools.mcp.section_classification",
    "member_resistance_ec3": "tools.mcp.member_resistance",
    "interaction_check_ec3": "tools.mcp.interaction_check",
    "ipe_moment_resistance_ec3": "tools.mcp.ipe_moment_resistance",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()

    if args.tool not in TOOL_MODULES:
        print(
            json.dumps(
                {
                    "tool": args.tool,
                    "status": "error",
                    "error": {"message": "Unknown tool."},
                }
            )
        )
        sys.exit(1)

    module = importlib.import_module(TOOL_MODULES[args.tool])

    model_type = None
    handler = None
    if args.tool == "section_classification_ec3":
        model_type = module.SectionClassificationInput
        handler = module.classify
    elif args.tool == "member_resistance_ec3":
        model_type = module.MemberResistanceInput
        handler = module.compute_resistance
    elif args.tool == "interaction_check_ec3":
        model_type = module.InteractionInput
        handler = module.check_interaction
    elif args.tool == "ipe_moment_resistance_ec3":
        model_type = module.IPEMomentResistanceInput
        handler = module.compute_ipe_moment_resistance

    payload: dict[str, Any] = json.loads(args.input_json)
    validated = model_type.model_validate(payload)
    result = handler(validated)
    print(json.dumps({"tool": args.tool, "status": "ok", "result": result}))


if __name__ == "__main__":
    main()
