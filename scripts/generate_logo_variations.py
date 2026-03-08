#!/usr/bin/env python3
"""
Generate variations of an existing logo using Gemini (Nano Banana).
Pass a source image and edit prompts to create refined versions.

Usage:
  python scripts/generate_logo_variations.py [source_image]
  python scripts/generate_logo_variations.py [source_image] "custom edit prompt" [output_suffix]
  python scripts/generate_logo_variations.py [source_image] "prompt" [suffix] --ref [reference_image]
  python scripts/generate_logo_variations.py --from-ref [image] "prompt" [suffix]
  # --from-ref: use only the image (no base logo), generate from sketch/concept alone
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VARIATION_PROMPTS = [
    """Create a variation of this logo. Key change: make the squares on the right panel all equal in size and equally spaced — a clean uniform grid, no perspective effect. Keep everything else (structure, colors, left panel with lines) the same.""",
    """Same logo, but on the right side use a perfect uniform grid: all squares identical in size, equal horizontal and vertical spacing. Maintain the blue gradient and overall book/building hybrid aesthetic.""",
    """Edit this logo: the right panel's grid should be strictly uniform — same square dimensions, equal gaps between rows and columns. Remove the perspective so it reads as a flat, clean grid. Preserve the rest of the design.""",
    """Refine this design: replace the perspective grid on the right with a simple, uniform grid of equal squares equally spaced. Same blue accent, same smart silhouette style.""",
]


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("ORCHESTRATOR_API_KEY")
    if not key:
        raise SystemExit(
            "Set GEMINI_API_KEY or ORCHESTRATOR_API_KEY in .env"
        )
    return key


def main() -> None:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise SystemExit("Run: pip install google-genai Pillow python-dotenv")

    try:
        from PIL import Image
        from io import BytesIO
    except ImportError:
        raise SystemExit("Run: pip install Pillow")

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "output" / "logo_variations"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ref_idx = next((i for i, a in enumerate(sys.argv) if a == "--ref"), None)
    from_ref_idx = next((i for i, a in enumerate(sys.argv) if a == "--from-ref"), None)
    ref_path = Path(sys.argv[ref_idx + 1]) if ref_idx is not None and ref_idx + 1 < len(sys.argv) else None
    from_ref_path = Path(sys.argv[from_ref_idx + 1]) if from_ref_idx is not None and from_ref_idx + 1 < len(sys.argv) else None
    if ref_path is not None:
        args = [a for a in args if a != str(ref_path)]
    if from_ref_path is not None:
        args = [a for a in args if a != str(from_ref_path)]
    if from_ref_path is not None:
        source = str(out_dir / "eurocode_logo_v3.png")
        custom_prompt = args[0] if args else None
        output_suffix = args[1] if len(args) > 1 else "refined"
    else:
        source = args[0] if args else str(out_dir / "eurocode_logo_v3.png")
        custom_prompt = args[1] if len(args) > 1 else None
        output_suffix = args[2] if len(args) > 2 else ("refined" if custom_prompt else "")
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = base_dir / source_path
    from_ref_mode = from_ref_path is not None
    if from_ref_mode:
        from_ref_resolved = from_ref_path if from_ref_path.is_absolute() else base_dir / from_ref_path
        if not from_ref_resolved.exists():
            raise SystemExit(f"Image not found: {from_ref_resolved}")
        source_path = from_ref_resolved
        image_bytes = source_path.read_bytes()
        stem = "eurocode_from_sketch"
    elif not source_path.exists():
        raise SystemExit(f"Source image not found: {source_path}")

    ref_path_resolved: Path | None = None
    if ref_path is not None and not from_ref_mode:
        ref_path_resolved = ref_path if ref_path.is_absolute() else base_dir / ref_path
        if not ref_path_resolved.exists():
            raise SystemExit(f"Reference image not found: {ref_path_resolved}")

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    if not from_ref_mode:
        image_bytes = source_path.read_bytes()
        stem = source_path.stem

    prompts: list[str]
    if custom_prompt:
        prompts = [custom_prompt]
    elif len(sys.argv) > 1 and "refined" in str(source_path):
        prompts = [
            "Create another version nearly identical to this: same clean parallel geometry, same left-panel lines (equal thickness, well spaced, slightly thick). Very subtle variation only.",
            "Same design, same refinements. Produce a close variant with the same parallel structure and line styling.",
        ]
    else:
        prompts = VARIATION_PROMPTS

    models_to_try = [
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
        "gemini-2.0-flash-exp-image-generation",
    ]

    print(f"Creating {len(prompts)} variation(s) from {source_path.name}...")

    for i, prompt in enumerate(prompts):
        print(f"  Variation {i + 1}/{len(prompts)}...", end=" ", flush=True)
        parts: list = []
        if ref_path_resolved is not None:
            ref_bytes = ref_path_resolved.read_bytes()
            parts.extend([
                types.Part.from_bytes(data=ref_bytes, mime_type="image/png"),
                "Reference sketch/concept (first image). ",
            ])
        parts.extend([
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ])
        contents = parts
        response = None
        for j, model_id in enumerate(models_to_try):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["Text", "Image"] if "2.0" in model_id else ["Image"],
                    ),
                )
                break
            except Exception:
                if j < len(models_to_try) - 1:
                    print("trying next model...", flush=True)
                else:
                    raise

        if not response or not response.candidates:
            print("no output")
            continue

        saved = False
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                img = Image.open(BytesIO(part.inline_data.data))
                path = out_dir / (f"{stem}_{output_suffix}.png" if custom_prompt else f"{stem}_var{i + 1}.png")
                img.save(path)
                print(f"saved {path}")
                saved = True
                break

        if not saved:
            print("no image in response")

    print(f"\nDone. Variations saved in: {out_dir}")


if __name__ == "__main__":
    main()
