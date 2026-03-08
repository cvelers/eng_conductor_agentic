#!/usr/bin/env python3
"""
Generate logo variations for Eurocode Chatbot using Gemini (Nano Banana / Nano Banana Pro).

Usage:
  pip install -r scripts/requirements-logo.txt
  # Set GEMINI_API_KEY or ORCHESTRATOR_API_KEY in .env
  python scripts/generate_logo.py
  # If ModuleNotFoundError: use .venv/bin/python3.11 scripts/generate_logo.py

Output: output/logo_variations/eurocode_logo_v1.png, v2.png, ...
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOGO_PROMPT = """Create a logo design for Eurocode Chatbot.

Style: neat, clean silhouette with minimal detail. Smart and tasteful.

Concept: a hybrid between an open book and a building — two parallelograms forming pages/walls. 
On the pages/walls: subtle lines suggesting writing (like text) or windows (like a building). 
The viewer should feel it sits right between book and building.

Color: blue accent on a simple silhouette. No letters or text.

Output: clean vector-style silhouette, high visual quality, professional taste."""


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("ORCHESTRATOR_API_KEY")
    if not key:
        raise SystemExit(
            "Set GEMINI_API_KEY or ORCHESTRATOR_API_KEY in .env "
            "(same key as your Gemini chat models)"
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

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    out_dir = Path(__file__).resolve().parent.parent / "output" / "logo_variations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best first: Nano Banana Pro (4K), then Nano Banana, then fallback
    models_to_try = [
        "gemini-3-pro-image-preview",      # Nano Banana Pro – best
        "gemini-2.5-flash-image",           # Nano Banana
        "gemini-2.0-flash-exp-image-generation",
    ]
    num_versions = 4
    print(f"Generating {num_versions} logo variations with Gemini (Nano Banana)...")

    for i in range(num_versions):
        print(f"  Version {i + 1}/{num_versions}...", end=" ", flush=True)
        response = None
        for j, model_id in enumerate(models_to_try):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=LOGO_PROMPT,
                    config=types.GenerateContentConfig(
                        response_modalities=["Image"],
                    ),
                )
                break
            except Exception as e:
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
                path = out_dir / f"eurocode_logo_v{i + 1}.png"
                img.save(path)
                print(f"saved {path}")
                saved = True
                break

        if not saved:
            print("no image in response")

    print(f"\nDone. Logos saved in: {out_dir}")


if __name__ == "__main__":
    main()
