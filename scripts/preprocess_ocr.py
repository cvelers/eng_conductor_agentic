"""Pre-process OCR JSON into structured clauses with proper IDs and titles."""
from __future__ import annotations

import json
import re
from pathlib import Path

CLAUSE_HEADING_RE = re.compile(
    r"(?:^|\n)"
    r"((?:Annex\s+[A-Z]+|Table)\s+[\w.]+|"
    r"\d{1,2}(?:\.\d{1,2}){0,3})"
    r"\s+"
    r"([A-Z][^\n]{3,80})"
)

SKIP_TITLE_PREFIXES = ("the ", "this ", "for ", "in ", "a ", "an ", "where ", "see ", "if ")


def load_ocr(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pages: list[str] = []
    for item in raw:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict) and sub.get("content"):
                    pages.append(sub["content"])
        elif isinstance(item, dict) and item.get("content"):
            pages.append(item["content"])
    return pages


def extract_clauses(pages: list[str]) -> list[dict]:
    full_text = "\n\n".join(pages)

    headings: list[tuple[int, str, str]] = []
    for m in CLAUSE_HEADING_RE.finditer(full_text):
        clause_id = m.group(1).strip()
        title = m.group(2).strip().rstrip(".")
        pos = m.start()

        if len(clause_id) < 2:
            continue
        if title.lower().startswith(SKIP_TITLE_PREFIXES):
            continue
        if len(title) < 4 or len(title) > 80:
            continue
        if re.match(r"^\d+$", title):
            continue
        if "..." in title:
            continue

        headings.append((pos, clause_id, title))

    headings.sort(key=lambda x: x[0])

    segments: list[tuple[int, int, str, str]] = []
    for i, (pos, cid, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(full_text)
        segments.append((pos, end, cid, title))

    best: dict[str, tuple[str, str, int]] = {}
    for pos, end, cid, title in segments:
        body = full_text[pos:end].strip()
        header_line = f"{cid} {title}"
        if body.startswith(header_line):
            body = body[len(header_line):].strip()

        body_len = len(body)
        if cid not in best or body_len > best[cid][2]:
            best[cid] = (title, body, body_len)

    clauses: list[dict] = []
    for cid in sorted(best.keys(), key=_clause_sort_key):
        title, body, body_len = best[cid]
        if body_len < 30:
            continue
        if body_len > 3000:
            body = body[:3000] + "..."

        clauses.append({
            "clause_id": cid,
            "title": title,
            "text": body,
            "pointer": f"en_1993_1_1_2005#/{cid}",
            "keywords": _extract_keywords(cid, title, body),
        })

    return clauses


def _clause_sort_key(cid: str) -> tuple:
    parts = re.split(r"[.\s]+", cid)
    result = []
    for p in parts:
        try:
            result.append((0, int(p)))
        except ValueError:
            result.append((1, p))
    return tuple(result)


def _extract_keywords(clause_id: str, title: str, text: str) -> list[str]:
    kw: set[str] = set()
    kw.add(clause_id)
    for word in title.lower().split():
        if len(word) > 3 and word.isalpha():
            kw.add(word)

    eng_terms = [
        "bending", "shear", "axial", "compression", "tension", "buckling",
        "resistance", "moment", "force", "deflection", "classification",
        "yield", "ultimate", "weld", "bolt", "connection", "section",
        "imperfection", "stability", "lateral", "torsional", "cross-section",
        "elastic", "plastic", "modulus", "flange", "web", "slenderness",
        "serviceability", "fatigue", "fracture", "ductility",
    ]
    text_lower = text.lower()
    for term in eng_terms:
        if term in text_lower:
            kw.add(term)

    return sorted(kw)


def main():
    project_root = Path(__file__).resolve().parents[1]
    ocr_path = project_root / "data" / "ec3" / "en_1993_1_1_2005_ocr.json"
    out_path = project_root / "data" / "ec3" / "en_1993_1_1_2005_structured.json"

    print(f"Reading OCR: {ocr_path}")
    pages = load_ocr(ocr_path)
    print(f"  {len(pages)} pages loaded")

    clauses = extract_clauses(pages)
    print(f"  {len(clauses)} clauses extracted\n")

    output = {"clauses": clauses}
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written: {out_path}\n")

    for c in clauses:
        print(f"  {c['clause_id']:16s} | {c['title'][:65]}")


if __name__ == "__main__":
    main()
