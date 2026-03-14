from __future__ import annotations

import base64
import binascii
import mimetypes
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.config import Settings

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[^;,]+)?(?P<params>(?:;[^;,=]+=[^;,=]+)*)(?P<base64>;base64)?,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".css",
    ".sql",
    ".tex",
}
_TEXT_MIME_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
)
_TEXTUTIL_EXTENSIONS = {
    ".doc",
    ".docx",
    ".rtf",
    ".rtfd",
    ".odt",
    ".html",
    ".htm",
    ".webarchive",
}
_MAX_DOC_CHARS = 18_000
_MAX_TOTAL_DOC_CHARS = 40_000


@dataclass
class SavedAttachment:
    name: str
    media_type: str
    size: int
    is_image: bool
    path: str
    public_url: str = ""
    extracted_text: str = ""
    extraction_note: str = ""


@dataclass
class AttachmentBundle:
    user_content: str | list[dict[str, Any]]
    routing_text: str
    saved_attachments: list[SavedAttachment] = field(default_factory=list)


def saved_attachments_payload(saved_attachments: list[SavedAttachment]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "type": item.media_type,
            "size": item.size,
            "is_image": item.is_image,
            "storage_path": item.path,
            "storage_url": item.public_url,
            "extracted_text": item.extracted_text,
            "extraction_note": item.extraction_note,
        }
        for item in saved_attachments
    ]


def process_attachments(
    settings: Settings,
    attachments: list[Any] | None,
    *,
    message_text: str,
    request_id: str,
) -> AttachmentBundle:
    items = list(attachments or [])
    if not items:
        return AttachmentBundle(user_content=message_text, routing_text=message_text)

    upload_dir = settings.resolved_uploads_path / request_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_char_budget = _MAX_TOTAL_DOC_CHARS
    image_parts: list[dict[str, Any]] = []
    text_blocks: list[str] = []
    summary_lines: list[str] = []
    saved: list[SavedAttachment] = []

    for index, attachment in enumerate(items, start=1):
        raw_name = str(_attachment_value(attachment, "name", "") or f"attachment-{index}")
        name = _safe_filename(raw_name) or f"attachment-{index}"
        media_type = str(_attachment_value(attachment, "type", "") or "")
        size = int(_attachment_value(attachment, "size", 0) or 0)
        is_image = bool(_attachment_value(attachment, "is_image", False))
        data_url = _attachment_value(attachment, "data_url", None)
        stored_extracted_text = str(_attachment_value(attachment, "extracted_text", "") or "").strip()
        stored_extraction_note = str(_attachment_value(attachment, "extraction_note", "") or "").strip()
        stored_path_hint = str(_attachment_value(attachment, "storage_path", "") or "").strip()

        data_bytes = b""
        inferred_mime = media_type
        if isinstance(data_url, str) and data_url.strip():
            data_bytes, inferred_mime = _decode_data_url(data_url)
            if inferred_mime and not media_type:
                media_type = inferred_mime

        suffix = Path(name).suffix
        if not suffix:
            guessed_ext = mimetypes.guess_extension(media_type or "") or ""
            suffix = guessed_ext if guessed_ext not in {".jpe"} else ".jpg"
        stored_name = f"{index:02d}_{uuid4().hex[:8]}_{Path(name).stem}{suffix}"
        stored_path: Path | None = None
        if data_bytes:
            stored_path = upload_dir / stored_name
            stored_path.write_bytes(data_bytes)
        elif stored_path_hint:
            stored_path = _resolve_saved_attachment_path(settings, stored_path_hint)

        extracted_text = ""
        extraction_note = ""
        persisted_excerpt = ""
        relative_storage_path = _relative_storage_path(settings, stored_path) if stored_path is not None else ""
        public_url = _storage_public_url(relative_storage_path)
        if is_image and isinstance(data_url, str) and data_url.strip():
            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
            summary_lines.append(f"- Image attachment: {raw_name}")
        else:
            if data_bytes:
                extracted_text, extraction_note = _extract_document_text(
                    stored_path or (upload_dir / stored_name),
                    media_type=media_type,
                    data_bytes=data_bytes,
                )
            elif stored_extracted_text:
                extracted_text = _normalize_extracted_text(stored_extracted_text)
                extraction_note = stored_extraction_note or "reused stored extraction"
            elif stored_path is not None and stored_path.exists():
                extracted_bytes = stored_path.read_bytes()
                extracted_text, extraction_note = _extract_document_text(
                    stored_path,
                    media_type=media_type,
                    data_bytes=extracted_bytes,
                )

            if extracted_text and doc_char_budget > 0:
                clipped = extracted_text[: min(len(extracted_text), doc_char_budget, _MAX_DOC_CHARS)].strip()
                if clipped:
                    persisted_excerpt = clipped
                    text_blocks.append(
                        "\n".join(
                            [
                                f"<attached-document name=\"{raw_name}\" type=\"{media_type or 'application/octet-stream'}\">",
                                clipped,
                                "</attached-document>",
                            ]
                        )
                    )
                    doc_char_budget -= len(clipped)
                    summary_lines.append(f"- Document attachment: {raw_name}")
                else:
                    summary_lines.append(f"- Document attachment: {raw_name} (empty after extraction)")
            elif extracted_text:
                summary_lines.append(f"- Document attachment: {raw_name} (stored excerpt omitted due prompt budget)")
            else:
                extraction_note = extraction_note or stored_extraction_note or "binary content not provided by the client"
                summary_lines.append(f"- Document attachment: {raw_name} ({extraction_note})")

        saved.append(
            SavedAttachment(
                name=raw_name,
                media_type=media_type or inferred_mime or "",
                size=size,
                is_image=is_image,
                path=relative_storage_path,
                public_url=public_url,
                extracted_text=persisted_excerpt,
                extraction_note=extraction_note,
            )
        )

    text_section = _build_attachment_text_section(summary_lines, text_blocks)
    routing_text = message_text
    if text_section:
        routing_text = f"{message_text}\n\n{text_section}".strip() if message_text else text_section

    if image_parts:
        image_instruction = (
            "One or more images are attached below and are available for direct inspection. "
            "Inspect the image content first, decide whether it depicts a civil/structural "
            "engineering subject, and then answer using the current scope gate. "
            "Do not claim that you cannot view the image unless this message explicitly says "
            "attachment processing failed."
        )
        prompt_text = (
            f"{image_instruction}\n\n{routing_text}".strip()
            if routing_text
            else image_instruction
        )
        user_content: str | list[dict[str, Any]] = [{"type": "text", "text": prompt_text}, *image_parts]
    else:
        user_content = routing_text

    return AttachmentBundle(
        user_content=user_content,
        routing_text=routing_text,
        saved_attachments=saved,
    )


def _build_attachment_text_section(summary_lines: list[str], text_blocks: list[str]) -> str:
    if not summary_lines and not text_blocks:
        return ""
    lines = ["Technical attachments supplied by the user:"]
    lines.extend(summary_lines)
    if text_blocks:
        lines.append("")
        lines.append("Lightweight extracted document text:")
        lines.append("")
        lines.extend(text_blocks)
    return "\n".join(lines).strip()


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name.strip())
    return cleaned[:120].strip("._")


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    match = _DATA_URL_RE.match(data_url.strip())
    if not match:
        raise ValueError("invalid data URL")
    mime = (match.group("mime") or "").strip()
    raw = match.group("data") or ""
    if match.group("base64"):
        try:
            return base64.b64decode(raw, validate=True), mime
        except binascii.Error as exc:
            raise ValueError("invalid base64 payload") from exc
    return raw.encode("utf-8"), mime


def _attachment_value(attachment: Any, key: str, default: Any = None) -> Any:
    if isinstance(attachment, dict):
        return attachment.get(key, default)
    return getattr(attachment, key, default)


def _extract_document_text(path: Path, *, media_type: str, data_bytes: bytes) -> tuple[str, str]:
    suffix = path.suffix.lower()

    if suffix == ".pdf" or media_type == "application/pdf":
        text = _run_subprocess_text(["pdftotext", "-layout", "-nopgbrk", str(path), "-"])
        if text:
            return _normalize_extracted_text(text), ""
        return "", "PDF extraction unavailable"

    if suffix in _TEXT_EXTENSIONS or any(media_type.startswith(prefix) for prefix in _TEXT_MIME_PREFIXES):
        return _normalize_extracted_text(_decode_text_bytes(data_bytes)), ""

    if suffix in _TEXTUTIL_EXTENSIONS:
        text = _run_subprocess_text(["textutil", "-convert", "txt", "-stdout", str(path)])
        if text:
            return _normalize_extracted_text(text), ""
        return "", "document extraction unavailable"

    return "", "unsupported document type for lightweight extraction"


def _run_subprocess_text(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout or ""


def _decode_text_bytes(data_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data_bytes.decode("utf-8", errors="replace")


def _normalize_extracted_text(text: str) -> str:
    compact = text.replace("\x00", " ")
    compact = re.sub(r"\r\n?", "\n", compact)
    compact = re.sub(r"[ \t]+\n", "\n", compact)
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    compact = compact.strip()
    if len(compact) > _MAX_DOC_CHARS:
        compact = compact[:_MAX_DOC_CHARS].rstrip() + "\n\n[Document text truncated]"
    return compact


def _resolve_saved_attachment_path(settings: Settings, raw_path: str) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    try:
        resolved = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (settings.project_root / candidate).resolve(strict=False)
        )
        uploads_root = settings.resolved_uploads_path.resolve(strict=False)
        resolved.relative_to(uploads_root)
    except Exception:
        return None
    return resolved


def _relative_storage_path(settings: Settings, path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(settings.project_root.resolve(strict=False)))
    except Exception:
        return str(path)


def _storage_public_url(relative_path: str) -> str:
    if not relative_path:
        return ""
    normalized = relative_path.replace("\\", "/").lstrip("./")
    if not normalized.startswith("data/"):
        return ""
    return "/" + normalized
