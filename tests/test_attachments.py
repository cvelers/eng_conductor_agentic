from __future__ import annotations

import base64

from backend.agent.loop import _build_conversation_history_for_validator
from backend.attachments import process_attachments, saved_attachments_payload
from backend.config import Settings
from backend.schemas import Attachment


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def test_process_attachments_extracts_document_text_and_builds_multimodal_content(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    bundle = process_attachments(
        settings,
        [
            {
                "name": "frame-note.txt",
                "type": "text/plain",
                "size": 24,
                "is_image": False,
                "data_url": _data_url("text/plain", b"Portal frame note: S355 beam"),
            },
            {
                "name": "detail.png",
                "type": "image/png",
                "size": 16,
                "is_image": True,
                "data_url": _data_url("image/png", b"\x89PNG\r\n\x1a\nfake"),
            },
        ],
        message_text="Check the attached detail.",
        request_id="req_1",
    )

    assert isinstance(bundle.user_content, list)
    assert bundle.user_content[0]["type"] == "text"
    assert "available for direct inspection" in bundle.user_content[0]["text"]
    assert "Portal frame note: S355 beam" in bundle.user_content[0]["text"]
    assert bundle.user_content[1]["type"] == "image_url"
    assert "detail.png" in bundle.routing_text
    assert len(bundle.saved_attachments) == 2
    assert all(tmp_path.joinpath(att.path).exists() for att in bundle.saved_attachments)


def test_process_attachments_returns_plain_text_when_only_documents_are_present(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    bundle = process_attachments(
        settings,
        [
            {
                "name": "calc.csv",
                "type": "text/csv",
                "size": 20,
                "is_image": False,
                "data_url": _data_url("text/csv", b"member,utilization\nB1,0.82\n"),
            },
        ],
        message_text="Review this schedule.",
        request_id="req_2",
    )

    assert isinstance(bundle.user_content, str)
    assert "member,utilization" in bundle.user_content
    assert "calc.csv" in bundle.routing_text


def test_process_attachments_accepts_pydantic_attachment_models(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    bundle = process_attachments(
        settings,
        [
            Attachment(
                name="detail.png",
                type="image/png",
                size=16,
                is_image=True,
                data_url=_data_url("image/png", b"\x89PNG\r\n\x1a\nfake"),
            )
        ],
        message_text="What's in the attached image?",
        request_id="req_pydantic",
    )

    assert isinstance(bundle.user_content, list)
    assert bundle.user_content[0]["type"] == "text"
    assert bundle.user_content[1]["type"] == "image_url"
    assert bundle.saved_attachments[0].is_image is True


def test_process_attachments_reuses_stored_text_without_binary_payload(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    bundle = process_attachments(
        settings,
        [
            {
                "name": "calc-note.pdf",
                "type": "application/pdf",
                "size": 32,
                "is_image": False,
                "storage_path": "data/uploads/existing/calc-note.pdf",
                "extracted_text": "Stored attachment text: portal frame utilization is 0.82.",
                "extraction_note": "reused stored extraction",
            },
        ],
        message_text="Continue from the prior document.",
        request_id="req_3",
    )

    assert isinstance(bundle.user_content, str)
    assert "portal frame utilization is 0.82" in bundle.user_content
    assert bundle.saved_attachments[0].path == "data/uploads/existing/calc-note.pdf"
    assert bundle.saved_attachments[0].extracted_text.startswith("Stored attachment text")


def test_saved_attachments_payload_is_compact_and_client_safe(tmp_path) -> None:
    settings = Settings(project_root=tmp_path)
    bundle = process_attachments(
        settings,
        [
            {
                "name": "schedule.csv",
                "type": "text/csv",
                "size": 20,
                "is_image": False,
                "data_url": _data_url("text/csv", b"member,utilization\nB1,0.82\n"),
            },
        ],
        message_text="Review this schedule.",
        request_id="req_4",
    )

    payload = saved_attachments_payload(bundle.saved_attachments)

    assert payload == [
        {
            "name": "schedule.csv",
            "type": "text/csv",
            "size": 20,
            "is_image": False,
            "storage_path": bundle.saved_attachments[0].path,
            "storage_url": bundle.saved_attachments[0].public_url,
            "extracted_text": bundle.saved_attachments[0].extracted_text,
            "extraction_note": "",
        }
    ]


def test_validator_history_flattens_multimodal_user_messages() -> None:
    history = _build_conversation_history_for_validator(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check the attached frame photo."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abcd"}},
                ],
            },
            {"role": "assistant", "content": "I need member dimensions."},
        ]
    )

    assert "[USER] Check the attached frame photo." in history
    assert "[Image attachment]" in history
