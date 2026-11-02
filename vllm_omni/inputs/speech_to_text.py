"""Per-request speech-to-text state a model's generation prompt reads."""

from contextvars import ContextVar

SPEECH_TO_TEXT_INSTRUCTIONS: ContextVar[str | None] = ContextVar("speech_to_text_instructions", default=None)
"""The system instructions of the transcription being preprocessed; None when the request sets none."""
