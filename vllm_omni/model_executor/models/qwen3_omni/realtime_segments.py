# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Segmentation helpers for Qwen3-Omni realtime transcription sessions."""

from __future__ import annotations

import unicodedata

import numpy as np

_UNSPACED_SCRIPT_RANGES = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x2E80, 0x30FF),  # CJK radicals, CJK symbols and punctuation, Hiragana, Katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF00, 0xFFEF),  # halfwidth and fullwidth forms
    (0x20000, 0x2FA1F),  # CJK extensions B onward
)

_NO_SPACE_BEFORE = frozenset(".,;:!?)]}%")

_CLOSING_PUNCTUATION_CATEGORIES = frozenset({"Pe", "Pf"})


def _is_unspaced(char: str) -> bool:
    codepoint = ord(char)
    return any(low <= codepoint <= high for low, high in _UNSPACED_SCRIPT_RANGES)


def _closes(char: str) -> bool:
    return char in _NO_SPACE_BEFORE or unicodedata.category(char) in _CLOSING_PUNCTUATION_CATEGORIES


def segment_separator(previous: str, following: str) -> str:
    """Return the text to insert between two consecutive segment transcripts.

    Args:
        previous: Transcript emitted so far.
        following: Start of the next segment's transcript.

    Returns:
        A single space when both sides are words of a space-delimited script, else the empty string.
    """
    if not previous or not following:
        return ""
    if previous[-1].isspace() or following[0].isspace() or _closes(following[0]):
        return ""
    if _is_unspaced(previous[-1]) or _is_unspaced(following[0]):
        return ""
    return " "


def continuation_prompt(audio_placeholder: str) -> str:
    """Return the prompt text that appends one more audio segment to a transcription session.

    The session's previous segment ends without its ``<|im_end|>``: the engine drops the last sampled
    token of each segment, which is the stop token. The continuation therefore closes that assistant
    turn before opening the next user turn, and repeats no system message.

    Args:
        audio_placeholder: The model's placeholder string for one audio item.
    """
    return f"<|im_end|>\n<|im_start|>user\n{audio_placeholder}<|im_end|>\n<|im_start|>assistant\n"


class SilenceAlignedBuffer:
    """Audio buffer that releases segments cut at the quietest point before a target length."""

    def __init__(
        self,
        sampling_rate: int,
        segment_duration_s: float,
        search_duration_s: float,
        energy_window_samples: int,
    ) -> None:
        """Create an empty buffer.

        Args:
            sampling_rate: Sample rate of the written audio, in Hz.
            segment_duration_s: Longest segment released before the final flush, in seconds.
            search_duration_s: Span before the segment limit searched for the quietest cut, in seconds;
                must be shorter than ``segment_duration_s``.
            energy_window_samples: Width of each energy measurement window, in samples.

        Raises:
            ValueError: If the search span is not shorter than the segment, or the window is not positive.
        """
        self._segment_size = int(segment_duration_s * sampling_rate)
        self._search_size = int(search_duration_s * sampling_rate)
        if not 0 <= self._search_size < self._segment_size:
            raise ValueError("search_duration_s must be non-negative and shorter than segment_duration_s")
        if energy_window_samples <= 0:
            raise ValueError("energy_window_samples must be positive")
        self._window = energy_window_samples
        self._buffer = np.empty(0, dtype=np.float32)

    def write_audio(self, audio: np.ndarray) -> None:
        """Append mono float audio to the buffer."""
        self._buffer = np.concatenate([self._buffer, audio.astype(np.float32, copy=False)])

    def read_audio(self) -> np.ndarray | None:
        """Release the next segment once enough audio is buffered.

        Returns:
            The audio up to the cut point, or None while the buffer holds less than one segment.
        """
        if len(self._buffer) < self._segment_size:
            return None
        cut = self._quietest_cut(self._segment_size - self._search_size, self._segment_size)
        segment, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return segment

    def flush(self) -> np.ndarray | None:
        """Release all remaining audio, or None when the buffer is empty."""
        if len(self._buffer) == 0:
            return None
        remaining, self._buffer = self._buffer, np.empty(0, dtype=np.float32)
        return remaining

    def _quietest_cut(self, start: int, end: int) -> int:
        windows = (end - start) // self._window
        if windows == 0:
            return end
        frames = self._buffer[start : start + windows * self._window].reshape(windows, self._window)
        quietest = int(np.argmin(np.mean(frames * frames, axis=1)))
        return start + quietest * self._window + self._window // 2
