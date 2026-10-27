# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A realtime session's conversation: rendering, item edits, and truncation."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.realtime_conversation import (
    SPEAKING_CHARS_PER_SECOND,
    Conversation,
    ConversationItem,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AUDIO = "<audio>"


def _user_audio(item_id: str, seconds: float = 1.0) -> ConversationItem:
    return ConversationItem(item_id, "message", "user", audio=np.zeros(int(seconds * 16_000), dtype=np.float32))


def _assistant(item_id: str, text: str, response_id: str | None = None, audio_out: float = 0.0) -> ConversationItem:
    return ConversationItem(
        item_id, "message", "assistant", text=text, response_id=response_id or item_id, audio_out_seconds=audio_out
    )


def _call(item_id: str, call_id: str, response_id: str) -> ConversationItem:
    return ConversationItem(
        item_id,
        "function_call",
        "assistant",
        call_id=call_id,
        name="get_weather",
        arguments='{"city": "Austin"}',
        response_id=response_id,
    )


class TestRender:
    def test_turns_render_in_order_with_audio_placeholders(self) -> None:
        conversation = Conversation([_user_audio("u1"), _assistant("a1", "Hi."), _user_audio("u2")])

        rendered = conversation.render(AUDIO, instructions="Be brief.")

        assert rendered.messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": AUDIO},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": AUDIO},
        ]
        assert rendered.audio_item_ids == ["u1", "u2"]
        assert len(rendered.audio) == 2
        assert [[item.item_id for item in unit.items] for unit in rendered.units] == [["u1"], ["a1"], ["u2"]]

    def test_a_response_with_text_and_a_tool_call_is_one_assistant_message(self) -> None:
        conversation = Conversation(
            [
                _user_audio("u1"),
                _assistant("a1", "Let me check.", response_id="r1"),
                _call("c1", "call_1", "r1"),
                ConversationItem("o1", "function_call_output", "tool", text="sunny", call_id="call_1"),
                _assistant("a2", "It is sunny.", response_id="r1"),
            ]
        )

        messages = conversation.render(AUDIO).messages

        assert messages[1] == {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Austin"}}}],
        }
        assert messages[2] == {"role": "tool", "content": "sunny"}
        assert messages[3] == {"role": "assistant", "content": "It is sunny."}

    def test_a_run_of_tool_results_is_one_unit(self) -> None:
        conversation = Conversation(
            [
                _call("c1", "call_1", "r1"),
                _call("c2", "call_2", "r1"),
                ConversationItem("o1", "function_call_output", "tool", text="sunny", call_id="call_1"),
                ConversationItem("o2", "function_call_output", "tool", text="warm", call_id="call_2"),
            ]
        )

        units = conversation.render(AUDIO).units

        assert [[item.item_id for item in unit.items] for unit in units] == [["c1", "c2"], ["o1", "o2"]]
        assert [len(unit.messages) for unit in units] == [1, 2]

    def test_text_user_messages_carry_no_audio(self) -> None:
        rendered = Conversation([ConversationItem("t1", "message", "user", text="hello")]).render(AUDIO)

        assert rendered.messages == [{"role": "user", "content": "hello"}]
        assert rendered.audio == []


class TestEdits:
    def test_an_item_is_inserted_after_the_named_item(self) -> None:
        conversation = Conversation([_user_audio("u1"), _user_audio("u2")])

        conversation.add(_assistant("a1", "Hi."), previous_item_id="u1")

        assert [item.item_id for item in conversation.items] == ["u1", "a1", "u2"]

    @pytest.mark.parametrize("item_id, previous", [("u1", None), ("x", "missing")])
    def test_a_taken_id_or_unknown_anchor_is_refused(self, item_id: str, previous: str | None) -> None:
        conversation = Conversation([_user_audio("u1")])

        with pytest.raises(KeyError):
            conversation.add(_user_audio(item_id), previous_item_id=previous)

    def test_delete_reports_whether_the_item_existed(self) -> None:
        conversation = Conversation([_user_audio("u1")])

        assert conversation.delete("u1") is True
        assert conversation.delete("u1") is False


class TestTruncate:
    def test_the_text_is_cut_to_the_share_of_audio_played_at_a_word_boundary(self) -> None:
        item = _assistant("a1", "The train to Chicago leaves at six.", audio_out=4.0)
        conversation = Conversation([item])

        assert conversation.truncate("a1", 2000) is True

        assert item.text == "The train to"
        assert item.audio_out_seconds == 2.0

    def test_cutting_the_text_drops_the_generated_ids_and_bumps_the_revision(self) -> None:
        item = _assistant("a1", "The train to Chicago leaves at six.", audio_out=4.0)
        item.token_ids = [1, 2, 3]

        Conversation([item]).truncate("a1", 2000)

        assert (item.token_ids, item.revision) == (None, 1)

    def test_text_without_spaces_keeps_the_estimated_character_cut(self) -> None:
        item = _assistant("a1", "去北京的火车将在下午六点出发。", audio_out=4.0)

        Conversation([item]).truncate("a1", 2000)

        assert item.text == "去北京的火车将"

    def test_playing_past_the_end_keeps_the_whole_reply(self) -> None:
        item = _assistant("a1", "Hello there.", audio_out=1.0)

        item.token_ids = [1, 2]

        assert Conversation([item]).truncate("a1", 5000) is True
        assert item.text == "Hello there."
        assert (item.token_ids, item.revision) == ([1, 2], 0)

    def test_nothing_played_empties_the_reply(self) -> None:
        item = _assistant("a1", "Hello there.", audio_out=1.0)

        Conversation([item]).truncate("a1", 0)

        assert item.text == ""

    def test_a_cancelled_reply_is_cut_at_the_speaking_rate(self) -> None:
        item = _assistant("a1", "The train to Chicago leaves at six and arrives at nine.", audio_out=4.0)
        item.audio_complete = False

        Conversation([item]).truncate("a1", 1000)

        assert item.text == "The train"
        assert (item.audio_out_seconds, item.audio_complete) == (1.0, True)

    def test_the_speaking_rate_is_measured_on_the_conversations_complete_replies(self) -> None:
        heard = _assistant("a0", "x" * 20, audio_out=1.0)
        item = _assistant("a1", "abcde fghij klmno pqrst", audio_out=2.0)
        item.audio_complete = False
        conversation = Conversation([heard, item])

        assert conversation.speaking_rate() == 20.0
        conversation.truncate("a1", 600)

        assert item.text == "abcde fghij"

    def test_a_short_reply_does_not_set_the_speaking_rate(self) -> None:
        ok = _assistant("a0", "OK.", audio_out=1.26)
        item = _assistant("a1", "The train to Chicago leaves at six and arrives at nine.", audio_out=4.0)
        item.audio_complete = False
        conversation = Conversation([ok, item])

        assert conversation.speaking_rate() == SPEAKING_CHARS_PER_SECOND

    def test_a_cancelled_reply_with_no_audio_sent_is_emptied(self) -> None:
        item = _assistant("a1", "Hello there.")
        item.audio_complete = False

        Conversation([item]).truncate("a1", 500)

        assert item.text == ""

    def test_only_assistant_messages_truncate(self) -> None:
        assert Conversation([_user_audio("u1")]).truncate("u1", 100) is False
        assert Conversation([]).truncate("a1", 100) is False
