# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A realtime conversation's prompt ids, joined from per-message pieces that are each tokenized once."""

from __future__ import annotations

import asyncio
import json
import random

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.realtime_conversation import Conversation, ConversationItem
from vllm_omni.entrypoints.openai.realtime_prompt import ChatPromptBuilder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AUDIO = "<|audio_start|><|audio_pad|><|audio_end|>"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def qwen_like(messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool) -> str:
    """Render the Qwen3-Omni chat template's rules that depend on a message's neighbours."""
    rest = list(messages)
    system = rest.pop(0)["content"] if rest and rest[0]["role"] == "system" else None
    out = ""
    if tools:
        out += f"<|im_start|>system\n{system or ''}\n\n# Tools {json.dumps(tools)}<|im_end|>\n"
    elif system is not None:
        out += f"<|im_start|>system\n{system}<|im_end|>\n"
    last_user = max((i for i, m in enumerate(rest) if m["role"] == "user"), default=len(rest) - 1)
    for i, m in enumerate(rest):
        if m["role"] == "user":
            out += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        elif m["role"] == "assistant":
            think = "<think>\n\n</think>\n\n" if i > last_user and i == len(rest) - 1 else ""
            calls = "".join(f"\n<tool_call>{json.dumps(c['function'])}</tool_call>" for c in m.get("tool_calls", []))
            out += f"<|im_start|>assistant\n{think}{m['content']}{calls}<|im_end|>\n"
        else:
            if i == 0 or rest[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += f"\n<tool_response>\n{m['content']}\n</tool_response>"
            if i == len(rest) - 1 or rest[i + 1]["role"] != "tool":
                out += "<|im_end|>\n"
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def numbered(messages: list[dict], tools: list[dict] | None, add_generation_prompt: bool) -> str:
    """Render each message with its position, which no cached piece can reproduce."""
    out = "".join(f"[{i}:{m['role']}]{m['content']}" for i, m in enumerate(messages))
    return out + ("[assistant]" if add_generation_prompt else "")


class _Encoder:
    """Tokenizes one id per character and records every text it was given."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def __call__(self, texts: list[str]) -> list[list[int]]:
        self.texts += texts
        return [[ord(c) for c in text] for text in texts]


def _decode(ids: list[int]) -> str:
    return "".join(map(chr, ids))


def _user_audio(item_id: str, text: str = "") -> ConversationItem:
    return ConversationItem(item_id, "message", "user", text=text, audio=np.zeros(16_000, dtype=np.float32))


def _user(item_id: str, text: str) -> ConversationItem:
    return ConversationItem(item_id, "message", "user", text=text)


def _reply(item_id: str, text: str, token_ids: list[int] | None = None) -> ConversationItem:
    return ConversationItem(item_id, "message", "assistant", text=text, response_id=item_id, token_ids=token_ids)


def _call(item_id: str, response_id: str) -> ConversationItem:
    return ConversationItem(
        item_id, "function_call", "assistant", call_id=f"call_{item_id}", name="get_weather", response_id=response_id
    )


def _output(item_id: str, text: str) -> ConversationItem:
    return ConversationItem(item_id, "function_call_output", "tool", text=text, call_id=f"call_{item_id}")


def _build(builder: ChatPromptBuilder, conversation: Conversation, instructions=None, tools=None):
    return asyncio.run(builder.build(conversation, instructions, tools))


def _full(render, conversation: Conversation, instructions=None, tools=None) -> str:
    return render(conversation.render(AUDIO, instructions).messages, tools, True)


CONVERSATIONS = {
    "audio turns": [_user_audio("u1"), _reply("a1", "Hi."), _user_audio("u2", " and text")],
    "tool round": [
        _user("u1", "Weather?"),
        _reply("r1", "Checking."),
        _call("c1", "r1"),
        _call("c2", "r1"),
        _output("o1", "sunny"),
        _output("o2", "warm"),
        _reply("a2", "Sunny and warm."),
        _user("u2", "\nThanks "),
    ],
    "ends with an assistant message": [_user("u1", "Hi"), _reply("a1", "Hello."), _reply("a2", "Anything else?")],
}


@pytest.mark.parametrize("tools", [None, TOOLS])
@pytest.mark.parametrize("instructions", [None, "Be brief."])
@pytest.mark.parametrize("name", list(CONVERSATIONS))
def test_the_joined_prompt_is_the_full_render(name: str, instructions: str | None, tools: list | None) -> None:
    conversation = Conversation(list(CONVERSATIONS[name]))

    prompt = _build(ChatPromptBuilder(qwen_like, _Encoder(), AUDIO), conversation, instructions, tools)

    assert _decode(prompt.token_ids) == _full(qwen_like, conversation, instructions, tools)


def test_each_message_is_tokenized_once_across_turns() -> None:
    encoder = _Encoder()
    builder = ChatPromptBuilder(qwen_like, encoder, AUDIO)
    conversation = Conversation([_user_audio("u1"), _reply("a1", "Hi."), _user("u2", "Where is the train?")])
    _build(builder, conversation)

    conversation.add(_reply("a2", "Platform two."))
    conversation.add(_user("u3", "Thanks."))
    encoder.texts.clear()
    _build(builder, conversation)

    assert encoder.texts == [
        "<|im_start|>assistant\nPlatform two.<|im_end|>\n",
        "<|im_start|>user\nThanks.<|im_end|>\n",
    ]


def test_a_truncated_reply_is_tokenized_again() -> None:
    encoder = _Encoder()
    builder = ChatPromptBuilder(qwen_like, encoder, AUDIO)
    reply = _reply("a1", "The train leaves at six.")
    reply.audio_out_seconds = 2.0
    conversation = Conversation([_user("u1", "When?"), reply, _user("u2", "Sorry?")])
    _build(builder, conversation)

    conversation.truncate("a1", 1000)
    encoder.texts.clear()
    prompt = _build(builder, conversation)

    assert encoder.texts == ["<|im_start|>assistant\nThe train<|im_end|>\n"]
    assert _decode(prompt.token_ids) == _full(qwen_like, conversation)


def test_a_reply_keeps_the_ids_the_model_generated() -> None:
    generated = [ord(c) for c in "\n\nSure."] + [ord("<")]
    conversation = Conversation([_user("u1", "Hi"), _reply("a1", "Sure.", token_ids=generated), _user("u2", "Bye")])

    prompt = _build(ChatPromptBuilder(qwen_like, _Encoder(), AUDIO), conversation)

    assert "<|im_start|>assistant\n\n\nSure.<|im_end|>\n<|im_start|>user\nBye" in _decode(prompt.token_ids)


def test_a_template_that_numbers_messages_gets_a_full_render() -> None:
    conversation = Conversation([_user("u1", "Hi"), _reply("a1", "Hello."), _user("u2", "Bye")])

    prompt = _build(ChatPromptBuilder(numbered, _Encoder(), AUDIO), conversation, "Be brief.")

    assert _decode(prompt.token_ids) == _full(numbered, conversation, "Be brief.")
    assert set(prompt.item_tokens) == {"u1", "a1", "u2"}


@pytest.mark.parametrize("render", [qwen_like, numbered])
def test_the_item_counts_and_overhead_add_up_to_the_expanded_prompt(render) -> None:
    conversation = Conversation(list(CONVERSATIONS["tool round"]) + [_user_audio("u3")])

    prompt = _build(ChatPromptBuilder(render, _Encoder(), AUDIO), conversation, "Be brief.", TOOLS)

    audio_expansion = conversation.get("u3").estimated_audio_tokens() - 1
    assert prompt.overhead_tokens + sum(prompt.item_tokens.values()) == len(prompt.token_ids) + audio_expansion
    assert prompt.item_tokens["c1"] == prompt.item_tokens["o2"] == 0


def _qwen3_omni_tokenizer_and_template():
    transformers = pytest.importorskip("transformers")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    repo = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True)
        template_file = huggingface_hub.hf_hub_download(repo, "chat_template.json", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3-Omni tokenizer is not in the local Hugging Face cache")
    with open(template_file) as f:
        return tokenizer, json.load(f)["chat_template"]


def test_qwen3_omni_prompts_join_exactly() -> None:
    tokenizer, template = _qwen3_omni_tokenizer_and_template()

    def render(messages, tools, add_generation_prompt):
        return tokenizer.apply_chat_template(
            messages, tools=tools, chat_template=template, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    async def encode(texts):
        return [tokenizer.encode(text, add_special_tokens=False) for text in texts]

    rng = random.Random(0)
    snippets = [
        "Hello.",
        " lead",
        "trail ",
        "line\nbreak",
        "\n\nnewlines",
        "12345",
        "🙂🚆",
        "日本語",
        "façade",
        "OK",
        "",
    ]

    def text() -> str:
        return "".join(rng.choice(snippets) for _ in range(rng.randint(1, 3)))

    for n in range(200):
        items: list[ConversationItem] = []
        for turn in range(rng.randint(1, 6)):
            items.append(_user_audio(f"u{turn}", text()) if rng.random() < 0.5 else _user(f"u{turn}", text()))
            if rng.random() < 0.3:
                items += [_call(f"c{turn}", f"r{turn}"), _output(f"o{turn}", text()), _output(f"p{turn}", text())]
            items.append(_reply(f"a{turn}", text()))
        if rng.random() < 0.5:
            items.pop()
        conversation = Conversation(items)
        instructions = text() if n % 3 else None
        tools = TOOLS if n % 2 else None

        prompt = asyncio.run(ChatPromptBuilder(render, encode, AUDIO).build(conversation, instructions, tools))

        full = tokenizer.encode(_full(render, conversation, instructions, tools), add_special_tokens=False)
        assert prompt.token_ids == full, n
