# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SPEV prompt construction, media preparation and decision interpretation."""

import io
import json
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pybase64 as base64
from scipy.signal import resample_poly
from transformers import Qwen3OmniMoeProcessor
from vllm import PoolingParams
from vllm.multimodal.media import ImageMediaIO, MediaWithBytes
from vllm.multimodal.media.audio import load_audio
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor

from vllm_omni.inputs.decisions import (
    DecisionAnswer,
    DecisionChoiceAnswer,
    DecisionChoiceQuestion,
    DecisionInputImage,
    DecisionInputText,
    DecisionPredicateAnswer,
    DecisionPredicateQuestion,
    DecisionScoreAnswer,
    DecisionsRequest,
)
from vllm_omni.model_executor.models.spev.config import SPEVConfig


def register_io_processor() -> str:
    return "vllm_omni.model_executor.models.spev.processor.SPEVProcessor"


def literal(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def candidates(question) -> list[object]:
    if isinstance(question, DecisionChoiceQuestion):
        return [
            {"value": c.value, "description": c.description}
            if isinstance(c.value, bool)
            else c.value
            if c.description is None
            else c.description
            for c in question.choices
        ]
    if isinstance(question, DecisionPredicateQuestion):
        return ["The assertion is true.", "The assertion is false."]
    return [level.model_dump(exclude_none=True) for level in question.levels]


@dataclass
class SPEVInput:
    request: DecisionsRequest
    state: object
    image_bytes: bytes | None
    video_bytes: bytes | None = None
    decision_layout: dict | None = None


class SPEVProcessor(IOProcessor[SPEVInput, list[DecisionAnswer] | list[float]]):
    supports_decisions = True

    def __init__(self, vllm_config, renderer):
        super().__init__(vllm_config, renderer)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            vllm_config.model_config.tokenizer,
            revision=vllm_config.model_config.tokenizer_revision,
            local_files_only=True,
            trust_remote_code=vllm_config.model_config.trust_remote_code,
        )
        self.config = SPEVConfig.from_hf(vllm_config.model_config.hf_config, self.processor.tokenizer)
        processor_config, _ = Qwen3OmniMoeProcessor.get_processor_dict(
            vllm_config.model_config.tokenizer,
            revision=vllm_config.model_config.tokenizer_revision,
            local_files_only=True,
        )
        self.config.video_fps = processor_config.get("video_fps", self.config.video_fps)
        self.lock = threading.Lock()
        marker_ids = self.processor.tokenizer.encode(self.config.marker, add_special_tokens=False)
        if marker_ids != self.config.marker_token_ids:
            raise ValueError("SPEV marker IDs do not match the checkpoint tokenizer")

    def parse_data(self, data: object) -> SPEVInput:
        request = data if isinstance(data, DecisionsRequest) else DecisionsRequest.model_validate(data)
        if len(request.questions) > self.config.max_questions:
            raise ValueError("Too many questions for this model")
        if sum(len(candidates(q)) for q in request.questions) > self.config.max_candidates:
            raise ValueError("Too many candidates for this model")
        image = None
        state = request.input
        if isinstance(state, list):
            state = []
            for message in request.input:
                if isinstance(message.content, str):
                    state.append({"role": "user", "content": message.content})
                    continue
                content = []
                for part in message.content:
                    if isinstance(part, DecisionInputText):
                        content.append(part.model_dump())
                    elif isinstance(part, DecisionInputImage):
                        if image is not None or part.detail not in (None, "auto"):
                            raise ValueError("SPEV accepts one image with automatic detail")
                        header, sep, encoded = part.image_url.partition(",")
                        if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
                            raise ValueError("SPEV requires an inline image")
                        image = self.decode_bytes(encoded)
                        content.append({"type": "input_image"})
                state.append({"role": "user", "content": content})
        video = self.decode_bytes(request.input_video.data) if request.input_video is not None else None
        parsed = SPEVInput(request=request, state=state, image_bytes=image, video_bytes=video)
        if len(self.render(parsed)) > self.config.max_text_chars:
            raise ValueError("SPEV input exceeds the text limit")
        return parsed

    def decode_bytes(self, encoded: str) -> bytes:
        if len(encoded) > 4 * ((self.config.max_media_bytes + 2) // 3):
            raise ValueError("Media exceeds the checkpoint byte limit")
        data = base64.b64decode(encoded, validate=True)
        if not data or len(data) > self.config.max_media_bytes:
            raise ValueError("Invalid media length")
        return data

    def decode_audio(self, encoded: str) -> np.ndarray:
        data = self.decode_bytes(encoded)
        seconds = self.config.max_audio_seconds
        wave, rate = load_audio(
            io.BytesIO(data),
            sr=None,
            mono=True,
            max_duration_s=seconds,
            max_decode_bytes=int(seconds * 192000 * 8 * np.dtype(np.float32).itemsize),
        )
        if wave.size == 0 or not 8000 <= rate <= 192000 or wave.size / rate > seconds:
            raise ValueError("Invalid audio duration or sample rate")
        if not np.isfinite(wave).all():
            raise ValueError("Nonfinite audio")
        if rate != self.config.sample_rate:
            common = math.gcd(rate, self.config.sample_rate)
            wave = resample_poly(wave, self.config.sample_rate // common, rate // common).astype(np.float32)
        return wave

    def decode_video(self, data: bytes):
        from tempfile import NamedTemporaryFile

        import av
        from transformers.video_utils import load_video

        if len(data) > self.config.max_media_bytes:
            raise ValueError("Video exceeds the checkpoint byte limit")
        try:
            with av.open(io.BytesIO(data)) as container:
                if not container.streams.video:
                    raise ValueError("Video has no visual stream")
                stream = container.streams.video[0]
                duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
                if duration is None and container.duration is not None:
                    duration = container.duration / av.time_base
                if duration is None or not math.isfinite(duration) or not 0 < duration <= self.config.max_video_seconds:
                    raise ValueError("Video duration is missing or exceeds the checkpoint limit")
                frame_rate = float(stream.average_rate) if stream.average_rate is not None else None
                if frame_rate is None or not math.isfinite(frame_rate) or frame_rate <= 0:
                    raise ValueError("Video frame rate is missing or invalid")
                frame_count = stream.frames or math.ceil(duration * frame_rate)
                if not 0 < frame_count <= self.config.max_video_frames:
                    raise ValueError("Video frame count is missing or exceeds the checkpoint limit")
        except av.FFmpegError as exc:
            raise ValueError("Invalid or unsupported video") from exc

        def sample_indices(metadata, **kwargs):
            metadata.total_num_frames = frame_count
            metadata.fps = frame_rate
            metadata.duration = frame_count / frame_rate
            return self.processor.video_processor.sample_frames(metadata=metadata, fps=self.config.video_fps)

        with NamedTemporaryFile() as source:
            source.write(data)
            source.flush()
            try:
                frames, metadata = load_video(
                    source.name,
                    backend="pyav",
                    sample_indices_fn=sample_indices,
                )
            except (av.FFmpegError, OSError) as exc:
                raise ValueError("Invalid or unsupported video") from exc
        if not len(frames) or not metadata.fps or not metadata.total_num_frames:
            raise ValueError("Video has no decodable frames")
        if metadata.frames_indices is None or len(frames) != len(metadata.frames_indices):
            raise ValueError("Video sampled-frame decode is incomplete")
        return MediaWithBytes(
            (
                frames,
                {
                    "total_num_frames": metadata.total_num_frames,
                    "fps": metadata.fps,
                    "duration": metadata.total_num_frames / metadata.fps,
                    "video_backend": "pyav",
                    "frames_indices": list(metadata.frames_indices),
                    "do_sample_frames": False,
                },
            ),
            data,
        )

    def render(self, prompt: SPEVInput) -> str:
        questions = prompt.request.questions
        schema = [{"instructions": q.instructions, "criteria": candidates(q)} for q in questions]
        text = self.config.task_prompt + "\nState: " + literal(prompt.state) + "\nQuestions: " + literal(schema) + "\n"
        for question in questions:
            for criterion in candidates(question):
                text += literal({"question": question.instructions, "criterion": criterion})
                text += "\n" + self.config.marker
        return text

    def pre_process(self, prompt: SPEVInput, request_id=None, **kwargs):
        content = []
        media = {}
        audio = prompt.request.input_audio
        if audio is not None:
            media["audio"] = (self.decode_audio(audio.data), self.config.sample_rate)
            content.append({"type": "audio", "audio": "provided-array"})
        if prompt.image_bytes is not None:
            try:
                media["image"] = [ImageMediaIO().load_bytes(prompt.image_bytes)]
            except (OSError, ValueError) as exc:
                raise ValueError("Invalid or unsupported image") from exc
            content.append({"type": "image", "image": "provided-array"})
        if prompt.video_bytes is not None:
            media["video"] = [self.decode_video(prompt.video_bytes)]
            content.append({"type": "video", "video": "provided-array"})
        isolated = self.config.architecture == "isolated_evidence_v1"
        text = self.config.task_prompt + "\nState: " + literal(prompt.state) + "\n" if isolated else self.render(prompt)
        content.append({"type": "text", "text": text})
        with self.lock:
            rendered = self.processor.apply_chat_template(
                [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = self.processor.tokenizer.encode(rendered, add_special_tokens=False)
            if isolated:
                from .decision_layout import pack_candidates

                ending = self.processor.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
                if ids[-len(ending) :] != ending:
                    raise ValueError("Evidence chat template has an unexpected ending")
                questions = {
                    str(index): {
                        "type": question.type,
                        "instructions": question.instructions,
                        "criteria": dict(enumerate(candidates(question))),
                    }
                    for index, question in enumerate(prompt.request.questions)
                }
                ids, layout = pack_candidates(self.processor.tokenizer, ids[: -len(ending)], questions)
                prompt.decision_layout = layout.to_dict()
        result = {"prompt_token_ids": ids}
        if media:
            result["multi_modal_data"] = media
        return result

    def merge_pooling_params(self, params=None):
        return PoolingParams(task="plugin", requires_token_ids=True)

    def prepare_pooling_params(self, prompt: SPEVInput, params=None):
        params = self.merge_pooling_params(params)
        if self.config.architecture == "isolated_evidence_v1":
            if prompt.decision_layout is None:
                raise ValueError("SPEV input must be preprocessed before preparing pooling parameters")
            params.extra_kwargs = {"spev_layout": prompt.decision_layout}
        return params

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id=None,
        *,
        prompt: SPEVInput | None = None,
        **kwargs,
    ) -> list[DecisionAnswer] | list[float]:
        if len(model_output) != 1:
            raise RuntimeError("SPEV expected one pooled output")
        values = model_output[0].outputs.data.tolist()
        if not all(math.isfinite(x) for x in values):
            raise RuntimeError("Nonfinite SPEV output")
        if prompt is None:
            return values
        if len(values) != sum(len(candidates(q)) for q in prompt.request.questions):
            raise RuntimeError("SPEV candidate count does not match the pooled output")
        answers = []
        offset = 0
        for q in prompt.request.questions:
            n = len(candidates(q))
            logits = [x / self.config.temperature for x in values[offset : offset + n]]
            weights = [math.exp(x - max(logits)) for x in logits]
            probabilities = [x / sum(weights) for x in weights]
            offset += n
            if isinstance(q, DecisionPredicateQuestion):
                answers.append(DecisionPredicateAnswer(type="predicate", name=q.name, probability=probabilities[0]))
            elif isinstance(q, DecisionChoiceQuestion):
                peak = max(probabilities)
                selected = min(
                    (i for i, p in enumerate(probabilities) if p == peak),
                    key=lambda i: (type(q.choices[i].value).__name__, str(q.choices[i].value)),
                )
                answers.append(
                    DecisionChoiceAnswer(
                        type="choice",
                        name=q.name,
                        choice=q.choices[selected].value,
                        probabilities=[{"value": c.value, "probability": p} for c, p in zip(q.choices, probabilities)],
                        confidence=max(0.0, min(1.0, (peak - 1 / n) / (1 - 1 / n))),
                    )
                )
            else:
                peak = probabilities.index(max(probabilities))
                spread = sum(p * abs(i - peak) for i, p in enumerate(probabilities))
                even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
                answers.append(
                    DecisionScoreAnswer(
                        type="score",
                        name=q.name,
                        score=sum(i * p for i, p in enumerate(probabilities)),
                        confidence=max(0.0, 1 - spread / even_spread),
                        probabilities=[
                            {"value": i, "label": level.label, "probability": probabilities[i]}
                            for i, level in enumerate(q.levels)
                        ],
                    )
                )
        return answers
