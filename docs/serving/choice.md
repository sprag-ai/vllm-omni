# Decisions API

`--choice-bundle PATH` serves `POST /v1/decisions` using the trained named-choice likelihood scorer. The public request and response follow the [OpenAI Decisions contract](https://developers.openai.com/api/docs/guides/decisions). Native audio is an additional Sprag input. This is wire-format compatibility within the model limits below, not equivalence to OpenAI model behavior or confidence calibration.

```sh
vllm serve /models/qwen3 --omni --choice-bundle /models/choice \
  --served-model-name spev --max-num-seqs 8 \
  --max-num-batched-tokens 16384 --api-key "$SPEV_API_KEY"
```

## Request

```json
{
  "model": "spev",
  "input": "I was charged twice. Please refund the duplicate payment.",
  "questions": [{
    "name": "department",
    "type": "choice",
    "instructions": "Which team should handle this?",
    "choices": [
      {"value": "billing", "description": "Payments and refunds"},
      {"value": "technical", "description": "Software issues"}
    ]
  }]
}
```

Authenticate with the existing vLLM bearer middleware. Questions and answers are arrays in the same order. Names are optional, unique when supplied, echoed as `null` when omitted, and excluded from scoring prompts. Instructions are strings. Choice values are strings or booleans; `true` and `"true"` are distinct. String values retain their exact trained target names. Boolean values receive collision-free internal names and preserve their original type in the response.

`predicate` questions take `instructions` and an optional `name`; their answer contains `probability`. `score` questions take `levels: [{"label": "Routine"}, {"label": "Urgent", "description": "Needs immediate action"}]`; their answer contains the probability-weighted zero-based level index, `confidence`, and per-level `value`, `label`, and `probability`.

## Response

Illustrative values:

```json
{
  "model": "spev",
  "answers": [{
    "type": "choice",
    "name": "department",
    "choice": "billing",
    "probabilities": [
      {"value": "billing", "probability": 0.9},
      {"value": "technical", "probability": 0.1}
    ],
    "confidence": 0.8
  }],
  "usage": {
    "input_tokens": 123,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "multimodal_tokens": null},
    "output_tokens": 0,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 123
  }
}
```

Probabilities come from complete named-target sequence likelihoods, not generated numeric text. Choice confidence remains `(max_probability - 1/N)/(1 - 1/N)`. Score confidence retains the scorer's spread formula. These formulas are implementation choices, not a claim about OpenAI's scoring internals. The schema can represent refusals; this candidate has no newly trained refusal classifier.

## Media and model limits

Text evidence can be a string or ordered user messages with string content or `input_text` and `input_image` parts. Image input uses `image_url: "data:image/png;base64,..."`. The model currently accepts at most one inline image, with omitted or `auto` detail, and requires `--choice-enable-vision`. Multiple images, remote image URLs, other detail levels, file IDs, and other message roles are rejected. This is a subset of the upstream image interface; no remote fetching occurs.

The Sprag audio extension is top-level `input_audio: {"data": "<base64>", "format": "wav"}` alongside `input` and `questions`. The decoder determines supported containers; `format` is a hint. Decoded bytes are bounded to 12 MiB, with the existing decoder channel/rate/duration and aggregate JSON-body limits. Video has no public Decisions input. Existing internal visual processing remains available to the scorer.

The model permits 16 questions, 255 total candidate outcomes, 2–255 choices per choice question, and 2–10 score levels. Text work is bounded to 65,536 characters and 4,096 structure nodes. The 8,192-token context includes expanded media and candidate suffixes. Unsupported generation parameters and `stream` are rejected. The HTTP endpoint evaluates a complete input; it does not implement streaming audio sessions.

## Runtime and accounting

The bundle remains `choice.json`, adapter config/weights, and `calibration.json`, with file hashes verified on startup. No weights or trained string targets change. The scorer uses native Qwen3-Omni Thinker, frozen BF16 base, FP32 adapter, eager execution, TP/PP=1, seed 17 and invariant kernels. The separate frozen `--decision-bundle` mode is unchanged and mutually exclusive with `--choice-bundle`.

The existing bundle temperature applies to text/audio choice scoring. Predicate and score questions, and visual requests, use identity temperature. Existing calibration does not establish accuracy or calibration for arbitrary questions, mixed boolean choices, or streaming prefixes.

Usage counts the logical shared prompt once per question, including native modality placeholders. It excludes candidate target suffixes and discarded internal samples. `multimodal_tokens` is a Sprag accounting extension whose counts come from vLLM's rendered placeholder spans. Absent modalities are omitted; text-only input uses null. The gateway preserves this canonical usage envelope while settling per-modality billing internally. Cache counts and output counts are zero in this scorer.

Candidates are scheduled concurrently but each currently repeats model prefill and media preprocessing. Questions run sequentially. Prefix caching and the multimodal processor cache remain disabled. No speedup is implied by adopting the Decisions contract.

Invalid requests return 422 at the native server (400 at the gateway). Full admission returns 429 with `Retry-After: 1`; an unavailable engine returns 503. `/load` includes outstanding inference after an HTTP waiter disconnects. Shutdown drains tracked work.

## Migration

`POST /v1/systemone` returns 410 Gone with `error.code: "endpoint_removed"` and instructions to use `/v1/decisions`. It never translates payloads or invokes inference. The former named-Choice chat wrapper is removed from this opt-in server mode.

Migrate `state` to `input`; convert the question map to an array with `name`; replace choice `criteria` with `choices` entries, `noul` with `predicate`, and score `criteria` with `levels`. Read answers and probabilities as ordered arrays. This structural change cannot be expressed as Pydantic field aliases.

The gateway and model catalog must advertise `/v1/decisions`, and the model must run a matching native server revision. Existing published artifacts do not gain this route automatically.
