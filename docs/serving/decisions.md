# Decisions with a pooling model

The `/v1/decisions` endpoint accepts shared evidence and ordered predicate,
choice, or score questions. A model's I/O processor prepares the evidence and
interprets the pooled output. The serving layer handles the public contract,
admission, errors, and native token usage.

## Model integration

Implement the model under `vllm_omni/model_executor/models/` and register its
architecture in the model registry. A decision model provides:

- A native vLLM `Pooler` that returns one tensor per request.
- A vLLM `IOProcessor` with `supports_decisions = True`.
- An installed `vllm.io_processor_plugins` entry point selected by the model's
  Hugging Face `config.json` through `io_processor_plugin`.

The route invokes `parse_data`, `pre_process`, `EngineClient.encode`, and
`post_process`. It passes the parsed input to `post_process` as `prompt`, so a
processor can interpret request-specific questions without retaining shared
request state. Decisions processors return the shared `DecisionAnswer` types.
The current Decisions route requires one prepared prompt and one pooled result
per request.

Prompts, calibration, media preprocessing, and architecture restrictions belong
to the model implementation and configuration. Registering a different decision
model does not require adding a model branch to the serving layer.

`--runner pooling` uses vLLM's single-stage engine. Multi-stage pooling through
`AsyncOmni.encode` is not implemented. Stage deployment, strategy, and override
options cannot be combined with this runner. Headless and multiple API-server
modes are also unsupported.

Pooling HTTP request bodies are limited to 64 MiB before JSON parsing. Set
`--max-request-body-bytes` to a positive byte count to change this server limit.
Oversized requests return HTTP 413. Model-specific media limits still apply.

## SPEV

SPEV uses a Qwen3-Omni backbone, FP32 residual adapters, and a contextual
candidate head. Its pooler reads candidate anchor positions from native pooling
metadata and returns the scores through ordinary pooling outputs.

The model configuration selects `SPEVForDecisions` and the `spev_decisions` I/O
processor. Its `spev` object declares the head architecture and dimensions,
adapter rank and scale, calibration temperature, anchor token IDs, asset hashes,
prompt strings, media preprocessing, and input limits. The model directory must
contain the base model weights and tokenizer files plus `adapter.safetensors`
and `head.safetensors` matching those hashes.

```bash
vllm serve /models/spev --omni --runner pooling \
  --served-model-name spev --dtype bfloat16 --max-model-len 8192 \
  --enforce-eager --no-enable-chunked-prefill --no-enable-prefix-caching \
  --max-num-seqs 8 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"audio":1,"image":1,"video":1}'
```

SPEV currently requires TP=1 and PP=1 with complete eager prefills. The normal
vLLM scheduler batches requests. The head produces all candidate scores in one
prefill, without generating a completion token.

```json
{
  "model": "spev",
  "input": "The object is red.",
  "questions": [{
    "type": "choice",
    "instructions": "What is the predominant color?",
    "choices": [{"value": "red"}, {"value": "blue"}]
  }]
}
```

The public Decisions input supports text, inline images, and the optional
`input_audio` extension. SPEV's model processor also supports video evidence
internally; the public Decisions schema does not define a video input part.
For direct `/pooling` plugin calls, SPEV returns the candidate logit vector.
The Decisions route supplies question context to obtain typed answers instead.
