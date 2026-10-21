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

The route invokes `parse_data`, `pre_process`, `prepare_pooling_params`,
`EngineClient.encode`, and `post_process`. Parameter preparation receives the
parsed input after preprocessing and returns request-local `PoolingParams`.
The default implementation delegates to `merge_pooling_params`.
The route passes the parsed input to `post_process` as `prompt`, so a
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
candidate head. Isolated-head checkpoints share an evidence prefix, keep
candidate attention within its own branch, and score all questions in a batch
with one head invocation. Scores use ordinary pooling outputs.

Isolated-head checkpoints require vLLM's `SupportsPoolingMetadata` interface.
The model receives request-aligned pooling metadata before its forward pass;
the existing MLP checkpoint format does not require that interface.

Standalone Transformers checkpoints select `SPEVModel`. The native loader reads
the nested Thinker configuration and indexed backbone, adapter, and head tensors.
Adapters and the head retain FP32 arithmetic; the unused generation head is not
required. Select the `spev_decisions` I/O processor when serving this format.

`SPEVForDecisions` checkpoints instead declare a `spev` configuration object and
separate, hash-verified `adapter.safetensors` and `head.safetensors` files.

```bash
VLLM_BATCH_INVARIANT=0 vllm-omni serve /models/spev --omni --runner pooling \
  --trust-remote-code --io-processor-plugin spev_decisions \
  --served-model-name spev --dtype bfloat16 --max-model-len 8192 \
  --enforce-eager --no-enable-chunked-prefill --no-enable-prefix-caching \
  --max-num-seqs 8 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"audio":1,"image":1,"video":1}'
```

SPEV currently requires TP=1 and PP=1 with complete eager prefills. The normal
vLLM scheduler batches requests. The head produces all candidate scores in one
prefill, without generating a completion token.
Isolated-head checkpoints require unquantized BF16 weights and
`VLLM_FLOAT32_MATMUL_PRECISION=highest` (the default). Their model implementation
preserves checkpoint arithmetic and isolates each request and candidate branch;
vLLM's global batch-invariant kernels must remain disabled.

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
`input_audio` and `input_video` extensions. Each extension contains `data`
(base64-encoded file bytes) and `format` (a container hint). Video supplies visual
frames; supply its soundtrack separately through `input_audio` when needed.
SPEV's checkpoint configuration controls frame sampling and media limits.
For direct `/pooling` plugin calls, SPEV returns the candidate logit vector.
The Decisions route supplies question context to obtain typed answers instead.
