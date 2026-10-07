# Native audio over the vLLM-Omni OpenAI server

This mode uses `vllm serve --omni`, upstream OpenAI HTTP handlers, response
schemas, authentication, CORS and request-ID middleware. Chat requests use a
strict Pydantic schema bound at the FastAPI edge; `/openapi.json` describes its
supported fields. A frozen
decision handler serves `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings` through one native
vLLM-Omni Thinker engine. It does not start the separate `/v1/decide` server.

## Separate prototype image builds

The integration branch is `prototype/audio-decision-v030`. It is independent of
`release/0.30`; `main` remains the upstream mirror. Merge the prototype PR there
before requesting a build.

A build is requested manually by pushing a tag at a reviewed prototype commit:

```sh
git fetch https://github.com/sprag-ai/vllm-omni.git prototype/audio-decision-v030
git tag audio-decision-build/2026-10-06-1 FETCH_HEAD
git push https://github.com/sprag-ai/vllm-omni.git refs/tags/audio-decision-build/2026-10-06-1
```

Choose a new build-tag name for each request. Normal branch pushes and PR updates
never publish an image. The tag-triggered `Native audio prototype image` workflow
works without adding a manual-dispatch workflow to `main`. It rejects commits
outside the prototype branch history.

Images are published to the separate GHCR package:

```text
ghcr.io/sprag-ai/native-audio-decision:v0.30.0-audio-decision.<full-commit-sha>
```

The version comes from the digest-pinned base in
`docker/Dockerfile.audio-decision-openai`. The build uses the repository's
`GITHUB_TOKEN` with `packages: write`; it needs no production registry credentials.
Organization package-creation policy must permit this repository to publish. If
the package already exists, grant this repository Actions write access to it.
GHCR package visibility and pull access are managed separately from repository
visibility; authenticate with a token with `read:packages` when required.

The run summary records the image tag and digest. Rebuilding the same commit can
replace that tag, so pin the digest for reproducibility. No moving `latest` tag
is produced and nothing deploys automatically. Weights and the frozen decision
bundle remain external mounts. The image is Linux amd64 and uses the same
prototype Dockerfile as the local build below.

## Build and start

```sh
docker build -f docker/Dockerfile.audio-decision-openai \
  -t sprag-audio-decision:openai-v0.30.0 .

docker run --gpus device=0 --ipc=host \
  -p 127.0.0.1:8917:8000 \
  -v /path/to/Qwen3-Omni-model-repository:/models/qwen3:ro \
  -v /path/to/decision-bundle:/models/decision:ro \
  sprag-audio-decision:openai-v0.30.0 \
  /models/qwen3/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695 \
  --omni --decision-bundle /models/decision --host 0.0.0.0 --port 8000
```

When overriding Docker arguments, put the model path first and include `--omni`.
The vLLM CLI rejects flags before the model when `--served-model-name` is set.

Mount the whole Hugging Face model repository, including `blobs`, because the
snapshot contains symlinks. The bundle is the unchanged vllm29 seed17 export;
this port does not refit or recalibrate it. Add `--api-key YOUR_KEY` for bearer
authentication. Keep the port private unless you provide an appropriate deployment
boundary. The existing `--served-model-name` and HTTP/TLS options apply.

The execution policy uses one GPU, BF16, eager, unchunked prefills, no prefix
cache and fresh audio encoding. Concurrent requests enter vLLM's `AsyncLLM`
scheduler and can share a prefill batch. `--max-num-seqs` defaults to 8 (range
1–32); `--max-num-batched-tokens` defaults to 2048 times that limit (range
2048–65536). The model context remains 2048 tokens per request.
`--gpu-memory-utilization` controls the memory allocation. Explicit engine flags such as `--dtype`,
`--max-model-len`, `--seed`, `--enforce-eager`, tokenizer overrides and stage
configuration are rejected at startup, including values supplied through
`--config`. Omit them even when they match the frozen setting. TP and PP may
only be explicitly set to 1. Supported HTTP/authentication/TLS options continue
to apply. The decision engine does not expose the general
vLLM model/sampling/parallelism configuration. `--decision-max-pending 16` bounds
active plus queued inference requests; overflow returns 429.

## Chat completions: calibrated turn decisions

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8917/v1", api_key="YOUR_KEY")
result = client.chat.completions.create(
    model="native-audio-decision",
    messages=[{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {
            "data": base64.b64encode(open("sample.wav", "rb").read()).decode(),
            "format": "wav",
        }},
        {"type": "text", "text": "audio_turn_decision"},
    ]}],
    max_completion_tokens=1,
    temperature=0,
    logprobs=True,
    top_logprobs=3,
)
print(result.choices[0].message.content)
print(result.choices[0].logprobs.content[0].top_logprobs)
```

`A = keep_listening`, `B = respond`, `C = insufficient_evidence`. Chat uses the
frozen calibrated early-exit/fallback policy (`auto`, threshold `0.95`). Its
label-restricted log probabilities are the calibrated policy scores. It does not
accept `decision_mode` or `decision_threshold` overrides. The existing completion
endpoint retains its raw-by-default experimental readout contract.

Supply exactly one user message with one inline audio part. `format` is a
nonempty container hint, not an allowlist: vLLM's shared `load_audio` decoder
inspects the bytes and uses its installed SoundFile/torchcodec/PyAV backends.
Chat, completions and embeddings all use this path. The decoder downmixes to mono;
the decision adapter preserves the frozen model's 16 kHz resampling. An optional
text part must be `audio_turn_decision` or the exact bundled prompt. Arbitrary instructions,
history, tools, multiple clips and streaming are rejected. `max_tokens` and
`max_completion_tokens`, if supplied, must both be 1; `n` must be 1 and
`temperature` 0. `top_logprobs` supports 0 through 3 and requires `logprobs=true`
when nonzero. `modalities`, if supplied, must be `["text"]`.

The standard assistant message, usage and chat log-probability fields survive
OpenAI-compatible gateway response normalization. The direct server's `decision`
metadata extension is optional diagnostics and may be dropped by a gateway.
`cache_salt` is accepted as gateway partition metadata; all cross-request model
caches remain disabled. Dialtone can use its existing chat request/response path
with the matching audio media profile and supported-parameter catalog entry.
Embeddings remain available only through their separate endpoint.

## Completions: label log probabilities

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8917/v1", api_key="EMPTY")
audio = {
    "data": base64.b64encode(open("sample.wav", "rb").read()).decode(),
    "format": "wav",
}
result = client.completions.create(
    model="native-audio-decision",
    prompt="audio_turn_decision",
    max_tokens=1,
    temperature=0,
    logprobs=3,
    extra_body={"input_audio": audio},
)
print(result.choices[0].text)
print(result.choices[0].logprobs.top_logprobs)
```

`A = keep_listening`, `B = respond`, `C = insufficient_evidence`.
`allowed_token_ids`, if supplied, must be the frozen A/B/C token IDs in that order.
The result uses `object: text_completion`, one choice, and standard `usage` and
`logprobs` fields. `decision` is an extension containing control-flow and timing
metadata.

The default `decision_mode="raw"` evaluates the entire decoder and returns raw
label-restricted log-softmax probabilities. This matches the probability contract
used by AnyJev's raw/L0 readout; it does not perform option rotations or calibration.

To preserve the previous API's calibrated early-exit/fallback behavior:

```python
extra_body={
    "input_audio": audio,
    "decision_mode": "auto",
    "decision_threshold": 0.95,
}
```

Other modes: `head` always uses the frozen block24 head; `full` always uses the
full decoder with the frozen full-readout temperature. The distinction matters:
raw and calibrated probability values have different meanings.

## Embeddings: apply the head client-side

```python
result = client.embeddings.create(
    model="native-audio-decision",
    input="audio_turn_decision",
    encoding_format="float",
    extra_body={"input_audio": audio},
)
hidden = result.data[0].embedding
```

This returns the **unnormalized last-token post-block24 residual**, cast losslessly
from BF16 to float32. It is not a semantic similarity embedding. `encoding_format`
can be `float` or `base64` (little-endian float32); the OpenAI SDK also handles its
default base64 behavior. No dimension reduction or activation is applied.

Use `tools/audio_decision/openai_client.py --mode embedding --bundle /path/to/bundle sample.wav` to
apply the exported FP64 feature normalization, ridge head and head temperature
client-side. The head bundle must match the served backbone, adapter and depth.
This head-only result does not invoke the full-decoder fallback. Use completion
mode `auto` when you want fallback in the same forward pass.

This implementation shares the validated one-prefill generation runner: it
internally samples one label token and discards it for embedding responses.
`decision.internal_sample_tokens=1` makes that cost explicit. It is not vLLM's
dedicated pooling runner, does not free unused later-layer weights, and does not
use a dedicated pooling schedule. Concurrent completion and embedding requests
can share the same scheduler batch. There is no autoregressive response generation or
incremental audio/KV streaming.

## Audio contract and supported task

All three endpoints accept inline base64 audio supported by the installed vLLM
audio loader, up to 30 seconds and 12 MiB after base64 decoding. The adapter also
bounds decoded PCM memory and requires a source rate of 8–192 kHz. Audio is
downmixed and resampled to 16 kHz. No URL fetching, server-local file paths, transcript, labels, source IDs
or provider IDs are accepted. Every call contributes one fresh audio item to the encoder batch; no decision
result, prefix or audio-feature cache is reused between requests.

Use `audio_turn_decision` or the exact bundled prompt as `prompt`/`input`.
Arbitrary questions, text-only calls, batch arrays, streaming, different labels,
multiple generated tokens, and unsupported sampling/pooling fields return errors.
These endpoints expose a fixed trained turn-taking task; standard paths do not
turn it into a general-purpose classifier.

Upstream AnyJev's `VLLMBackend` supplies text, not audio. Its endpoint/response
conventions are reused here, but an audio-aware caller must provide `input_audio`
and the correct frozen head. The included OpenAI SDK client is a working example;
an unmodified text-only AnyJev client cannot supply this native-audio task.

Readiness is `GET /health` (200), model discovery is `GET /v1/models`, and queued
request load is `GET /load`. Unrelated generation routes are not advertised in
this mode.

## Async execution and batch diagnostics

The server decodes audio in a bounded CPU thread pool, then awaits the async
engine. It does not serialize GPU calls in a Python inference lock. A worker
binds request metadata after the native scheduler compacts/reorders its batch,
and the model reads each request's final prefill token. Thresholds, execution
modes and results remain keyed to individual requests. `--decision-max-pending`
continues to cap all admitted work, including disconnected requests while their
inference drains.

If every request accepts the early head, the batch exits at block 24. If any
request needs full depth, the batch continues together; accepted requests still
return their saved block-24 head scores. `decoder_depth` records the selected
readout depth and `batch_decoder_depth` records the layers actually executed.
`batch_size`, `audio_encoder_items`, `batch_audio_items` and
`batch_audio_encoder_calls` distinguish per-request audio from shared encoder
executions. The engine defaults to and requires `VLLM_BATCH_INVARIANT=1`:
ordinary batch-dependent kernels changed a forced-head action in regression
testing. FP64 response log-probabilities are computed on CPU because vLLM's
invariant CUDA log-softmax does not support FP64. This kernel configuration
can differ numerically from the legacy serial runtime; parity and performance
must be measured on the target device. It does not imply Transformers parity.

The audio tower groups recordings by their standalone convolution padding
width. Otherwise, a longer recording changes a short recording's boundary
features through biased convolutions. Long recordings still share an encoder
batch; mixed short lengths may require several internal encoder groups. The
decoder retains the native scheduler batch across those groups.
With invariant mode enabled, the CUDA audio convolutions use unfold plus vLLM's
invariant linear kernel; cuDNN's small-batch algorithm changed BF16 results in
the third convolution despite identical inputs. This uses additional temporary
workspace and must be included in target-device latency/memory validation.

This path requires model runner V1 (`VLLM_USE_V2_MODEL_RUNNER=0`). Async HTTP
admission and native scheduler batching are enabled; vLLM's separate
`async_scheduling` option for CPU/GPU step pipelining remains disabled. Chunked
prefill, speculative decoding, pipeline/tensor parallelism and incremental
audio streaming remain unsupported. The standalone `/v1/decide` prototype
continues to use its legacy serial engine.
