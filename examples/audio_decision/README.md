# Standalone native audio decision prototype (vLLM-Omni 0.30.0)

This page describes the legacy serial `/v1/decide` server. For the async,
batched OpenAI endpoints, see [OPENAI.md](OPENAI.md).

This is a single-GPU, audio-in/decision-out service for the frozen runtime26
conversational-turn model. It runs the **native vLLM-Omni Qwen3-Omni Thinker** on
the vLLM engine; it does not execute a Transformers model behind the API.
The Thinker-only path does not start Talker or Code2Wav stages.

It returns `keep_listening`, `respond`, or `insufficient_evidence`, plus the
three probabilities, confidence, actual decoder depth, and timing. The caller
supplies a threshold. The first port intentionally executes one request at a
time; simultaneous HTTP requests enter a bounded queue. It is not a batched
or incrementally streaming implementation.

## Export the existing research checkpoint

Use the preserved research Python environment (NumPy and safetensors required):

```sh
python tools/audio_decision/export_bundle.py \
  --runtime /path/to/research/runtime26 \
  --output /path/to/decision-bundle \
  --model weight1_s17
```

The export verifies the original weights/head hashes and losslessly renames all
192 FP32 Q/V adapter tensors. It does not merge, quantize, train, or recalibrate.
`weight1_s29` is supported by the schema but requires its own validation; the
initial deployment uses the validated seed17 configuration at decoder block 24.

## Build and run

```sh
docker build -f docker/Dockerfile.audio-decision \
  -t sprag-audio-decision:v0.30.0 .

docker run --name native-audio-decision --gpus 'device=0' --ipc=host \
  -p 127.0.0.1:8917:8000 \
  -v /path/to/hf-model-repository:/models/qwen3:ro \
  -v /path/to/decision-bundle:/models/decision:ro \
  -v decision-kernel-cache:/root/.cache \
  sprag-audio-decision:v0.30.0 \
  --model /models/qwen3/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695
```

The model repository mount must contain **both `snapshots/` and `blobs/`** so
Hugging Face symlinks resolve. Alternatively mount a complete, dereferenced
checkpoint directory and point `--model` to that directory. Backbone weights
and the exported head/adapter are external read-only mounts, not baked into the
image. The runtime operates offline. One H100 80GB was used for validation.

The HTTP service is private to the node with this port binding. For remote
access, use the existing configured SSH connection's port forwarding or a
separately managed authenticated proxy. Set `DECISION_API_KEY` in the container
environment to require a bearer token on `/v1/decide`; health is unauthenticated.
No registry push or external publication is required.

```sh
curl http://127.0.0.1:8917/health
curl --fail-with-body 'http://127.0.0.1:8917/v1/decide?threshold=0.95' \
  -H 'Content-Type: audio/wav' --data-binary @sample.wav
```

The request body is raw audio bytes (WAV/FLAC supported by libsndfile), not
multipart form data, a transcript, or a server file path. Audio is converted to
mono 16 kHz. The shared vLLM audio loader handles container detection and decoder fallback.
Limits: 30 seconds, 12 MiB upload, bounded decoded PCM memory, sample rates
8–192 kHz, 16 pending requests. Invalid input returns 422, oversized upload
413, full queue 429. The service does not persist uploaded audio.

`elapsed_ms` covers the warm engine call. `server_elapsed_ms` also includes
request reading, decoding, and queue wait; neither includes client-side audio
capture or all network overhead. First requests can incur kernel compilation.

## Execution and numerical contract

The frozen head uses the logical post-block residual: vLLM's separate hidden
state and residual are added in BF16 before the original FP64 normalization and
head projection. Accepted requests exit at the configured depth. Rejected
requests continue the same forward through all 48 decoder blocks. There is no
second pass, split decoder, autoregressive answer generation, or cross-request
KV/prefix/embedding reuse. vLLM still allocates internal attention KV storage.

The standard vLLM LoRA kernels cannot combine BF16 activations and our FP32
adapter tensors. This port uses persistent Q/V projection hooks to apply the
original FP32 low-rank updates after vLLM's fused base QKV projection. K is
unchanged. Native vLLM attention and MoE kernels remain in use. CPU tests check
this arithmetic and early-exit control flow. Native engine arithmetic can still
differ from the Transformers reference; see the deployment validation report
for measured decision, gate, and probability differences. Do not assume bitwise
parity or fresh-source calibration.

Supported execution is eager, TP=1, PP=1, max_num_seqs=1, unchunked prefill,
max_tokens=1, prefix caching disabled. Unsupported combinations fail explicitly.
The inference lock and worker request IDs prevent cross-request metadata mixing.
The original prompt, temperatures and weights stay frozen.

## Tests

Run CPU contract tests in the built image with the test directory mounted:

```sh
docker run --rm --entrypoint python3 \
  -v "$PWD/tests/entrypoints/audio_decision:/validation:ro" \
  sprag-audio-decision:v0.30.0 -m pytest -q /validation
```

For a GPU smoke, override the entrypoint and run
`tools/audio_decision/smoke.py --model ... --bundle ... --audio ... --output ...`
with a 16 kHz mono WAV. It exercises forced head, forced full depth and automatic
routing. The output location must be writable.
