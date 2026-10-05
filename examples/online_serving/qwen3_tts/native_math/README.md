# Qwen3-TTS native arithmetic correctness baseline

Build from the Sprag `release/0.28` fork with the audited, immutable Omni base:

```sh
docker build -f docker/Dockerfile.qwen-native --build-arg SOURCE_REVISION=$(git rev-parse HEAD) -t qwen-native:local .
docker run --rm --gpus device=0 --ipc=host -p 8000:8000 \
  -v /path/to/huggingface/hub:/root/.cache/huggingface/hub:ro \
  -v /path/to/approved/voices:/voices:ro \
  qwen-native:local serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni \
  --served-model-name chorus-clone \
  --deploy-config /app/vllm-omni/examples/online_serving/qwen3_tts/native_math/deploy.yaml
```

Weights and voice enrollments are external read-only inputs. The image includes
all runtime code and the deployment YAML; no source overlays are required.
Use the exact cached checkpoint path in offline environments.

The opt-in `SPRAG_QWEN_NATIVE_MATH=1` uses native BF16 residual, normalization,
RoPE and all-16 codec reduction boundaries; reproduces Torch 2.7's FP32 mean
addition tree; and explicitly selects Torch flash SDPA with repeated KV heads.
The residual predictor stays eager. Its temperature division is FP32.
`SPRAG_PRESERVE_RNG_STATE=1` bypasses vLLM startup/profile reseeding. Requests in
the smoke test supply no seed. This image is for Qwen3-TTS only.

## Supported serving envelope

CUDA BF16, TP=PP=1, eager talker, one active sequence, complete unchunked prefill,
and prefix caching disabled. Unsupported settings fail at model initialization.
HTTP requests may queue concurrently; execution remains serial. Every prefill
at position zero replaces each layer's full KV cache. Noncontiguous decode fails
rather than reusing stale audio context. Cancellation/preemption resumes with
complete replay because paged/prefix cache reuse is disabled.

This is a slower correctness baseline, not a scalable paged backend. KV buffers
retain the last request until the next prefill and are bounded by max_model_len.
The stock audio decoder remains chunked/compiled. High-concurrency rollout needs
a paged-cache adapter and separate regression tests. Do not merge or deploy to
production based only on this experiment.

Render repeat and queued requests:

```sh
python examples/online_serving/qwen3_tts/native_math/render.py \
  --url http://localhost:8000 --output /tmp/qwen-native-audition
```

Record the built image ID/digest, source revision, checkpoint and profile hashes
with results. Listen for delivery as well as validating WAV shape. Numerical
teacher-forced checks belong outside the image; generated audio is not a
reference for reenrollment.
