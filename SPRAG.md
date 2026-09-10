# The Sprag fork

This is a fork of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) carrying
patches we need for serving symphony and rhythm. It exists because those patches are not ones we
can land upstream: they encode our serving decisions, not general improvements.

Everything here is sprag-only. Upstream has no file at this path and no workflow named
`sprag-image-release.yml`, so syncing from upstream never conflicts with either.

## Branch model

`main` is a **mirror of upstream**. It carries no sprag patches, and nothing is ever built from it.
Keeping it clean is what makes an upstream sync a fast-forward instead of a merge.

Our code lives on a **release branch per upstream minor version**:

```
main (upstream mirror)
  └── release/0.28        <- every sprag patch for the v0.28 line; images are built from here
        ├── feat/...      <- work in progress, merged in by PR
        └── feat/...
```

One release branch is the integration point for a whole upstream line. A feature branch is
short-lived: open a PR into `release/0.28`, merge, and the branch is done. Images are built only
from `release/**`, so a feature branch never produces one — test by merging into the release branch
and rolling **dev**, which is safe because nothing deploys automatically (see *Deploying*).

### History, for anyone reading old tags

Before September 2026 there was no integration branch, and each image was built from whichever
feature branch happened to hold the work. That is why older tags carry a `realtime` label that
stopped describing their contents: the label was inherited from a predecessor tag, not derived from
the source. `release/0.28` was cut from `feat/realtime-instructions-0.28-transcription`, which
already contained `feat/realtime-instructions-0.28`, which is the v0.28 port of the older
`feat/realtime-instructions`. That last branch is **superseded, not missing** — its two files
(`realtime_connection.py`, `realtime_tool_calls.py`) are present on `release/0.28` byte-identical,
and the branch itself sits ~180 commits behind `main`. Do not merge it forward.

## Image tags

```
<upstream base version>-sprag.<commit>

v0.28.0-sprag.3bdf9321
└─ upstream ──┘      └── the release-branch commit that built it
```

Published to `us-central1-docker.pkg.dev/steady-method-485022-e5/sprag-prod/vllm-omni`.

Both halves are load-bearing. The upstream half is read from `ARG BASE_IMAGE` in
`docker/Dockerfile.cuda` rather than hardcoded in the workflow, so it cannot drift from the base
actually used. The commit half names the source exactly, which makes tags immutable by
construction: a given tag always means one tree. The workflow relies on that — if the tag already
exists it skips the build rather than republishing a new digest under a name seed has pinned.

There is deliberately **no moving tag** (no `latest`, no `v0.28.0-sprag`). Seed pins exact tags, so
a moving tag would add a second answer to "what is deployed" without giving us anything.

## Deploying

Building publishes an image. It does not deploy one. Rolling it out is a separate, explicit change
in [seed](https://github.com/sprag-ai/seed) — pin the tag in the cluster's model values file:

```
seed/clusters/us-central1/inference/models/symphony/values.yaml.gotmpl   # dev
seed/clusters/us-west1/inference/models/symphony/values.yaml.gotmpl      # prod
```

Dev and prod are pinned independently and are routinely on different tags. That separation is the
reason a release-branch build is safe: merging into `release/0.28` can never move prod.

## Publishing identity

The workflow authenticates by Workload Identity Federation, with no long-lived key. The binding is
scoped to **one repository and one ref**:

```
principalSet://iam.googleapis.com/projects/254791206063/locations/global/workloadIdentityPools/
  github-actions/attribute.repository_ref/sprag-ai/vllm-omni@refs/heads/release/0.28
```

so only a run on `release/0.28` can assume the pushing service account. A run on any other branch
cannot publish even if the workflow were altered to try, which is the point: production pulls from
`sprag-prod`, so "can push a branch" must not imply "can publish an image production pulls".

The workflow's `if: startsWith(github.ref, 'refs/heads/release/')` guard is a convenience that fails
the run early with a clear message. The IAM binding is the actual control.

## Cutting a new release branch

When upstream ships a new minor version — say v0.29.0:

1. Sync `main` from upstream (fast-forward).
2. Branch `release/0.29` from `main`, then carry the sprag patches over from `release/0.28`.
3. Point `ARG BASE_IMAGE` in `docker/Dockerfile.cuda` at `vllm/vllm-openai:v0.29.0`. The tag scheme
   follows automatically; no workflow edit.
4. Add the IAM binding for the new ref — this is the one manual step, and it is manual on purpose,
   since it is the grant that lets a branch publish:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  vllm-omni-ci@steady-method-485022-e5.iam.gserviceaccount.com \
  --project=steady-method-485022-e5 \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/254791206063/locations/global/workloadIdentityPools/github-actions/attribute.repository_ref/sprag-ai/vllm-omni@refs/heads/release/0.29"
```

5. Keep `release/0.28` until nothing pins its tags, then remove its binding.

## Repository configuration

Two repository secrets, both identifiers rather than credentials:

| secret | value |
| --- | --- |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/254791206063/locations/global/workloadIdentityPools/github-actions/providers/github` |
| `GCP_SERVICE_ACCOUNT` | `vllm-omni-ci@steady-method-485022-e5.iam.gserviceaccount.com` |

The service account needs `roles/artifactregistry.writer` on the `sprag-prod` repository — writer,
not admin, so a compromised run can add images but cannot delete what production is running.

## Build cost

`docker/Dockerfile.cuda` layers a pure-Python install onto the CUDA base; nothing compiles. The
install step measured ~42 seconds. Almost all wall-clock is pulling and unpacking the ~30 GB base,
which is also why the workflow reclaims runner disk before building.
