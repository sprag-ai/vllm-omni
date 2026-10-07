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
  └── release/0.30        <- every sprag patch for the v0.30 line; images are built from here
        ├── feat/...      <- work in progress, merged in by PR
        └── feat/...
```

One release branch is the integration point for a whole upstream line. A feature branch is
short-lived: open a PR into the release branch, merge, and the branch is done. Images are built only
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

v0.30.0-sprag.3bdf9321
└─ upstream ──┘      └── the release-branch commit that built it
```

Published to `us-central1-docker.pkg.dev/steady-method-485022-e5/sprag-prod/vllm-omni`.

The installed `vllm_omni` package reports `<upstream version>+sprag` (e.g. `0.30.0+sprag`), passed to the
build as `VLLM_OMNI_VERSION_OVERRIDE`. PEP 440 has no `-sprag` form, so the local-version label carries it.
Without the override the build context has no tags and the package would report a `0.1.devN` version that
vLLM flags as mismatched.

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
reason a release-branch build is safe: merging into a release branch can never move prod.

## Publishing identity

The workflow authenticates by Workload Identity Federation, with no long-lived key, as
`vllm-omni-ci@steady-method-485022-e5.iam.gserviceaccount.com`. The service account, its
`roles/artifactregistry.writer` on `sprag-prod`, and its binding are managed in seed
(`infra/envs/project`). The binding admits **any `release/*` branch of this repository and no other ref**:

```
principalSet://iam.googleapis.com/projects/254791206063/locations/global/workloadIdentityPools/
  github-actions/attribute.release_repository/sprag-ai/vllm-omni
```

`attribute.release_repository` is mapped on the pool's provider (seed `infra/bootstrap`) to the
repository name when the token's ref is `refs/heads/release/*` and `none` otherwise; IAM matches
attribute values exactly, so a prefix cannot be expressed in the binding itself. A run on any other
branch cannot publish even if the workflow were altered to try: production pulls from `sprag-prod`, so
"can push a branch" must not imply "can publish an image production pulls". Writer, not admin, so a
compromised run can add images but cannot delete what production is running.

The workflow's `if: startsWith(github.ref, 'refs/heads/release/')` guard is redundant with its
branch filter; it is kept so a trigger added later cannot silently widen what publishes. The IAM
binding is the actual control.

The workflow is push-triggered only. `workflow_dispatch` would have to live on the default branch to
be selectable, and `main` is a clean upstream mirror, so there is no dispatcher to add. To rebuild
without a new commit, re-run the previous run from the Actions UI: it replays against the same ref.

## Cutting a new release branch

When upstream ships a new minor version, say v0.31.0:

1. Sync `main` from upstream (fast-forward).
2. Branch `release/0.31` from `main`, then carry the sprag patches over from the previous release branch.
3. Point `ARG BASE_IMAGE` in `docker/Dockerfile.cuda` at `vllm/vllm-openai:v0.31.0`. The image tag and
   package version follow automatically; no workflow edit and no IAM change.
4. Keep the previous release branch until nothing pins its tags.

## Repository configuration

Two repository secrets, both identifiers rather than credentials:

| secret | value |
| --- | --- |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | `projects/254791206063/locations/global/workloadIdentityPools/github-actions/providers/github` |
| `GCP_SERVICE_ACCOUNT` | `vllm-omni-ci@steady-method-485022-e5.iam.gserviceaccount.com` |

## Build cost

`docker/Dockerfile.cuda` layers a pure-Python install onto the CUDA base; nothing compiles. The
install step measured ~42 seconds. Almost all wall-clock is pulling and unpacking the ~30 GB base,
which is also why the workflow reclaims runner disk before building.
