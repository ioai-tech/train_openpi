# Contributing

## Development setup

Use Python 3.11 for local tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m pytest
```

Tests must use generated fixtures. Do not commit robot recordings, datasets,
checkpoints, or other user data.

## Pull requests

- Keep LeRobot v3 conversion and the training wrapper independently testable.
- Add a regression test for bug fixes.
- Preserve the `/data/input`, `/data/output`, and checkpoint contracts unless
  the change is explicitly documented as breaking.
- Run all tests and at least one Docker build before requesting review.

## Dependencies and CUDA images

The supported build matrix is defined in `.github/workflows/docker.yml`.
All published images use CUDA 12.6.

| Tag | Model | CUDA |
| --- | --- | --- |
| `pi0-cuda126` (`pi0`, `latest`) | Pi0 | 12.6 |
| `pi05-cuda126` (`pi05`) | Pi0.5 | 12.6 |

A variant must use the digest-pinned `nvidia/cuda:12.6.3-runtime-ubuntu22.04`
base in `Dockerfile`. `latest` and `pi0` remain aliases for `pi0-cuda126`.
`pi05` remains an alias for `pi05-cuda126`.

## Updating OpenPI

OpenPI is pinned by `OPENPI_GIT_REF` in `Dockerfile`. To update it:

1. Review upstream changes between the old and new commits.
2. Update the commit in `Dockerfile` and `THIRD_PARTY_NOTICES.md`.
3. Build both image variants and run a GPU smoke training.

Do not point published images at a mutable upstream branch.

## Releases

Pushes to `main` publish the floating tags (`pi0-cuda126`, `pi0`, `latest`,
`pi05-cuda126`, `pi05`). A Git tag matching `vMAJOR.MINOR.PATCH` publishes
versioned tags such as `MAJOR.MINOR.PATCH-pi0-cuda126`. Docker Hub credentials
are provided through the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository
secrets.
