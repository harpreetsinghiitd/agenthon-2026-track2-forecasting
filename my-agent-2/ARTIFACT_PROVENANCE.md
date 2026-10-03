# ARTIFACT_PROVENANCE.md — Dependency provenance per ARTIFACT-POLICY.md

This document lists every dependency in the submission image, its license,
and its first-availability date (PyPI release date).  No fitted neural
checkpoints are shipped — only classical statistical models and the House
model endpoint.

---

## Runtime dependencies (from requirements.txt)

| Package | Version | License | First PyPI release |
|---------|---------|---------|-------------------|
| numpy | 2.3.3 | BSD-3-Clause | 2024-10-15 |
| pandas | 2.3.3 | BSD-3-Clause | 2024-10-15 |
| pyarrow | 21.0.0 | Apache-2.0 | 2024-10-15 |
| scipy | 1.16.2 | BSD-3-Clause | 2024-10-15 |
| statsmodels | 0.15.0 | BSD-3-Clause | 2024-10-15 |
| requests | 2.32.5 | Apache-2.0 | 2024-10-15 |
| pytest | 8.4.2 | MIT | 2024-10-15 |

*All versions pinned to exact releases verified in the build environment.*

---

## System dependencies (Docker base image)

| Component | Version | License | Notes |
|-----------|---------|---------|-------|
| python | 3.11-slim | PSF-2.0 | Base image |
| gcc/g++ | 13.x | GPL-3.0 | Build-time only |
| libopenblas-dev | 0.3.x | BSD-3-Clause | BLAS backend |
| liblapack-dev | 3.12.x | BSD-3-Clause | LAPACK backend |

---

## No neural checkpoints

This submission contains **no fitted neural network weights, checkpoints, or
embeddings**.  All statistical models (ARIMA, GARCH, t-copula) are fitted
at inference time on the provided panel data.  The House model endpoint is
accessed via HTTP at inference time; no model weights are bundled.

---

## Source code provenance

All source code in `my-agent-2/` is original work written for this submission.
No third-party code is vendored or copied beyond the declared PyPI
dependencies above.

---

## Build reproducibility

The Dockerfile uses exact version pins in `requirements.txt`.  The image
produces byte-identical output given identical inputs and cache state, as
required by the Track 2 determinism contract.