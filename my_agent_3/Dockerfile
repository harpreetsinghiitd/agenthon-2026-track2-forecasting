# Agenthon 2026 Track 2 — my_agent_3 submission image.
#
# Same shape as the organizer's reference image: the verb is an executable on
# PATH, CMD is `forecast --help` so `docker run <img>` is self-describing, and
# the harness overrides argv with the real invocation.
#
# Build (from the repository root, the build context):
#   docker build -t my_agent_3:smoke -f my_agent_3/Dockerfile .
#
# Run one unit exactly the way the scoring harness does:
#   docker run --rm --network=none \
#     -v units/t2-F2-abenomics-2012:/input:ro -v /tmp/out:/output \
#     my_agent_3:smoke \
#     forecast --panels /input --text /input/text --asof 2012-11-30 \
#              --out /output/forecast.parquet

FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version=2.0
LABEL qfbench2.track=forecasting
LABEL qfbench2.verb=forecast

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Pinned to the exact combination the agent passed its local smoke and gate
# runs with (see my_agent_3/testing/). `qfbench2-common` is required because
# statistical_layer/model_selection.py imports `qfbench2_common.scoring` and
# `qfbench2_track_forecasting.tail` at module load, and main.py imports
# model_selection unconditionally; an image without it builds cleanly and then
# dies on the first forecast call. Same tarball form as the reference image
# (no git client in the base image).
RUN pip install --no-cache-dir \
        "numpy==2.5.3" \
        "pandas==3.0.6" \
        "pyarrow==25.0.1" \
        "scipy==1.18.1" \
        "statsmodels==0.15.0" \
        "arch==8.0.0" \
        "jsonschema==4.23.0" \
        "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.2.tar.gz#subdirectory=common"

WORKDIR /work
COPY my_agent_3 /opt/my_agent_3
COPY qfbench2_track_forecasting /opt/qfbench2_track_forecasting
# forecast.sh derives PYTHONPATH itself; this makes imports work for direct
# `python3 /opt/my_agent_3/main.py` debugging too.
ENV PYTHONPATH=/opt:/opt/my_agent_3

RUN printf '#!/bin/sh\nexec /opt/my_agent_3/forecast.sh "$@"\n' \
      > /usr/local/bin/forecast \
 && chmod +x /usr/local/bin/forecast

# The harness mounts /input read-only and /output writable; nothing here needs
# to write anywhere else.
RUN useradd --create-home --uid 1000 runner
USER runner

CMD ["forecast", "--help"]
