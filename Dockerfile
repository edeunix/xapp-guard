# =====================================================================
# xApp-Guard reproducible simulation environment
# ---------------------------------------------------------------------
# A single, self-contained image that runs the full xApp-Guard
# evaluation (detection accuracy + scaling) on any machine, with no
# software-defined-radio hardware and no GPU.
#
# Build:  docker build -t xapp-guard .
# Run:    docker run --rm -v "$PWD/results:/app/results" xapp-guard
# =====================================================================
FROM python:3.12-slim

LABEL org.opencontainers.image.title="xApp-Guard simulation"
LABEL org.opencontainers.image.description="Trust-aware xApp \
orchestration -- reproducible behavioural-detection evaluation"

WORKDIR /app

# --- python dependencies --------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- project source -------------------------------------------------
COPY xapp_guard/ ./xapp_guard/
COPY sim/        ./sim/
COPY scripts/    ./scripts/
COPY run_all.sh  ./run_all.sh
RUN chmod +x run_all.sh

# results are written here; mount a host directory to keep them
RUN mkdir -p /app/results
VOLUME ["/app/results"]

# default: run the whole evaluation pipeline
CMD ["./run_all.sh"]
