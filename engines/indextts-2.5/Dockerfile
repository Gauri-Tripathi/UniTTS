FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir /workspace
RUN python -m pip install --no-cache-dir -r /workspace/engines/indextts-2.5/requirements.lock
ENV UNITTS_ENGINE_ID=indextts-2.5
ENV UNITTS_DEVICE=cuda
ENV UNITTS_CACHE_DIR=/var/cache/unitts
VOLUME ["/var/cache/unitts"]
CMD ["python", "-m", "unitts.services.engine_worker"]
