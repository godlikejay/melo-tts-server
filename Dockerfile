FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    ca-certificates build-essential libsndfile1 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip certifi
ENV SSL_CERT_FILE=/usr/local/lib/python3.9/site-packages/certifi/cacert.pem

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY melo/ /app/melo/

ARG PRELOAD_LANGUAGES="ZH,JP,KR"
ENV PRELOAD_LANGUAGES="${PRELOAD_LANGUAGES}" \
    HF_HOME=/opt/hf_cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    XDG_CACHE_HOME=/opt/hf_cache

RUN mkdir -p /opt/hf_cache && chmod -R 777 /opt/hf_cache

COPY init_models.py /app/
RUN python init_models.py --languages "$PRELOAD_LANGUAGES"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

COPY pool.py server.py melo_ext.py /app/

CMD ["python", "server.py"]
