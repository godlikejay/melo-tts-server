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

RUN python - <<'PYTHON'
from melo.api import TTS
from melo import utils

sample_text = {
    'ZH': '你好，世界。',
    'JP': 'こんにちは世界。',
    'KR': '안녕하세요 세계.',
}

for lang, text in sample_text.items():
    model = TTS(language=lang, device='cpu')
    utils.get_text_for_tts_infer(
        text=text,
        language_str=model.language,
        hps=model.hps,
        device=model.device,
        symbol_to_id=model.symbol_to_id,
    )
PYTHON

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

COPY pool.py server.py /app/

CMD ["python", "server.py"]
