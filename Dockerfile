FROM python:3.9-slim
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

RUN . /etc/os-release; \
    codename="${VERSION_CODENAME:-bookworm}"; \
    echo "Using Debian codename: $codename"; \
    sed -ri 's#http(s)?://deb.debian.org/debian#http://mirrors.aliyun.com/debian#g' /etc/apt/sources.list || true; \
    sed -ri 's#http(s)?://security.debian.org/debian-security#http://mirrors.aliyun.com/debian-security#g' /etc/apt/sources.list || true; \
    printf 'deb http://mirrors.aliyun.com/debian %s main contrib non-free non-free-firmware\n' "$codename" >  /etc/apt/sources.list; \
    printf 'deb http://mirrors.aliyun.com/debian %s-updates main contrib non-free non-free-firmware\n' "$codename" >> /etc/apt/sources.list; \
    printf 'deb http://mirrors.aliyun.com/debian-security %s-security main contrib non-free non-free-firmware\n' "$codename" >> /etc/apt/sources.list; \
    apt-get -y update

RUN	apt-get update && apt-get install -y \
    ca-certificates build-essential libsndfile1 \
	&& update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip certifi
ENV SSL_CERT_FILE=/usr/local/lib/python3.9/site-packages/certifi/cacert.pem

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY melo/ /app/melo/
RUN python - <<EOF
from melo.api import TTS
for lang in ['ZH', 'JP', 'KR']:
    model = TTS(language=lang, device='cpu')
    model.tts_to_file('test', model.hps.data.spk2id[lang], '/tmp/tmp.wav', speed=1.0)
EOF
RUN rm /tmp/tmp.wav

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

COPY pool.py server.py /app

CMD ["python", "server.py"]