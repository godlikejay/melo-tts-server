# melo-tts

![Docker Publish](https://github.com/godlikejay/melo-tts-server/actions/workflows/docker-publish.yml/badge.svg)

基于 [Melo-TTS](https://github.com/myshell-ai/MeloTTS) ，修正相关依赖，并提供 HTTP server 实现。

由于 melo 的实现每个语言为单独一个模型，同时提供多语言服务时容易 OOM，所以增加进程池管理，可通过环境变量 `MAX_LANGUAGES` 来指定内存最多可载入的语言模型数量。

## Build

```shell
docker build -t melo-tts .
```

## Run

```shell
docker run --rm -d --name melo-tts -p 8080:8080 melo-tts:latest
```

## API

```shell
# Health
curl http://localhost:8080/health

# Synthesize
curl -X POST http://localhost:8080/synthesize   -F lang=ZH   -F text='银行' -F speed=0.8 | sox -t wav - -d
```