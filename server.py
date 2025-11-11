# server.py
import atexit
import os
import tempfile
import uuid
from typing import Optional, Tuple

from flask import Flask, after_this_request, jsonify, request, send_file

import compat  # noqa: F401  # ensures importlib.metadata patch
from pool import LanguageProcessPool

MAX_LANGUAGES = int(os.getenv("MAX_LANGUAGES", "1"))
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cpu")
SYNTH_TIMEOUT = float(os.getenv("SYNTH_TIMEOUT", "300"))
TMP_DIR = os.getenv("TTS_TMP_DIR", os.path.join(tempfile.gettempdir(), "melo-tts"))
os.makedirs(TMP_DIR, exist_ok=True)

app = Flask(__name__)
pool = LanguageProcessPool(
    max_languages=MAX_LANGUAGES,
    default_device=DEFAULT_DEVICE,
    start_method="spawn",
)

LANG_MAP = {
    "zh": "ZH",
    "ko": "KR",
    "ja": "JP",
    "es": "ES",
    "en": "EN",
    "fr": "FR",
}

EN_REGION_TO_SPK = {
    "US": "EN-US",
    "CA": "EN-Default",
    "GB": "EN-Default",
    "UK": "EN-Default",
    "AU": "EN-AU",
    "NZ": "EN-AU",
    "IN": "EN_INDIA",
    "BR": "EN-BR",
}


def parse_lang_tag(raw_lang: str) -> Tuple[str, Optional[str]]:
    if not raw_lang:
        return "", None
    cleaned = raw_lang.strip().replace("_", "-")
    parts = [p for p in cleaned.split("-") if p]
    if not parts:
        return "", None
    base = parts[0].lower()
    region = None
    for part in parts[1:]:
        subtag = part.strip()
        if len(subtag) in (2, 3) and subtag.isalpha():
            region = subtag.upper()
            break
    language = LANG_MAP.get(base, base.upper())
    return language, region


def infer_speaker_label(language: str, region: Optional[str]) -> Optional[str]:
    if language != "EN" or not region:
        return None
    return EN_REGION_TO_SPK.get(region.upper())


@app.route("/health", methods=["GET"])
def health():
    """健康检查接口"""
    return jsonify(
        {
            "status": "ok",
            "max_languages": MAX_LANGUAGES,
            "default_device": DEFAULT_DEVICE,
            "tmp_dir": TMP_DIR,
        }
    )


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    表单参数：
      - lang: 必填，如 JP / ZH / EN ...
      - text: 必填，要合成的文本
      - spk:  可选，说话人索引，默认 0
      - speed: 可选，语速(浮点)，默认 1.0
      - timeout: 可选，超时(秒)，默认取 SYNTH_TIMEOUT
      - context_threshold: 可选，音节低于该值时启用加前后缀合成后裁剪方案
      - context_prefix: 可选，短文本补句前缀
      - context_suffix: 可选，短文本补句后缀
      - context_pause:  可选，目标词前后附加的 blank 个数
    返回：audio/wav 二进制
    """
    form = request.form
    raw_lang = form.get("lang", "").strip()
    lang, region = parse_lang_tag(raw_lang)
    text = form.get("text", "").strip()
    spk_field = form.get("spk", None)
    spk = None
    if spk_field is not None and spk_field.strip() != "":
        spk = int(spk_field)
    speed = float(form.get("speed", 1.0))
    timeout = float(form.get("timeout", SYNTH_TIMEOUT))
    context_prefix = form.get("context_prefix", None)
    if context_prefix is not None:
        context_prefix = context_prefix.strip()
    context_suffix = form.get("context_suffix", None)
    if context_suffix is not None:
        context_suffix = context_suffix.strip()
    context_pause = form.get("context_pause", None)
    if context_pause is not None and context_pause.strip() != "":
        try:
            context_pause = int(context_pause)
        except ValueError:
            return ("context_pause must be an integer", 400)
    else:
        context_pause = None
    context_threshold = form.get("context_threshold", None)
    if context_threshold is not None and context_threshold.strip() != "":
        try:
            context_threshold = int(context_threshold)
        except ValueError:
            return ("context_threshold must be an integer", 400)
    else:
        context_threshold = None

    if not lang:
        return ("lang is required", 400)
    if not text:
        return ("text is required", 400)

    spk_label = None
    if spk is None:
        spk_label = infer_speaker_label(lang, region)

    out_path = os.path.join(TMP_DIR, f"{lang.lower()}_{uuid.uuid4().hex}.wav")
    ok, result = pool.synthesize(
        language=lang,
        text=text,
        spk=spk,
        spk_label=spk_label,
        out_path=out_path,
        speed=speed,
        device=DEFAULT_DEVICE,
        timeout=timeout,
        context_prefix=context_prefix,
        context_suffix=context_suffix,
        context_pause=context_pause,
        context_threshold=context_threshold,
    )

    if not ok:
        try:
            if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
                os.remove(out_path)
        except Exception:
            pass
        return (f"synthesize failed: {result}", 500)

    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(result):
                os.remove(result)
        except Exception:
            pass
        return response

    download_tag = str(spk) if spk is not None else (spk_label or "auto")

    return send_file(
        result,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"{lang.lower()}_{download_tag}.wav",
        etag=False,
        conditional=False,
        last_modified=None,
    )


@atexit.register
def _cleanup():
    try:
        pool.close()
    except Exception:
        pass


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port=int(os.getenv("PORT", "8080")), threaded=False, processes=1
    )
