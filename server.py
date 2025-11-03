# server.py
import atexit
import os
import tempfile
import uuid

from flask import Flask, after_this_request, jsonify, request, send_file

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
    返回：audio/wav 二进制
    """
    form = request.form
    lang = form.get("lang", "").strip()
    text = form.get("text", "").strip()
    spk = form.get("spk", None)
    if spk is not None:
        spk = int(spk)
    speed = float(form.get("speed", 1.0))
    timeout = float(form.get("timeout", SYNTH_TIMEOUT))

    if not lang:
        return ("lang is required", 400)
    if not text:
        return ("text is required", 400)

    out_path = os.path.join(TMP_DIR, f"{lang.lower()}_{uuid.uuid4().hex}.wav")
    ok, result = pool.synthesize(
        language=lang,
        text=text,
        spk=spk,
        out_path=out_path,
        speed=speed,
        device=DEFAULT_DEVICE,
        timeout=timeout,
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

    return send_file(
        result,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"{lang.lower()}_{spk}.wav",
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
