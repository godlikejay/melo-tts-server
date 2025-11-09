# pool_melotts.py
import gc
import multiprocessing as mp
import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


def _worker_loop(language: str, device: str, in_q: mp.Queue, out_q: mp.Queue):
    """
    每个语言一个常驻进程：仅在启动时加载一次对应语言的 MeloTTS 模型。
    进程内循环消费任务；收到 STOP 命令后清理退出。
    """
    import torch

    from melo.api import TTS

    torch.set_grad_enabled(False)

    # --- 音频有效性检测函数（放子进程内，避免主进程依赖） ---
    def _has_sound(path: str, threshold: float = 1e-4):
        try:
            import soundfile as sf
        except Exception:
            # 缺少依赖则跳过检测，返回 True 以不阻塞业务
            return True, None

        try:
            data, sr = sf.read(path, always_2d=False)
        except Exception as e:
            return False, f"read_error: {repr(e)}"

        try:
            # 转单声道
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)
            # 计算绝对值平均能量
            import numpy as _np

            energy = float(_np.mean(_np.abs(data))) if data.size else 0.0
            return (energy > threshold), energy
        except Exception as e:
            return False, f"energy_error: {repr(e)}"

    try:
        with torch.inference_mode():
            tts = TTS(language=language, device=device)
    except Exception as e:
        out_q.put(("__init__", False, f"Init error for {language}: {repr(e)}"))
        return

    out_q.put(("__init__", True, f"{language} ready on {device}"))

    while True:
        msg = in_q.get()
        if msg is None:
            break

        cmd = msg.get("cmd")
        if cmd == "STOP":
            break

        if cmd == "SYNTH":
            job_id = msg["id"]
            text = msg["text"]
            spk = msg["spk"]
            out = msg["out"]
            speed = msg.get("speed", 1.0)
            # 可选：自定义阈值；未提供则使用默认值
            validate_threshold = float(msg.get("validate_threshold", 1e-4))
            context_prefix = msg.get("context_prefix")
            context_suffix = msg.get("context_suffix")
            context_pause = msg.get("context_pause")
            context_threshold = msg.get("context_threshold")
            try:
                if context_pause is not None:
                    context_pause = int(context_pause)
            except Exception:
                context_pause = None
            try:
                if context_threshold is not None:
                    context_threshold = int(context_threshold)
            except Exception:
                context_threshold = None

            if spk is None:
                spk = tts.hps.data.spk2id[language]

            try:
                with torch.inference_mode():
                    tts.tts_to_file(
                        text,
                        spk,
                        out,
                        speed__=speed,
                        context_prefix=context_prefix,
                        context_suffix=context_suffix,
                        context_pause_blanks=context_pause,
                        context_threshold=context_threshold,
                    )  # 兼容旧版/新版参数名
            except TypeError:
                # 某些版本参数名为 speed 而非 speed__
                try:
                    with torch.inference_mode():
                        tts.tts_to_file(
                            text,
                            spk,
                            out,
                            speed=speed,
                            context_prefix=context_prefix,
                            context_suffix=context_suffix,
                            context_pause_blanks=context_pause,
                            context_threshold=context_threshold,
                        )
                except Exception as e:
                    out_q.put((job_id, False, repr(e)))
                    continue
            except Exception as e:
                out_q.put((job_id, False, repr(e)))
                continue

            # --- 合成后音频有效性检测 ---
            try:
                ok, detail = _has_sound(out, threshold=validate_threshold)
                if not ok:
                    # 失败：删除无效文件并返回错误
                    try:
                        import os

                        if os.path.exists(out):
                            os.remove(out)
                    except Exception:
                        pass

                    # detail 可能为能量值或错误字符串
                    if isinstance(detail, (int, float)):
                        out_q.put(
                            (
                                job_id,
                                False,
                                f"Silent/invalid audio, energy={detail:.6g}",
                            )
                        )
                    else:
                        out_q.put((job_id, False, f"Silent/invalid audio: {detail}"))
                    continue
            except Exception as e:
                # 检测流程异常则视为失败（更安全）
                try:
                    import os

                    if os.path.exists(out):
                        os.remove(out)
                except Exception:
                    pass
                out_q.put((job_id, False, f"Validation error: {repr(e)}"))
                continue

            # 通过校验
            out_q.put((job_id, True, out))

        else:
            out_q.put((msg.get("id", "unknown"), False, f"Unknown cmd: {cmd}"))

    # --- 清理 ---
    try:
        del tts
        gc.collect()
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
            if hasattr(_torch, "mps") and _torch.backends.mps.is_available():
                _torch.mps.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


@dataclass
class _LangProc:
    language: str
    device: str
    in_q: mp.Queue
    out_q: mp.Queue
    proc: mp.Process
    last_used_tick: int = field(default=0)


class LanguageProcessPool:
    """
    语言维度的进程池（LRU 淘汰）：
      - 池里每个 entry = 某语言的常驻子进程
      - max_languages 控制最多同时驻留的语言数
      - synthesize(...) 同步提交任务并等待子进程返回
    """

    def __init__(
        self,
        max_languages: int = 1,
        default_device: str = "cpu",
        start_method: Optional[str] = "spawn",
    ):
        if start_method:
            try:
                mp.set_start_method(start_method, force=True)
            except RuntimeError:
                pass
        self.max_languages = max(1, int(max_languages))
        self.default_device = default_device
        self._pool: Dict[str, _LangProc] = {}
        self._tick = 0  # 用于 LRU

    def synthesize(
        self,
        language: str,
        text: str,
        out_path: str,
        spk: int = None,
        speed: float = 1.0,
        device: Optional[str] = None,
        timeout: Optional[float] = None,
        # 可选：按需覆盖检测阈值
        validate_threshold: Optional[float] = None,
        context_prefix: Optional[str] = None,
        context_suffix: Optional[str] = None,
        context_pause: Optional[int] = None,
        context_threshold: Optional[int] = None,
    ) -> Tuple[bool, str]:
        lang = language
        dev = device or self.default_device

        worker = self._ensure_worker(lang, dev)
        if worker is None:
            return False, f"Failed to start worker for {lang}"

        self._tick += 1
        worker.last_used_tick = self._tick

        job_id = str(uuid.uuid4())
        msg = {
            "cmd": "SYNTH",
            "id": job_id,
            "text": text,
            "spk": spk,
            "out": out_path,
            "speed": speed,
        }
        if validate_threshold is not None:
            msg["validate_threshold"] = float(validate_threshold)
        if context_prefix is not None:
            msg["context_prefix"] = context_prefix
        if context_suffix is not None:
            msg["context_suffix"] = context_suffix
        if context_pause is not None:
            msg["context_pause"] = context_pause
        if context_threshold is not None:
            msg["context_threshold"] = context_threshold

        worker.in_q.put(msg)

        start_t = time.time()
        while True:
            try:
                jid, ok, payload = worker.out_q.get(timeout=0.1)
            except queue.Empty:
                if timeout is not None and (time.time() - start_t) > timeout:
                    return False, f"Timeout waiting for synth result: {lang}"
                if not worker.proc.is_alive():
                    return False, f"Worker for {lang} died unexpectedly."
                continue

            if jid == job_id:
                return (ok, payload)

    def close(self):
        for lang in list(self._pool.keys()):
            self._stop_worker(lang)
        self._pool.clear()

    def _ensure_worker(self, language: str, device: str) -> Optional[_LangProc]:
        w = self._pool.get(language)
        if w and w.proc.is_alive():
            return w

        if len(self._pool) >= self.max_languages:
            self._evict_one(language_to_keep=language)

        in_q = mp.Queue()
        out_q = mp.Queue()
        p = mp.Process(
            target=_worker_loop, args=(language, device, in_q, out_q), daemon=True
        )
        p.start()

        ok = False
        init_msg = ""
        try:
            init_jid, ok, init_msg = out_q.get(timeout=120.0)
        except Exception as e:
            init_msg = f"Init wait error: {repr(e)}"

        if not ok:
            try:
                if p.is_alive():
                    in_q.put({"cmd": "STOP"})
                    time.sleep(0.2)
                    p.terminate()
            except Exception:
                pass
            return None

        wrapper = _LangProc(
            language=language, device=device, in_q=in_q, out_q=out_q, proc=p
        )
        self._pool[language] = wrapper
        return wrapper

    def _evict_one(self, language_to_keep: Optional[str] = None):
        """
        按 LRU 淘汰一个进程（不淘汰 language_to_keep）。
        """
        candidates = [w for lang, w in self._pool.items() if lang != language_to_keep]
        if not candidates:
            if language_to_keep in self._pool:
                self._stop_worker(language_to_keep)
                self._pool.pop(language_to_keep, None)
            return

        victim = min(candidates, key=lambda w: w.last_used_tick)
        self._stop_worker(victim.language)
        self._pool.pop(victim.language, None)

    def _stop_worker(self, language: str):
        w = self._pool.get(language)
        if not w:
            return
        try:
            if w.proc.is_alive():
                w.in_q.put({"cmd": "STOP"})
                w.proc.join(timeout=5.0)
                if w.proc.is_alive():
                    w.proc.terminate()
        except Exception:
            try:
                if w.proc.is_alive():
                    w.proc.terminate()
            except Exception:
                pass
