import re
from typing import Any, Dict, Optional, Tuple

import compat  # noqa: F401  # ensures importlib.metadata patch
import soundfile
import torch
from tqdm import tqdm

from melo import commons, utils
from melo.api import TTS as BaseTTS
from melo.models import SynthesizerTrn
from melo.text import cleaned_text_to_sequence
from melo.text.cleaner import clean_text


def get_text_for_tts_infer_with_meta(
    text, language_str, hps, device, symbol_to_id=None
):
    """
    包装 melo.utils.get_text_for_tts_infer，额外返回 meta 信息，避免直接改动 melo 包。
    """
    norm_text, phone, tone, _ = clean_text(text, language_str)
    content_phonemes = sum(1 for p in phone if p != "_")
    bert, ja_bert, phone_ids, tone_ids, lang_ids = utils.get_text_for_tts_infer(
        text, language_str, hps, device, symbol_to_id
    )
    meta = {"norm_text": norm_text, "content_phonemes": content_phonemes}
    return bert, ja_bert, phone_ids, tone_ids, lang_ids, meta


def _patch_synthesizer_infer():
    """
    在运行时为 SynthesizerTrn 增加 blank_trim_frames / min_phone_frames 支持，
    避免直接修改 melo/models.py。
    """

    if getattr(SynthesizerTrn, "_ext_patched", False):
        return

    def infer(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
        y=None,
        g=None,
        phone_ids=None,
        blank_trim_frames=None,
        min_phone_frames=None,
    ):
        if g is None:
            if self.n_speakers > 0:
                g = self.emb_g(sid).unsqueeze(-1)
            else:
                g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        if self.use_vc:
            g_p = None
        else:
            g_p = g
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g_p
        )
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        w = torch.exp(logw) * x_mask * length_scale
        if min_phone_frames is not None and phone_ids is not None:
            phone_ids_tensor = phone_ids
            if phone_ids_tensor.dim() == 2:
                phone_ids_tensor = phone_ids_tensor.unsqueeze(1)
            phone_ids_tensor = phone_ids_tensor[..., : w.size(-1)]
            nonblank_mask = (phone_ids_tensor != 0).to(w.dtype)
            min_tensor = torch.full_like(w, float(min_phone_frames))
            w = torch.where(
                nonblank_mask > 0,
                torch.maximum(w, min_tensor),
                w,
            )

        w_ceil = torch.ceil(w)
        if blank_trim_frames is not None and phone_ids is not None:
            trim = max(int(blank_trim_frames), 1)
            phone_ids_tensor = phone_ids
            if phone_ids_tensor.dim() == 2:
                phone_ids_tensor = phone_ids_tensor.unsqueeze(1)
            phone_ids_tensor = phone_ids_tensor[..., : w_ceil.size(-1)]
            start_blank = phone_ids_tensor[..., 0] == 0
            end_blank = phone_ids_tensor[..., -1] == 0
            trim_tensor = torch.full_like(w_ceil[..., 0], trim)
            w_ceil[..., 0] = torch.where(
                start_blank, torch.minimum(w_ceil[..., 0], trim_tensor), w_ceil[..., 0]
            )
            w_ceil[..., -1] = torch.where(
                end_blank, torch.minimum(w_ceil[..., -1], trim_tensor), w_ceil[..., -1]
            )
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    SynthesizerTrn.infer = infer
    SynthesizerTrn._ext_patched = True


_patch_synthesizer_infer()


class MeloTTS(BaseTTS):
    EXT_CONFIG = {
        "JP": {
            "threshold": 5,
            "min_ph": 3,
            "max_blank_sec": 0.05,
            "min_phone_sec": 0.08,
            "min_phone_sec_extra": 0.12,
            "prefix": "これは「",
            "suffix": "」です。",
            "pause_blanks": 3,
        },
        "ZH": {
            "threshold": 2,
            "min_ph": 3,
            "max_blank_sec": 0.05,
            "min_phone_sec": 0.08,
            "min_phone_sec_extra": 0.12,
            "prefix": "这个词读：“",
            "suffix": "。”",
            "pause_blanks": 5,
        },
        "KR": {
            "threshold": 2,
            "min_ph": 3,
            "max_blank_sec": 0.05,
            "min_phone_sec": 0.08,
            "min_phone_sec_extra": 0.12,
            "prefix": '이것은 "',
            "suffix": '" 입니다.',
            "pause_blanks": 3,
        },
    }

    def __init__(
        self, language, device="auto", use_hf=True, config_path=None, ckpt_path=None
    ):
        super().__init__(
            language=language,
            device=device,
            use_hf=use_hf,
            config_path=config_path,
            ckpt_path=ckpt_path,
        )
        self._phone_seq_cache: Dict[Tuple[str, str], Tuple[int, ...]] = {}

    def _get_context_cfg(self, language: str) -> Optional[Dict[str, Any]]:
        if not language:
            return None
        cfg = self.EXT_CONFIG.get(language)
        if cfg is None and "_" in language:
            base_lang = language.split("_")[0]
            cfg = self.EXT_CONFIG.get(base_lang)
        if cfg is None:
            return None
        return cfg.copy()

    def _get_length_scale(self, _language, _meta, speed):
        return 1.0 / max(speed, 1e-3)

    def _get_blank_trim_frames(self, cfg, meta):
        if cfg is None:
            return None
        ph_count = meta.get("content_phonemes", 0)
        threshold = cfg.get("threshold", 0)
        if ph_count <= threshold:
            sr = self.hps.data.sampling_rate
            hop = self.hps.data.hop_length
            sec = cfg.get("max_blank_sec", 0.0)
            if ph_count <= cfg.get("min_ph", 0):
                sec *= 0.5
            frames = int(round(sec * sr / hop))
            return max(frames, 1)
        return None

    def _get_min_phone_frames(self, cfg, meta):
        if cfg is None:
            return None
        ph_count = meta.get("content_phonemes", 0)
        threshold = cfg.get("threshold", 0)
        if ph_count <= threshold:
            sr = self.hps.data.sampling_rate
            hop = self.hps.data.hop_length
            sec = (
                cfg.get("min_phone_sec_extra", cfg.get("min_phone_sec", 0.0))
                if ph_count <= cfg.get("min_ph", 0)
                else cfg.get("min_phone_sec", 0.0)
            )
            frames = int(round(sec * sr / hop))
            return max(frames, 1)
        return None

    def _get_phone_ids_for_text(self, text, language):
        key = (language, text)
        cached = self._phone_seq_cache.get(key)
        if cached is not None:
            return cached
        norm_text, phone, tone, _ = clean_text(text, language)
        phone_ids, _, _ = cleaned_text_to_sequence(
            phone, tone, language, self.symbol_to_id
        )
        if getattr(self.hps.data, "add_blank", False):
            phone_ids = commons.intersperse(phone_ids, 0)
        self._phone_seq_cache[key] = tuple(phone_ids)
        return tuple(phone_ids)

    @staticmethod
    def _count_nonblank(phone_ids):
        return sum(1 for pid in phone_ids if pid != 0)

    def _maybe_apply_context(self, text, language, meta, context_cfg):
        if context_cfg is None:
            return None
        ph_count = meta.get("content_phonemes", 0)
        threshold = context_cfg.get("threshold", 0)
        if ph_count > threshold:
            return None
        prefix_text = context_cfg.get("prefix")
        suffix_text = context_cfg.get("suffix")
        pause_value = context_cfg.get("pause")
        if pause_value is None:
            pause_value = context_cfg.get("pause_blanks", 0)
        try:
            pause_blanks = max(0, int(pause_value))
        except Exception:
            pause_blanks = 0
        if not (prefix_text or suffix_text):
            return None
        target_ids = self._get_phone_ids_for_text(text, language)
        prefix_ids = (
            self._get_phone_ids_for_text(prefix_text, language) if prefix_text else ()
        )
        context_text = f"{prefix_text or ''}{text}{suffix_text or ''}"
        return {
            "text": context_text,
            "prefix_nonblank": self._count_nonblank(prefix_ids),
            "target_nonblank": self._count_nonblank(target_ids),
            "pause_blanks": pause_blanks,
        }

    def _apply_target_pauses(self, phones, tones, lang_ids, bert, ja_bert, strategy):
        if not strategy or strategy.get("pause_blanks", 0) <= 0:
            return phones, tones, lang_ids, bert, ja_bert
        try:
            phone_list = phones.tolist()
            tone_list = tones.tolist()
            lang_list = lang_ids.tolist()
        except Exception:
            return phones, tones, lang_ids, bert, ja_bert
        nonblank_positions = [idx for idx, pid in enumerate(phone_list) if pid != 0]
        prefix_nb = strategy.get("prefix_nonblank", 0)
        target_nb = strategy.get("target_nonblank", 0)
        if target_nb <= 0 or prefix_nb + target_nb > len(nonblank_positions):
            return phones, tones, lang_ids, bert, ja_bert
        pause = strategy.get("pause_blanks", 0)
        start_idx = nonblank_positions[prefix_nb]
        end_idx = nonblank_positions[prefix_nb + target_nb - 1]

        def _insert(seq, idx, value):
            seq.insert(idx, value)

        def _insert_feat(feat_tensor, idx):
            if feat_tensor is None or feat_tensor.numel() == 0:
                return feat_tensor
            zeros = feat_tensor.new_zeros(feat_tensor.size(0), 1)
            return torch.cat([feat_tensor[:, :idx], zeros, feat_tensor[:, idx:]], dim=1)

        for _ in range(pause):
            _insert(phone_list, start_idx, 0)
            _insert(tone_list, start_idx, 0)
            _insert(lang_list, start_idx, 0)
            bert = _insert_feat(bert, start_idx)
            ja_bert = _insert_feat(ja_bert, start_idx)
            end_idx += 1

        insert_pos = end_idx + 1
        for _ in range(pause):
            _insert(phone_list, insert_pos, 0)
            _insert(tone_list, insert_pos, 0)
            _insert(lang_list, insert_pos, 0)
            bert = _insert_feat(bert, insert_pos)
            ja_bert = _insert_feat(ja_bert, insert_pos)
            insert_pos += 1

        device = phones.device
        phones = torch.tensor(phone_list, dtype=phones.dtype, device=device)
        tones = torch.tensor(tone_list, dtype=tones.dtype, device=tones.device)
        lang_ids = torch.tensor(lang_list, dtype=lang_ids.dtype, device=lang_ids.device)
        return phones, tones, lang_ids, bert, ja_bert

    def _trim_context_audio(self, audio, attn, phone_ids, strategy):
        if strategy is None:
            return audio
        try:
            phone_seq = phone_ids.squeeze(0).tolist()
        except Exception:
            return audio
        nonblank_positions = [idx for idx, pid in enumerate(phone_seq) if pid != 0]
        prefix_nb = strategy.get("prefix_nonblank", 0)
        target_nb = strategy.get("target_nonblank", 0)
        if target_nb <= 0 or prefix_nb + target_nb > len(nonblank_positions):
            return audio
        pause_blanks = strategy.get("pause_blanks", 0)
        start_nb = prefix_nb
        end_nb = prefix_nb + target_nb - 1
        start_idx = nonblank_positions[start_nb]
        end_idx = nonblank_positions[end_nb] + 1
        attn_path = (attn.squeeze(0).squeeze(0) > 0).to(torch.float32)
        per_phone_frames = attn_path.sum(dim=0)
        start_sum_idx = max(0, start_idx - pause_blanks)
        end_sum_idx = min(per_phone_frames.size(0), end_idx + pause_blanks)
        start_frames = int(per_phone_frames[:start_sum_idx].sum().item())
        end_frames = int(per_phone_frames[:end_sum_idx].sum().item())
        hop = self.hps.data.hop_length
        start_sample = max(0, start_frames * hop)
        end_sample = min(audio.size(-1), max(start_sample + hop, end_frames * hop))
        return audio[..., start_sample:end_sample]

    def _prepare_inputs_with_meta(self, text, language):
        return get_text_for_tts_infer_with_meta(
            text, language, self.hps, self.device, self.symbol_to_id
        )

    def tts_to_file(
        self,
        text,
        speaker_id,
        output_path=None,
        sdp_ratio=0.2,
        noise_scale=0.6,
        noise_scale_w=0.8,
        speed=1.0,
        pbar=None,
        format=None,
        position=None,
        quiet=False,
        context_prefix=None,
        context_suffix=None,
        context_pause_blanks=None,
        context_threshold=None,
    ):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        base_cfg = self._get_context_cfg(language)
        context_cfg = base_cfg.copy() if base_cfg else None
        if context_cfg:
            if "pause" not in context_cfg and "pause_blanks" in context_cfg:
                context_cfg["pause"] = context_cfg["pause_blanks"]
            if context_prefix is not None:
                context_cfg["prefix"] = context_prefix
            if context_suffix is not None:
                context_cfg["suffix"] = context_suffix
            if context_pause_blanks is not None:
                try:
                    context_cfg["pause"] = max(0, int(context_pause_blanks))
                except Exception:
                    pass
            if context_threshold is not None:
                try:
                    context_cfg["threshold"] = max(0, int(context_threshold))
                except Exception:
                    pass
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ["EN", "ZH_MIX_EN"]:
                t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            (
                bert,
                ja_bert,
                phones,
                tones,
                lang_ids,
                meta,
            ) = self._prepare_inputs_with_meta(t, language)
            blank_trim_frames = self._get_blank_trim_frames(context_cfg, meta)
            min_phone_frames = self._get_min_phone_frames(context_cfg, meta)
            context_strategy = None
            if (
                context_cfg
                and blank_trim_frames is not None
                and min_phone_frames is not None
            ):
                context_strategy = self._maybe_apply_context(
                    t, language, meta, context_cfg
                )
                if context_strategy:
                    (
                        bert,
                        ja_bert,
                        phones,
                        tones,
                        lang_ids,
                        _,
                    ) = self._prepare_inputs_with_meta(
                        context_strategy["text"], language
                    )
                    (
                        phones,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                    ) = self._apply_target_pauses(
                        phones, tones, lang_ids, bert, ja_bert, context_strategy
                    )
            with torch.no_grad():
                x_tst = phones.to(self.device).unsqueeze(0)
                tones_t = tones.to(self.device).unsqueeze(0)
                lang_t = lang_ids.to(self.device).unsqueeze(0)
                bert_t = bert.to(self.device).unsqueeze(0)
                ja_bert_t = ja_bert.to(self.device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(self.device)
                speakers = torch.LongTensor([speaker_id]).to(self.device)
                length_scale = self._get_length_scale(language, meta, speed)
                audio_tensor, attn, _, _ = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones_t,
                    lang_t,
                    bert_t,
                    ja_bert_t,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    phone_ids=x_tst,
                    blank_trim_frames=blank_trim_frames,
                    min_phone_frames=min_phone_frames,
                )
                if context_strategy:
                    audio_tensor = self._trim_context_audio(
                        audio_tensor, attn, x_tst, context_strategy
                    )
                audio = audio_tensor[0][0].data.cpu().float().numpy()
                del (
                    x_tst,
                    tones_t,
                    lang_t,
                    bert_t,
                    ja_bert_t,
                    x_tst_lengths,
                    speakers,
                    audio_tensor,
                    attn,
                )
            audio_list.append(audio)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(
            audio_list, sr=self.hps.data.sampling_rate, speed=speed
        )
        if output_path is None:
            return audio
        if format:
            soundfile.write(
                output_path, audio, self.hps.data.sampling_rate, format=format
            )
        else:
            soundfile.write(output_path, audio, self.hps.data.sampling_rate)
