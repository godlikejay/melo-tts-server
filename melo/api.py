import re

import numpy as np
import soundfile
import torch
import torch.nn as nn
from tqdm import tqdm

from . import commons, utils
from .download_utils import load_or_download_config, load_or_download_model
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .text import cleaned_text_to_sequence
from .text.cleaner import clean_text


class TTS(nn.Module):
    SHORT_JP_PH_THRESHOLD = 5
    SHORT_JP_MIN_PH = 3
    SHORT_JP_MAX_BLANK_SEC = 0.05
    SHORT_JP_MIN_PHONE_SEC = 0.08
    SHORT_JP_MIN_PHONE_SEC_EXTRA = 0.12
    SHORT_JP_CONTEXT_PREFIX_TEXT = "これは「"
    SHORT_JP_CONTEXT_SUFFIX_TEXT = "」です。"
    SHORT_JP_CONTEXT_PAUSE_BLANKS = 2

    def __init__(
        self, language, device="auto", use_hf=True, config_path=None, ckpt_path=None
    ):
        super().__init__()
        if device == "auto":
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            if torch.backends.mps.is_available():
                device = "mps"
        if "cuda" in device:
            assert torch.cuda.is_available()

        # config_path =
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device

        # load state_dict
        checkpoint_dict = load_or_download_model(
            language, device, use_hf=use_hf, ckpt_path=ckpt_path
        )
        self.model.load_state_dict(checkpoint_dict["model"], strict=True)

        language = language.split("_")[0]
        self.language = (
            "ZH_MIX_EN" if language == "ZH" else language
        )  # we support a ZH_MIX_EN model
        self._phone_seq_cache = {}

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.0):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print("\n".join(texts))
            print(" > ===========================")
        return texts

    def _get_length_scale(self, language, meta, speed):
        base = 1.0 / max(speed, 1e-3)
        return base

    def _get_blank_trim_frames(self, language, meta):
        if language != "JP":
            return None
        ph_count = meta.get("content_phonemes", 0)
        if ph_count <= self.SHORT_JP_PH_THRESHOLD:
            sr = self.hps.data.sampling_rate
            hop = self.hps.data.hop_length
            sec = self.SHORT_JP_MAX_BLANK_SEC
            if ph_count <= self.SHORT_JP_MIN_PH:
                sec *= 0.5
            frames = int(round(sec * sr / hop))
            return max(frames, 1)
        return None

    def _get_min_phone_frames(self, language, meta):
        if language != "JP":
            return None
        ph_count = meta.get("content_phonemes", 0)
        if ph_count <= self.SHORT_JP_PH_THRESHOLD:
            sr = self.hps.data.sampling_rate
            hop = self.hps.data.hop_length
            sec = (
                self.SHORT_JP_MIN_PHONE_SEC_EXTRA
                if ph_count <= self.SHORT_JP_MIN_PH
                else self.SHORT_JP_MIN_PHONE_SEC
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
        self._phone_seq_cache[key] = phone_ids
        return phone_ids

    @staticmethod
    def _count_nonblank(phone_ids):
        return sum(1 for pid in phone_ids if pid != 0)

    def _maybe_apply_context(self, text, language, meta, context_cfg):
        if language != "JP":
            return None
        ph_count = meta.get("content_phonemes", 0)
        threshold = context_cfg.get("threshold", self.SHORT_JP_PH_THRESHOLD)
        if ph_count > threshold:
            return None
        prefix_text = context_cfg.get("prefix")
        suffix_text = context_cfg.get("suffix")
        pause_blanks = max(0, int(context_cfg.get("pause", 0)))
        if not (prefix_text or suffix_text):
            return None
        target_ids = self._get_phone_ids_for_text(text, language)
        prefix_ids = (
            self._get_phone_ids_for_text(prefix_text, language) if prefix_text else []
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

        def _insert_feat(feat, idx):
            if feat is None or feat.numel() == 0:
                return feat
            zeros = feat.new_zeros(feat.size(0), 1)
            return torch.cat([feat[:, :idx], zeros, feat[:, idx:]], dim=1)

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
        ctx_prefix = (
            self.SHORT_JP_CONTEXT_PREFIX_TEXT
            if context_prefix is None
            else context_prefix
        )
        ctx_suffix = (
            self.SHORT_JP_CONTEXT_SUFFIX_TEXT
            if context_suffix is None
            else context_suffix
        )
        if context_pause_blanks is None:
            ctx_pause = self.SHORT_JP_CONTEXT_PAUSE_BLANKS
        else:
            try:
                ctx_pause = max(0, int(context_pause_blanks))
            except Exception:
                ctx_pause = self.SHORT_JP_CONTEXT_PAUSE_BLANKS
        if context_threshold is None:
            ctx_threshold = self.SHORT_JP_PH_THRESHOLD
        else:
            try:
                ctx_threshold = max(0, int(context_threshold))
            except Exception:
                ctx_threshold = self.SHORT_JP_PH_THRESHOLD
        context_cfg = {
            "prefix": ctx_prefix,
            "suffix": ctx_suffix,
            "pause": ctx_pause,
            "threshold": ctx_threshold,
        }
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
            device = self.device
            base_inputs = utils.get_text_for_tts_infer(
                t, language, self.hps, device, self.symbol_to_id
            )
            bert, ja_bert, phones, tones, lang_ids, meta = base_inputs
            blank_trim_frames = self._get_blank_trim_frames(language, meta)
            min_phone_frames = self._get_min_phone_frames(language, meta)
            context_strategy = None
            if blank_trim_frames is not None and min_phone_frames is not None:
                context_strategy = self._maybe_apply_context(
                    t, language, meta, context_cfg
                )
                if context_strategy:
                    bert, ja_bert, phones, tones, lang_ids, _ = (
                        utils.get_text_for_tts_infer(
                            context_strategy["text"],
                            language,
                            self.hps,
                            device,
                            self.symbol_to_id,
                        )
                    )
                    phones, tones, lang_ids, bert, ja_bert = self._apply_target_pauses(
                        phones, tones, lang_ids, bert, ja_bert, context_strategy
                    )
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                speakers = torch.LongTensor([speaker_id]).to(device)
                length_scale = self._get_length_scale(language, meta, speed)
                audio_tensor, attn, _, _ = self.model.infer(
                    x_tst,
                    x_tst_lengths,
                    speakers,
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
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
                    tones,
                    lang_ids,
                    bert,
                    ja_bert,
                    x_tst_lengths,
                    speakers,
                    audio_tensor,
                    attn,
                )
                #
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(
            audio_list, sr=self.hps.data.sampling_rate, speed=speed
        )

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(
                    output_path, audio, self.hps.data.sampling_rate, format=format
                )
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
