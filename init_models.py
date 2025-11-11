#!/usr/bin/env python3
"""Utility script to pre-download MeloTTS models for selected languages."""

import argparse
import os
from typing import Iterable, List, Optional, Set

import nltk

import compat  # noqa: F401  # ensures importlib.metadata patch
from melo import utils
from melo.api import TTS

DEFAULT_LANGUAGES = ("ZH", "JP", "KR")
DEFAULT_SAMPLE_TEXT = {
    "ZH": "你好，世界。",
    "JP": "こんにちは世界。",
    "KR": "안녕하세요 세계.",
    "EN": "Hello world.",
    "FR": "Bonjour le monde.",
    "ES": "Hola mundo.",
}


def _tokenize_languages(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(DEFAULT_LANGUAGES)
    tokens: List[str] = []
    for token in raw.replace(";", ",").split(","):
        cleaned = token.strip()
        if cleaned:
            tokens.append(cleaned)
    if not tokens:
        return list(DEFAULT_LANGUAGES)
    return tokens


def normalize_language(token: str) -> str:
    upper = token.strip().upper().replace("-", "_")
    if not upper:
        raise ValueError("Empty language token provided")
    if upper.startswith("EN"):
        return "EN"
    if upper in DEFAULT_SAMPLE_TEXT:
        return upper
    raise ValueError(f"Unsupported language token: {token}")


def unique_languages(tokens: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    unique: List[str] = []
    for token in tokens:
        normalized = normalize_language(token)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def warmup_language(language: str) -> None:
    print(f"[init_models] Preparing language: {language}")
    model = TTS(language=language, device="cpu")
    sample_text = DEFAULT_SAMPLE_TEXT.get(language, DEFAULT_SAMPLE_TEXT["EN"])
    utils.get_text_for_tts_infer(
        text=sample_text,
        language_str=model.language,
        hps=model.hps,
        device=model.device,
        symbol_to_id=model.symbol_to_id,
    )
    del model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download MeloTTS model assets")
    parser.add_argument(
        "--languages",
        "-l",
        help=(
            "Comma-separated list of languages (e.g. 'ZH,JP,KR,EN-US,FR'). "
            "Defaults to ZH,JP,KR if omitted. English accents such as US/BR map to EN."
        ),
        default=os.environ.get("PRELOAD_LANGUAGES"),
    )
    args = parser.parse_args()

    tokens = _tokenize_languages(args.languages)
    languages = unique_languages(tokens)
    ensure_nltk_data(languages)
    print(f"[init_models] Languages to prepare: {', '.join(languages)}")
    for language in languages:
        warmup_language(language)
    print("[init_models] Done")


def ensure_nltk_data(languages: Iterable[str]) -> None:
    if "EN" not in languages:
        return
    resource = "averaged_perceptron_tagger_eng"
    try:
        nltk.data.find(f"taggers/{resource}")
    except LookupError:
        print(f"[init_models] Downloading NLTK resource: {resource}")
        nltk.download(resource, quiet=True)


if __name__ == "__main__":
    main()
