"""Sample authentic monolingual preset sentences from all-lab-text-mono.

    python tools/sample_mono_sentences.py

Reads HF_TOKEN from `.env` (never printed). Uses the local sentence cache when
present, otherwise downloads ONE parquet shard per language and scans the first
row group. Does not call the dataset-viewer API.

Languages with no HF config (English) keep their existing CSV rows.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.config_loader import TestCatalog  # noqa: E402

REPO = "African-Languages-Lab/all-lab-text-mono"
OUT = PROJECT_ROOT / "config" / "test_sentences.csv"
LANGUAGES_YAML = PROJECT_ROOT / "config" / "languages.yaml"
CACHE = PROJECT_ROOT / "data" / "mono_sentence_cache.json"
PARQUET_DIR = PROJECT_ROOT / "data" / "hf_mono"
SEED = 20260904
PER_LANGUAGE = 10
MAX_SENTENCE_CHARS = 600
SCAN_ROWS = 8000

HF_CONFIG: dict[str, str | None] = {
    "chichewa": "chewa",
    "sepedi": "northern sotho",
    "sesotho": "southern sotho",
    "english": None,
}

SCRIPT_HINTS: dict[str, str] = {
    "amharic": r"[\u1200-\u137F]",
    "arabic": r"[\u0600-\u06FF]",
    "tigrinya": r"[\u1200-\u137F]",
    "twi": r"[ɛɔƐƆ]",
}

_ENGLISH_MARKERS = re.compile(
    r"\b(the|because|together|before|after|please|children|today|tomorrow)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_REPEAT_RE = re.compile(r"(.)\1{6,}")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_N_NEWLINE_RE = re.compile(r"(?<=[.!?\"'”’)])n(?=[A-ZƐƆÁÉÍÓÚ\"“'(])")


def _hf_config(language: str) -> str | None:
    if language in HF_CONFIG:
        return HF_CONFIG[language]
    return language


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _load_token() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is missing from .env")
    return token


def _load_languages() -> list[dict[str, str]]:
    raw = yaml.safe_load(LANGUAGES_YAML.read_text(encoding="utf-8"))
    entries = raw.get("languages") if isinstance(raw, dict) else raw
    languages = []
    for entry in entries:
        key = str(entry["key"]).strip().lower()
        languages.append({"key": key, "code": str(entry.get("code") or key[:2]).upper()})
    return languages


def _load_existing() -> dict[str, list[str]]:
    by_language: dict[str, list[str]] = {}
    if not OUT.exists():
        return by_language
    with OUT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            language = (row.get("language") or "").strip().lower()
            text = (row.get("text") or "").strip()
            if language and text:
                by_language.setdefault(language, []).append(text)
    return by_language


def _load_cache(seed: int) -> dict[str, list[str]]:
    if not CACHE.exists():
        return {}
    try:
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if raw.get("seed") != seed or raw.get("repo") != REPO:
        return {}
    sentences = raw.get("sentences") or {}
    return {
        str(key): [str(item) for item in value]
        for key, value in sentences.items()
        if isinstance(value, list) and len(value) == PER_LANGUAGE
    }


def _save_cache(seed: int, sentences: dict[str, list[str]]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "repo": REPO, "sentences": sentences}
    CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize(text: str) -> str:
    text = (text or "").replace("\u00a0", " ").replace("\r", "\n").replace("+", "")
    return " ".join(text.split())


def _spans(text: str) -> list[str]:
    raw = _N_NEWLINE_RE.sub("\n", (text or "").replace("\r", "\n"))
    chunks = [part for part in re.split(r"[\n]+", raw) if part.strip()] or [raw]
    spans: list[str] = []
    for chunk in chunks:
        cleaned = _normalize(chunk)
        if cleaned:
            spans.extend(part.strip(" \"'") for part in _SENTENCE_SPLIT_RE.split(cleaned) if _normalize(part))
    return spans


def _quality(text: str, language: str, min_words: int, max_words: int) -> bool:
    if not text or text.upper().startswith("[PLACEHOLDER]") or "\ufffd" in text:
        return False
    if len(text) < 40 or len(text) > MAX_SENTENCE_CHARS:
        return False
    if text[0].isdigit() or text.startswith("(") or _URL_RE.search(text) or _REPEAT_RE.search(text):
        return False
    if any(marker in text for marker in ("[[", "]]", "##", "{{", ".n")):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < 0.55:
        return False
    words = _word_count(text)
    if words < min_words or words > max_words:
        return False
    hint = SCRIPT_HINTS.get(language)
    if hint and not re.search(hint, text):
        return False
    if language != "english" and len(_ENGLISH_MARKERS.findall(text)) >= 2:
        return False
    return True


def _first_shard_name(token: str, config: str) -> str | None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    prefix = f"{config}/"
    names = [
        item.rfilename
        for item in (api.list_repo_files(REPO, repo_type="dataset") or [])
        if item.startswith(prefix) and item.endswith(".parquet")
    ]
    names.sort()
    return names[0] if names else None


def _download_parquet(token: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=filename,
        token=token,
        local_dir=str(PARQUET_DIR),
    )
    return Path(path)


def _collect_from_parquet(path: Path, language: str, config: str, rng: random.Random) -> list[str]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    names = list(parquet.schema_arrow.names)
    text_col = config if config in names else next(
        (name for name in names if name not in {"dataset_name", "source_data_url"} and not name.endswith("_token_count")),
        names[0],
    )
    columns = [text_col]
    if "dataset_name" in names:
        columns.append("dataset_name")
    group = parquet.read_row_group(0, columns=columns)
    texts = group.column(text_col).to_pylist()[:SCAN_ROWS]
    pool: list[str] = []
    seen: set[str] = set()
    for raw in texts:
        for span in _spans(raw or ""):
            if not _quality(span, language, 12, 40):
                continue
            key = span.casefold()
            if key in seen:
                continue
            seen.add(key)
            pool.append(span)
            if len(pool) >= 40:
                break
        if len(pool) >= 40:
            break
    rng.shuffle(pool)
    picked: list[str] = []
    prefixes: set[str] = set()
    for span in pool:
        prefix = span[:48].casefold()
        if prefix in prefixes:
            continue
        prefixes.add(prefix)
        picked.append(span)
        if len(picked) == PER_LANGUAGE:
            return picked
    return picked


def _write_csv(languages: list[dict[str, str]], texts: dict[str, list[str]]) -> None:
    rows = []
    for language in languages:
        key = language["key"]
        code = language["code"]
        for index, text in enumerate(texts[key], start=1):
            rows.append(
                {
                    "sentence_id": f"{code}-{index:03d}",
                    "language": key,
                    "accent": "",
                    "text": text,
                }
            )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["sentence_id", "language", "accent", "text"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    OUT.write_text(buffer.getvalue(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    token = _load_token()
    languages = _load_languages()
    existing = _load_existing()
    cache = _load_cache(args.seed)
    sampled: dict[str, list[str]] = dict(cache)
    missing: list[str] = []
    failed: list[str] = []
    mapping: list[tuple[str, str, str]] = []

    print(f"dataset: {REPO}")
    print(f"seed: {args.seed}")
    print(f"cached languages: {len(cache)}")

    for language in languages:
        key = language["key"]
        config = _hf_config(key)
        rng = random.Random(f"{args.seed}:{key}")
        if not config:
            lines = existing.get(key, [])[:PER_LANGUAGE]
            sampled[key] = lines
            missing.append(key)
            mapping.append((key, "(none)", f"kept {len(lines)} existing"))
            print(f"{key:16}  no HF config; keeping {len(lines)}")
            continue
        if key in cache:
            mapping.append((key, config, "cached 10"))
            print(f"{key:16}  cached 10")
            continue
        print(f"{key:16}  parquet {config!r} ...", flush=True)
        try:
            filename = _first_shard_name(token, config)
            if not filename:
                raise RuntimeError("no parquet shard")
            path = _download_parquet(token, filename)
            lines = _collect_from_parquet(path, key, config, rng)
        except Exception as exc:
            lines = existing.get(key, [])[:PER_LANGUAGE]
            failed.append(key)
            mapping.append((key, config, f"kept existing ({type(exc).__name__})"))
            print(f"{key:16}  failed {type(exc).__name__}; keeping {len(lines)} existing")
            sampled[key] = lines
            continue
        if len(lines) < PER_LANGUAGE:
            extra = [text for text in existing.get(key, []) if text not in lines]
            lines = (lines + extra)[:PER_LANGUAGE]
            failed.append(f"{key}:{len(lines)}")
        sampled[key] = lines
        if len(lines) == PER_LANGUAGE:
            cache[key] = lines
            _save_cache(args.seed, cache)
        mapping.append((key, config, f"sampled {len(lines)}"))
        print(f"{key:16}  sampled {len(lines)}")

    for language in languages:
        sampled.setdefault(language["key"], existing.get(language["key"], [])[:PER_LANGUAGE])
    _write_csv(languages, sampled)

    print(f"\nwrote {OUT.relative_to(PROJECT_ROOT)}")
    print("\nMapping (app key -> HF config)")
    for key, config, note in mapping:
        print(f"  {key:16}  {config:20}  {note}")
    if missing:
        print("\nMissing from dataset:")
        for key in missing:
            print(f"  - {key}")
    if failed:
        print("\nFailed / incomplete:")
        for item in failed:
            print(f"  - {item}")

    catalog = TestCatalog.load(PROJECT_ROOT / "config")
    print(f"\ncatalog languages: {len(catalog.languages)}")
    print(f"catalog sentences: {len(catalog.sentences)}")
    under = [
        f"{language.key}:{len(catalog.sentences_for(language.key))}"
        for language in catalog.languages
        if len(catalog.sentences_for(language.key)) != PER_LANGUAGE
    ]
    if under:
        print("languages without 10 sentences:", ", ".join(under))
        return 1
    print("catalog: 38 languages, 10 sentences each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
