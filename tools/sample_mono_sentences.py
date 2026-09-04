"""Sample authentic monolingual preset sentences from all-lab-text-mono.

    python tools/sample_mono_sentences.py

Reads HF_TOKEN from the project `.env` (never printed). Pages rows through the
Hugging Face dataset-viewer API so multi-GB shards are not downloaded.

Overwrites config/test_sentences.csv for every evaluation language that has a
matching config in African-Languages-Lab/all-lab-text-mono. Languages with no
config (currently English) keep their existing CSV rows; nothing is invented.
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
import time
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.config_loader import TestCatalog  # noqa: E402

REPO = "African-Languages-Lab/all-lab-text-mono"
VIEWER = "https://datasets-server.huggingface.co"
OUT = PROJECT_ROOT / "config" / "test_sentences.csv"
LANGUAGES_YAML = PROJECT_ROOT / "config" / "languages.yaml"
CACHE = PROJECT_ROOT / "data" / "mono_sentence_cache.json"
PARQUET_DIR = PROJECT_ROOT / "data" / "hf_mono"
SEED = 20260904
PER_LANGUAGE = 10
PAGE_SIZE = 100
MAX_SENTENCE_CHARS = 600
SMALL_PARQUET_BYTES = 120 * 1024 * 1024
API_PAUSE_SECONDS = 0.45
RATE_LIMIT_SLEEP = 25

# App language key -> dataset-viewer config name. Identity if omitted.
# English has no monolingual config in this repo.
HF_CONFIG: dict[str, str | None] = {
    "chichewa": "chewa",
    "sepedi": "northern sotho",
    "sesotho": "southern sotho",
    "english": None,
}

# Require a native-script / orthography cue so English or sister-language
# rows cannot sneak through. Latin-script languages without a stable cue are
# filtered with the generic quality checks only.
SCRIPT_HINTS: dict[str, str] = {
    "amharic": r"[\u1200-\u137F]",
    "arabic": r"[\u0600-\u06FF]",
    "tigrinya": r"[\u1200-\u137F]",
    "twi": r"[ɛɔƐƆ]",
    "ewe": r"[ɖŋƒƉŊ]",
    "yoruba": r"[ẹọṣńáàéèíìóòúùẸỌṢ]",
    "igbo": r"[ịọụṅẹỊỌỤṄẸ]",
    "fon": r"[ɔɖɛƉƆƐ]",
    "berber": r"[ɣɛčḍẓṛţḥɛƐ]",
}

_ENGLISH_MARKERS = re.compile(
    r"\b(the|because|together|before|after|please|children|today|tomorrow|"
    r"between|without|through|during|something|everything)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://|www\.|\.html\b", re.IGNORECASE)
_REPEAT_RE = re.compile(r"(.)\1{6,}")
_TRIPLE_LETTER_RE = re.compile(r"([A-Za-zƐƆɛɔɖŋ])\1{2,}")
_VERSE_RE = re.compile(r"\d+:\d+")
_GLUED_NEWLINE_RE = re.compile(r"(?<=\w)n(?=[A-ZƐƆÁÉÍÓÚ])")
_N_NEWLINE_RE = re.compile(r"(?<=[.!?\"'”’)])n(?=[A-ZƐƆÁÉÍÓÚ\"“'(])")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STRICT_SCRIPT = frozenset({"amharic", "arabic", "tigrinya", "twi"})


def _hf_config(language: str) -> str | None:
    if language in HF_CONFIG:
        return HF_CONFIG[language]
    return language


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _request_json(
    session: requests.Session,
    path: str,
    token: str,
    params: dict[str, str | int],
    timeout: int = 90,
) -> dict:
    for attempt in range(8):
        response = session.get(
            f"{VIEWER}{path}",
            params=params,
            headers=_auth_headers(token),
            timeout=timeout,
        )
        if response.status_code == 429:
            time.sleep(RATE_LIMIT_SLEEP * (1 + attempt // 2))
            continue
        if response.status_code != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status_code}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path} returned a non-object payload")
        return payload
    raise RuntimeError(f"{path} still rate-limited after retries")


def _load_token() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is missing from .env")
    return token


def _load_languages() -> list[dict[str, str]]:
    raw = yaml.safe_load(LANGUAGES_YAML.read_text(encoding="utf-8"))
    entries = raw.get("languages") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise SystemExit("config/languages.yaml has no languages list")
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


def _normalize(text: str) -> str:
    text = (text or "").replace("\u00a0", " ").replace("\r", "\n")
    text = text.replace("+", "")
    return " ".join(text.split())


def _spans(text: str) -> list[str]:
    raw = (text or "").replace("\r", "\n")
    raw = _N_NEWLINE_RE.sub("\n", raw)
    chunks = [part for part in re.split(r"[\n]+", raw) if part.strip()]
    if not chunks:
        chunks = [raw]
    spans: list[str] = []
    for chunk in chunks:
        cleaned = _normalize(chunk)
        if not cleaned:
            continue
        parts = _SENTENCE_SPLIT_RE.split(cleaned)
        spans.extend(part.strip(" \"'") for part in parts if _normalize(part))
    return spans


def _quality(
    text: str,
    *,
    language: str,
    min_words: int,
    max_words: int,
    min_letter_ratio: float,
) -> bool:
    if not text or text.upper().startswith("[PLACEHOLDER]") or "\ufffd" in text:
        return False
    if len(text) < 40 or len(text) > MAX_SENTENCE_CHARS:
        return False
    if text.startswith("(") or text.startswith("[") or text[0].isdigit():
        return False
    if min_words >= 15 and text[0].isascii() and text[0].islower():
        return False
    if " ' " in text or sum(1 for tok in text.split() if len(tok) == 1) >= 3:
        return False
    if _URL_RE.search(text) or _REPEAT_RE.search(text) or _TRIPLE_LETTER_RE.search(text):
        return False
    if _VERSE_RE.search(text) or _GLUED_NEWLINE_RE.search(text):
        return False
    if any(marker in text for marker in ("[[", "]]", "##", "<br", "{{", "}}", ".n", "!n", "?n")):
        return False
    if re.match(r"^[\W\d_]{3,}", text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < min_letter_ratio:
        return False
    if sum(ch.isdigit() for ch in text) > 8:
        return False
    words = _word_count(text)
    if words < min_words or words > max_words:
        return False
    hint = SCRIPT_HINTS.get(language)
    require_hint = language in _STRICT_SCRIPT or min_words >= 15
    if hint and require_hint and not re.search(hint, text):
        return False
    if language != "english" and len(_ENGLISH_MARKERS.findall(text)) >= 2:
        return False
    return True


def _source_rank(name: str) -> int:
    lowered = (name or "").lower()
    if "nllb" in lowered:
        return 3
    if any(tag in lowered for tag in ("madlad", "hplt", "ccaligned", "cc-aligned")):
        return 2
    if any(tag in lowered for tag in ("wiki", "bible", "leipzig", "fineweb", "masakhane")):
        return 0
    return 1


def _offsets(num_rows: int, rng: random.Random, pages: int) -> list[int]:
    if num_rows <= 0:
        return []
    page_size = min(PAGE_SIZE, num_rows)
    if num_rows <= page_size * pages:
        return list(range(0, num_rows, page_size))
    stride = max(page_size, num_rows // pages)
    offsets: list[int] = []
    for index in range(pages):
        base = min(index * stride, num_rows - page_size)
        jitter = rng.randint(0, max(0, min(stride, page_size * 8) - 1))
        offsets.append(max(0, min(base + jitter, num_rows - page_size)))
    return sorted(set(offsets))


def _text_from_row(row: dict, config: str) -> tuple[str, str]:
    payload = row.get("row") if isinstance(row.get("row"), dict) else row
    if not isinstance(payload, dict):
        return "", ""
    text = payload.get(config)
    if not isinstance(text, str):
        text = next((value for key, value in payload.items() if isinstance(value, str) and key not in {"dataset_name", "source_data_url"} and not key.endswith("_token_count")), "")
    source = payload.get("dataset_name") if isinstance(payload.get("dataset_name"), str) else ""
    return text or "", source or ""


def _num_rows(session: requests.Session, token: str, config: str) -> int:
    payload = _request_json(
        session,
        "/size",
        token,
        {"dataset": REPO, "config": config, "split": "train"},
    )
    size = payload.get("size") or {}
    config_size = size.get("config") if isinstance(size, dict) else {}
    rows = (config_size or {}).get("num_rows")
    if not rows:
        splits = size.get("splits") if isinstance(size, dict) else None
        if isinstance(splits, list) and splits:
            rows = splits[0].get("num_rows")
    return int(rows or 0)


def _fetch_page(
    session: requests.Session,
    token: str,
    config: str,
    offset: int,
    length: int,
) -> list[dict]:
    payload = _request_json(
        session,
        "/rows",
        token,
        {
            "dataset": REPO,
            "config": config,
            "split": "train",
            "offset": offset,
            "length": length,
        },
    )
    rows = payload.get("rows") or []
    return rows if isinstance(rows, list) else []


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


def _small_shards(token: str) -> dict[str, str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.dataset_info(REPO, files_metadata=True)
    first: dict[str, tuple[str, int]] = {}
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if not name.endswith(".parquet"):
            continue
        folder = name.split("/", 1)[0]
        if folder in first:
            continue
        first[folder] = (name, int(getattr(sibling, "size", 0) or 0))
    return {
        folder: path
        for folder, (path, size) in first.items()
        if 0 < size <= SMALL_PARQUET_BYTES
    }


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


def _iter_parquet_rows(path: Path, config: str):
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
    for index in range(parquet.num_row_groups):
        group = parquet.read_row_group(index, columns=columns)
        texts = group.column(text_col).to_pylist()
        sources = group.column("dataset_name").to_pylist() if "dataset_name" in columns else [""] * len(texts)
        for text, source in zip(texts, sources):
            yield text or "", source or ""


def _pick(pool: list[tuple[int, str]], rng: random.Random) -> list[str]:
    by_rank: dict[int, list[str]] = {}
    for rank, span in pool:
        by_rank.setdefault(rank, []).append(span)
    picked: list[str] = []
    prefixes: set[str] = set()
    for rank in sorted(by_rank):
        group = by_rank[rank]
        rng.shuffle(group)
        for span in group:
            prefix = span[:48].casefold()
            if prefix in prefixes:
                continue
            prefixes.add(prefix)
            picked.append(span)
            if len(picked) == PER_LANGUAGE:
                return picked
    return picked


def _fill_from_pairs(
    pairs,
    language: str,
    min_words: int,
    max_words: int,
    letter_ratio: float,
    limit: int,
) -> list[tuple[int, str]]:
    pool: list[tuple[int, str]] = []
    seen: set[str] = set()
    for raw, source in pairs:
        rank = _source_rank(source)
        for span in _spans(raw):
            if not _quality(
                span,
                language=language,
                min_words=min_words,
                max_words=max_words,
                min_letter_ratio=letter_ratio,
            ):
                continue
            key = span.casefold()
            if key in seen:
                continue
            seen.add(key)
            pool.append((rank, span))
            if len(pool) >= limit:
                return pool
    return pool


def _collect_from_parquet(path: Path, language: str, config: str, rng: random.Random) -> list[str]:
    passes = (
        (15, 35, 0.62, 120),
        (12, 40, 0.55, 160),
        (8, 50, 0.50, 200),
    )
    pool: list[tuple[int, str]] = []
    for min_words, max_words, letter_ratio, limit in passes:
        pool = _fill_from_pairs(
            _iter_parquet_rows(path, config),
            language,
            min_words,
            max_words,
            letter_ratio,
            limit,
        )
        if len(pool) >= PER_LANGUAGE:
            return _pick(pool, rng)
    return _pick(pool, rng) if pool else []


def _collect_from_api(
    session: requests.Session,
    token: str,
    language: str,
    config: str,
    rng: random.Random,
) -> list[str]:
    num_rows = _num_rows(session, token, config)
    if num_rows <= 0:
        return []
    passes = (
        (15, 35, 0.62, 12),
        (12, 40, 0.55, 16),
        (8, 50, 0.50, 20),
    )
    pool: list[tuple[int, str]] = []
    for min_words, max_words, letter_ratio, pages in passes:
        pairs = []
        for offset in _offsets(num_rows, rng, pages):
            try:
                rows = _fetch_page(
                    session,
                    token,
                    config,
                    offset,
                    min(PAGE_SIZE, max(1, num_rows - offset)),
                )
            except RuntimeError as exc:
                print(f"  warning: {language} offset {offset}: {exc}")
                time.sleep(RATE_LIMIT_SLEEP)
                continue
            time.sleep(API_PAUSE_SECONDS)
            for row in rows:
                pairs.append(_text_from_row(row, config))
            pool = _fill_from_pairs(pairs, language, min_words, max_words, letter_ratio, 80)
            if len(pool) >= 80:
                break
        if len(pool) >= PER_LANGUAGE:
            return _pick(pool, rng)
    return _pick(pool, rng) if pool else []


def _collect(
    session: requests.Session,
    token: str,
    language: str,
    config: str,
    rng: random.Random,
    small_shards: dict[str, str],
) -> list[str]:
    filename = small_shards.get(config)
    if filename:
        print(f"  using local parquet {filename}", flush=True)
        try:
            path = _download_parquet(token, filename)
            return _collect_from_parquet(path, language, config, rng)
        except Exception as exc:
            print(f"  parquet download failed ({type(exc).__name__}); falling back to dataset-viewer")
    print("  using dataset-viewer pages", flush=True)
    return _collect_from_api(session, token, language, config, rng)


def _write_csv(languages: list[dict[str, str]], texts: dict[str, list[str]]) -> None:
    rows = []
    for language in languages:
        key = language["key"]
        code = language["code"]
        lines = texts[key]
        for index, text in enumerate(lines, start=1):
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
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of app language keys")
    args = parser.parse_args()

    token = _load_token()
    languages = _load_languages()
    existing = _load_existing()
    if args.only:
        wanted = {key.lower() for key in args.only}
        languages = [item for item in languages if item["key"] in wanted]
        if not languages:
            print("No matching --only languages")
            return 1

    session = _session()
    cache = _load_cache(args.seed)
    small_shards = _small_shards(token)
    sampled: dict[str, list[str]] = dict(cache)
    missing: list[str] = []
    short: list[str] = []
    mapping_rows: list[tuple[str, str, str]] = []

    print(f"dataset: {REPO}")
    print(f"seed: {args.seed}")
    print(f"token set: {True}")
    print(f"cached languages: {len(cache)}")
    print(f"small parquet configs: {len(small_shards)}")

    for language in languages:
        key = language["key"]
        config = _hf_config(key)
        rng = random.Random(f"{args.seed}:{key}")
        if not config:
            lines = existing.get(key, [])[:PER_LANGUAGE]
            sampled[key] = lines
            missing.append(key)
            mapping_rows.append((key, "(none)", f"kept {len(lines)} existing"))
            print(f"{key:16}  no HF config; keeping {len(lines)} existing sentence(s)")
            continue
        if key in cache:
            lines = cache[key]
            sampled[key] = lines
            mapping_rows.append((key, config, f"cached {len(lines)}"))
            print(f"{key:16}  cached {len(lines)}")
            continue
        print(f"{key:16}  config={config!r} ...", flush=True)
        lines = _collect(session, token, key, config, rng, small_shards)
        if len(lines) < PER_LANGUAGE:
            short.append(f"{key}:{len(lines)}")
        sampled[key] = lines
        if len(lines) == PER_LANGUAGE:
            cache[key] = lines
            _save_cache(args.seed, cache)
        mapping_rows.append((key, config, f"sampled {len(lines)}"))
        print(f"{key:16}  sampled {len(lines)}")

    if args.only:
        all_languages = _load_languages()
        texts = {}
        for language in all_languages:
            key = language["key"]
            texts[key] = sampled.get(key) or existing.get(key, [])[:PER_LANGUAGE]
        sampled = texts
        languages = all_languages
    else:
        for language in languages:
            key = language["key"]
            sampled.setdefault(key, existing.get(key, [])[:PER_LANGUAGE])

    _write_csv(languages, sampled)
    print(f"\nwrote {OUT.relative_to(PROJECT_ROOT)}")

    print("\nMapping (app key -> HF config)")
    for key, config, note in mapping_rows:
        print(f"  {key:16}  {config:20}  {note}")
    if missing:
        print("\nMissing from dataset (kept existing rows, nothing invented):")
        for key in missing:
            print(f"  - {key}")
    if short:
        print("\nFewer than 10 authentic sentences after filtering:")
        for item in short:
            print(f"  - {item}")

    catalog = TestCatalog.load(PROJECT_ROOT / "config")
    counts = {language.key: len(catalog.sentences_for(language.key)) for language in catalog.languages}
    print(f"\ncatalog languages: {len(catalog.languages)}")
    print(f"catalog sentences: {len(catalog.sentences)}")
    under = [f"{key}:{count}" for key, count in counts.items() if count != PER_LANGUAGE]
    if under:
        print("languages without 10 sentences:", ", ".join(under))
        return 1
    print("catalog: 38 languages, 10 sentences each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
