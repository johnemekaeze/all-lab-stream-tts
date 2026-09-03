"""Loads and validates the test configuration (languages, speakers, sentences).

The rest of the application only ever talks to :class:`TestCatalog`; no
language, speaker or sentence is hard-coded anywhere else.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

GENDERS = ("male", "female")
UNSPECIFIED_GENDER = "unspecified"


class ConfigError(RuntimeError):
    """Raised when the YAML/CSV configuration is missing or inconsistent."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Accent:
    key: str
    label: str
    code: str


@dataclass(frozen=True)
class Language:
    key: str
    label: str
    code: str
    accents: tuple[Accent, ...] = ()
    #: False for a language the deployed systems cannot speak yet. It stays in
    #: the config (speakers, sentences and all) but is never offered to a
    #: tester, so nobody can start a trial the model will reject.
    available: bool = True
    #: Researcher-facing reason shown in the dashboard when unavailable.
    unavailable_reason: str = ""

    @property
    def has_accents(self) -> bool:
        return bool(self.accents)

    def accent(self, key: str) -> Accent:
        for accent in self.accents:
            if accent.key == key:
                return accent
        raise ConfigError(f"Language {self.key!r} has no accent {key!r}")


@dataclass(frozen=True)
class Speaker:
    """A reference speaker preset that already exists inside the model.

    `reference_audio` is OPTIONAL and unused by the default (preset) generation
    path: the speaker prompt is baked into the model, so `speaker_id` is all
    that gets sent. It only serves as a convenient default cloning prompt.
    Placeholder values are discarded at load time so they can never be sent.
    """

    speaker_id: str
    language: str
    gender: str
    accent: str | None = None
    label: str = ""
    reference_audio: tuple[str, ...] = ()
    reference_text: str | None = None

    @property
    def primary_reference_audio(self) -> str | None:
        return self.reference_audio[0] if self.reference_audio else None

    @property
    def has_reference_audio(self) -> bool:
        return bool(self.primary_reference_audio)

    @property
    def researcher_label(self) -> str:
        return self.label or self.speaker_id

    @property
    def neutral_label(self) -> str:
        """Speaker description shown to testers - reveals no system identity."""
        if self.gender in GENDERS:
            return f"{self.speaker_id} ({self.gender})"
        return self.speaker_id


CUSTOM_SENTENCE_PREFIX = "custom-"
MAX_SENTENCE_LENGTH = 600


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    language: str
    text: str
    accent: str | None = None
    #: True for text typed into the app rather than read from the CSV.
    is_custom: bool = False

    @property
    def is_placeholder(self) -> bool:
        return self.text.strip().upper().startswith("[PLACEHOLDER]")

    @property
    def preview(self) -> str:
        text = self.text.strip()
        return text if len(text) <= 70 else text[:67] + "..."


def make_custom_sentence(text: str, language: str, accent: str | None = None) -> Sentence:
    """Build a sentence from text typed by the user.

    The ID is derived from the text so the same custom sentence always reuses
    the same cached audio and groups together in the results.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        raise ConfigError("Enter the sentence you want the systems to speak.")
    if len(cleaned) > MAX_SENTENCE_LENGTH:
        raise ConfigError(
            f"That sentence is {len(cleaned)} characters long; please keep it under "
            f"{MAX_SENTENCE_LENGTH}."
        )
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:10]
    return Sentence(
        sentence_id=f"{CUSTOM_SENTENCE_PREFIX}{digest}",
        language=language,
        text=cleaned,
        accent=accent,
        is_custom=True,
    )


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TestCatalog:
    languages: tuple[Language, ...]
    speakers: tuple[Speaker, ...]
    sentences: tuple[Sentence, ...]
    source_dir: Path

    # -- lookups -----------------------------------------------------------
    def language(self, key: str) -> Language:
        for language in self.languages:
            if language.key == key:
                return language
        raise ConfigError(f"Unknown language {key!r}")

    @property
    def available_languages(self) -> tuple[Language, ...]:
        """Languages the deployed systems can actually speak."""
        return tuple(language for language in self.languages if language.available)

    @property
    def unavailable_languages(self) -> tuple[Language, ...]:
        return tuple(language for language in self.languages if not language.available)

    def language_keys(self) -> tuple[str, ...]:
        """Keys offered in the UI - unavailable languages are excluded."""
        return tuple(language.key for language in self.available_languages)

    def accents(self, language_key: str) -> tuple[Accent, ...]:
        return self.language(language_key).accents

    def genders(self, language_key: str, accent: str | None = None) -> tuple[str, ...]:
        """Genders that actually have a configured speaker."""
        found = {s.gender for s in self.speakers_for(language_key, accent=accent)}
        ordered = [g for g in GENDERS if g in found]
        ordered += sorted(found - set(GENDERS))
        return tuple(ordered)

    def speakers_for(
        self,
        language_key: str,
        *,
        accent: str | None = None,
        gender: str | None = None,
    ) -> tuple[Speaker, ...]:
        result = [s for s in self.speakers if s.language == language_key]
        if accent is not None:
            result = [s for s in result if s.accent == accent]
        if gender is not None:
            result = [s for s in result if s.gender == gender]
        return tuple(result)

    def speaker(self, speaker_id: str) -> Speaker:
        for speaker in self.speakers:
            if speaker.speaker_id == speaker_id:
                return speaker
        raise ConfigError(f"Unknown speaker {speaker_id!r}")

    def sentences_for(self, language_key: str, accent: str | None = None) -> tuple[Sentence, ...]:
        result = [s for s in self.sentences if s.language == language_key]
        if accent is not None:
            # A sentence without an accent is usable with every accent.
            result = [s for s in result if s.accent in (None, accent)]
        return tuple(result)

    def sentence(self, sentence_id: str) -> Sentence:
        for sentence in self.sentences:
            if sentence.sentence_id == sentence_id:
                return sentence
        raise ConfigError(f"Unknown sentence {sentence_id!r}")

    # -- coverage ----------------------------------------------------------
    def iter_conditions(self) -> Iterator[tuple[Language, Accent | None, Speaker, Sentence]]:
        """Every (language, accent, speaker, sentence) combination on offer."""
        for language in self.available_languages:
            if language.has_accents:
                for accent in language.accents:
                    for speaker in self.speakers_for(language.key, accent=accent.key):
                        for sentence in self.sentences_for(language.key, accent=accent.key):
                            yield language, accent, speaker, sentence
            else:
                for speaker in self.speakers_for(language.key):
                    for sentence in self.sentences_for(language.key):
                        yield language, None, speaker, sentence

    def condition_count(self) -> int:
        return sum(1 for _ in self.iter_conditions())

    def summary(self) -> dict[str, int]:
        return {
            "languages": len(self.available_languages),
            "languages_configured": len(self.languages),
            "speakers": len(self.speakers),
            "sentences": len(self.sentences),
            "conditions": self.condition_count(),
        }

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, config_dir: Path) -> "TestCatalog":
        languages = _load_languages(config_dir / "languages.yaml")
        speakers = _load_speakers(config_dir / "speakers.yaml", languages)
        sentences = _load_sentences(config_dir / "test_sentences.csv", languages)
        catalog = cls(
            languages=languages,
            speakers=speakers,
            sentences=sentences,
            source_dir=config_dir,
        )
        _validate(catalog)
        return catalog


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc


def _load_languages(path: Path) -> tuple[Language, ...]:
    raw = _read_yaml(path)
    entries = raw.get("languages") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{path.name} must contain a non-empty 'languages' list")

    languages: list[Language] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("key"):
            raise ConfigError(f"{path.name}: language #{index + 1} needs a 'key'")
        key = str(entry["key"]).strip().lower()
        if key in seen:
            raise ConfigError(f"{path.name}: duplicate language key {key!r}")
        seen.add(key)

        accents: list[Accent] = []
        for accent_entry in entry.get("accents") or []:
            if not isinstance(accent_entry, dict) or not accent_entry.get("key"):
                raise ConfigError(f"{path.name}: accent of {key!r} needs a 'key'")
            a_key = str(accent_entry["key"]).strip().lower()
            accents.append(
                Accent(
                    key=a_key,
                    label=str(accent_entry.get("label") or a_key.replace("_", " ").title()),
                    code=str(accent_entry.get("code") or a_key[:2].upper()),
                )
            )

        languages.append(
            Language(
                key=key,
                label=str(entry.get("label") or key.title()),
                code=str(entry.get("code") or key[:2].upper()),
                accents=tuple(accents),
                available=bool(entry.get("available", True)),
                unavailable_reason=str(entry.get("unavailable_reason") or ""),
            )
        )
    return tuple(languages)


def _as_speaker_entries(value: Any) -> list[dict]:
    """Accept either a single mapping or a list of mappings."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        entries = []
        for item in value:
            if not isinstance(item, dict):
                raise ConfigError(f"speakers.yaml: expected a mapping, got {type(item).__name__}")
            entries.append(item)
        return entries
    raise ConfigError(f"speakers.yaml: unsupported speaker entry of type {type(value).__name__}")


def _reference_audio(entry: dict) -> tuple[str, ...]:
    """Optional cloning prompts. Placeholder values are dropped."""
    raw = entry.get("reference_audio")
    if raw is None:
        return ()
    if isinstance(raw, str):
        candidates: tuple[str, ...] = (raw,)
    elif isinstance(raw, (list, tuple)):
        candidates = tuple(str(item) for item in raw if item)
    else:
        raise ConfigError(
            f"speakers.yaml: reference_audio for {entry.get('speaker_id')!r} must be a string or a list"
        )
    return tuple(
        value for value in candidates if value.strip() and not value.strip().upper().startswith("PLACEHOLDER")
    )


def _build_speaker(entry: dict, *, language: str, gender: str, accent: str | None) -> Speaker:
    speaker_id = str(entry.get("speaker_id") or "").strip()
    if not speaker_id:
        raise ConfigError(f"speakers.yaml: a speaker under {language!r} is missing 'speaker_id'")
    reference_text = entry.get("reference_text")
    return Speaker(
        speaker_id=speaker_id,
        language=language,
        gender=gender,
        accent=accent,
        label=str(entry.get("label") or ""),
        reference_audio=_reference_audio(entry),
        reference_text=str(reference_text) if reference_text else None,
    )


def _load_speakers(path: Path, languages: Sequence[Language]) -> tuple[Speaker, ...]:
    raw = _read_yaml(path)
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{path.name} must be a mapping of language -> speakers")
    # Tolerate an optional top-level 'speakers:' wrapper.
    if set(raw.keys()) == {"speakers"} and isinstance(raw["speakers"], dict):
        raw = raw["speakers"]

    by_key = {language.key: language for language in languages}
    speakers: list[Speaker] = []

    for language_key, block in raw.items():
        language_key = str(language_key).strip().lower()
        language = by_key.get(language_key)
        if language is None:
            raise ConfigError(
                f"{path.name}: {language_key!r} is not declared in languages.yaml"
            )
        if not isinstance(block, dict):
            raise ConfigError(f"{path.name}: {language_key!r} must be a mapping")

        if language.has_accents:
            accent_keys = {accent.key for accent in language.accents}
            for accent_key, entries in block.items():
                accent_key = str(accent_key).strip().lower()
                if accent_key not in accent_keys:
                    raise ConfigError(
                        f"{path.name}: {language_key!r} has speakers for unknown accent {accent_key!r}"
                    )
                for entry in _as_speaker_entries(entries):
                    gender = str(entry.get("gender") or UNSPECIFIED_GENDER).strip().lower()
                    speakers.append(
                        _build_speaker(entry, language=language_key, gender=gender, accent=accent_key)
                    )
        else:
            for gender_key, entries in block.items():
                gender = str(gender_key).strip().lower()
                for entry in _as_speaker_entries(entries):
                    speakers.append(
                        _build_speaker(entry, language=language_key, gender=gender, accent=None)
                    )

    return tuple(speakers)


def _load_sentences(path: Path, languages: Sequence[Language]) -> tuple[Sentence, ...]:
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")

    by_key = {language.key: language for language in languages}
    sentences: list[Sentence] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sentence_id", "language", "text"}
        missing = required - {(name or "").strip() for name in (reader.fieldnames or [])}
        if missing:
            raise ConfigError(
                f"{path.name} is missing required column(s): {', '.join(sorted(missing))}"
            )
        for line_no, row in enumerate(reader, start=2):
            sentence_id = (row.get("sentence_id") or "").strip()
            language_key = (row.get("language") or "").strip().lower()
            text = (row.get("text") or "").strip()
            accent = (row.get("accent") or "").strip().lower() or None
            if not sentence_id:
                raise ConfigError(f"{path.name} line {line_no}: empty sentence_id")
            if not text:
                raise ConfigError(f"{path.name} line {line_no}: empty text for {sentence_id!r}")
            if text.upper().startswith("[PLACEHOLDER]"):
                continue
            language = by_key.get(language_key)
            if language is None:
                raise ConfigError(
                    f"{path.name} line {line_no}: unknown language {language_key!r} for {sentence_id!r}"
                )
            if accent:
                if not language.has_accents:
                    raise ConfigError(
                        f"{path.name} line {line_no}: {language_key!r} does not define accents, "
                        f"but {sentence_id!r} sets accent={accent!r}"
                    )
                language.accent(accent)  # raises when the accent is unknown
            sentences.append(
                Sentence(sentence_id=sentence_id, language=language_key, text=text, accent=accent)
            )
    if not sentences:
        raise ConfigError(f"{path.name} contains no sentences")
    return tuple(sentences)


def _validate(catalog: TestCatalog) -> None:
    problems: list[str] = []

    duplicates = _duplicates([s.speaker_id for s in catalog.speakers])
    if duplicates:
        problems.append("duplicate speaker_id(s): " + ", ".join(sorted(duplicates)))

    duplicates = _duplicates([s.sentence_id for s in catalog.sentences])
    if duplicates:
        problems.append("duplicate sentence_id(s): " + ", ".join(sorted(duplicates)))

    for language in catalog.languages:
        if language.has_accents:
            for accent in language.accents:
                if not catalog.speakers_for(language.key, accent=accent.key):
                    problems.append(f"no speakers configured for {language.key}/{accent.key}")
                if not catalog.sentences_for(language.key, accent=accent.key):
                    problems.append(f"no test sentences configured for {language.key}/{accent.key}")
        else:
            if not catalog.speakers_for(language.key):
                problems.append(f"no speakers configured for {language.key}")
            if not catalog.sentences_for(language.key):
                problems.append(f"no test sentences configured for {language.key}")

    if problems:
        raise ConfigError("Invalid test configuration:\n  - " + "\n  - ".join(problems))


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
