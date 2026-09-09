"""Evaluation logic: test conditions, A/B randomisation, trials, validation.

Vocabulary
----------
system      internal identifier of a TTS system ("individual" / "combined").
            NEVER shown to a tester.
sample      what the tester sees: "A" or "B".
condition   language + accent + gender + speaker + sentence. Both systems
            always receive the *same* condition.
trial       one condition plus a fixed sample<->system assignment plus the two
            generated audio clips.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
import re
from pathlib import Path
import secrets
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

from .audio_cache import AudioCache, CacheKey
from .config_loader import Accent, ConfigError, Language, Sentence, Speaker, TestCatalog
from .hf_endpoint import (
    CLONE_MODE,
    GENERATION_MODES,
    PRESET_MODE,
    AudioClip,
    EndpointAdapter,
    EndpointError,
    EndpointLoadingError,
    ReferenceAudio,
    SynthesisRequest,
    TTSClient,
    build_clients,
)
from .settings import SYSTEMS, Settings

logger = logging.getLogger("stream_tts.evaluation")

SAMPLE_LABELS: tuple[str, ...] = ("A", "B")
PREFERENCE_CHOICES: tuple[str, ...] = ("A", "B", "same")
RATING_SCALE: tuple[int, ...] = (1, 2, 3, 4, 5)

RATING_SCALE_HELP = {
    1: "1 - Bad",
    2: "2 - Poor",
    3: "3 - Fair",
    4: "4 - Good",
    5: "5 - Excellent",
}

INSTRUCTIONS = (
    "Listen to both samples before rating them. You can rate one pair or as many as you "
    "like. Judge only speech quality, naturalness, pronunciation, and voice."
)

INSTRUCTIONS_SINGLE = (
    "Listen to the sample before rating it. You can rate one sentence or as many as you "
    "like. Judge only speech quality, naturalness, pronunciation, and voice."
)


def tester_instructions(*, hide_sample_b: bool) -> str:
    return INSTRUCTIONS_SINGLE if hide_sample_b else INSTRUCTIONS


class EvaluationError(RuntimeError):
    """Base class for evaluation problems."""


class SampleGenerationError(EvaluationError):
    """One or both samples could not be produced.

    `message` is safe to show to a tester; `detail` is for the log file only.
    """

    def __init__(self, detail: str, message: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.message = message or "Unable to generate one of the samples. Please try again."


# ---------------------------------------------------------------------------
# Test condition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TestCondition:
    language: Language
    speaker: Speaker
    sentence: Sentence
    accent: Accent | None = None
    #: "preset" uses the speaker prompts baked into the model (the default and
    #: only path that needs no reference audio); "clone" sends a user clip.
    generation_mode: str = PRESET_MODE
    reference_audio: ReferenceAudio | None = None
    reference_text: str | None = None

    @property
    def gender(self) -> str:
        return self.speaker.gender

    @property
    def is_clone(self) -> bool:
        return self.generation_mode == CLONE_MODE

    @property
    def reference_id(self) -> str:
        return self.reference_audio.identifier if (self.is_clone and self.reference_audio) else ""

    @property
    def mode_label(self) -> str:
        """Tester-facing description of the generation path."""
        return "Cloned voice" if self.is_clone else "Preset voice"

    @property
    def accent_key(self) -> str | None:
        return self.accent.key if self.accent else None

    @property
    def accent_label(self) -> str | None:
        return self.accent.label if self.accent else None

    @property
    def key(self) -> str:
        """Coverage key: the five dataset dimensions, independent of voice mode."""
        return "/".join(
            [
                self.language.key,
                self.accent_key or "-",
                self.gender,
                self.speaker.speaker_id,
                self.sentence.sentence_id,
            ]
        )

    @property
    def selection_key(self) -> str:
        """Full identity of what the user asked for, including the voice mode."""
        return "/".join([self.key, self.generation_mode, self.reference_id])

    def synthesis_request(self) -> SynthesisRequest:
        """Identical request for both systems - only the endpoint differs."""
        return SynthesisRequest(
            text=self.sentence.text,
            language=self.language.key,
            language_label=self.language.label,
            language_code=self.language.code,
            speaker_id=self.speaker.speaker_id,
            gender=self.gender,
            sentence_id=self.sentence.sentence_id,
            accent=self.accent_key,
            accent_label=self.accent_label,
            generation_mode=self.generation_mode,
            reference_audio=self.reference_audio,
            reference_text=self.reference_text,
        )


def _clean_reference_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


@lru_cache(maxsize=32)
def _read_house_prompt_bytes(path: str) -> bytes:
    data = Path(path).read_bytes()
    if len(data) < 64:
        raise EvaluationError(f"House prompt {path} is too small to be audio")
    return data


def load_house_prompt(catalog: TestCatalog, speaker: Speaker) -> ReferenceAudio | None:
    """Load the speaker's bundled clip, if speakers.yaml points at one."""
    rel = speaker.primary_reference_audio
    if not rel:
        return None
    path = (catalog.source_dir.parent / rel).resolve()
    return ReferenceAudio.from_upload(_read_house_prompt_bytes(str(path)), path.name)


def build_condition(
    catalog: TestCatalog,
    *,
    language_key: str,
    speaker_id: str,
    sentence_id: str | None = None,
    sentence: Sentence | None = None,
    accent_key: str | None = None,
    generation_mode: str = PRESET_MODE,
    reference_audio: ReferenceAudio | None = None,
    reference_text: str | None = None,
) -> TestCondition:
    """Assemble one test condition.

    Pass either `sentence_id` (from the CSV) or a ready-made `sentence`
    (custom text typed in the app).
    """
    language = catalog.language(language_key)
    speaker = catalog.speaker(speaker_id)
    if sentence is None:
        if not sentence_id:
            raise EvaluationError("A test sentence is required")
        sentence = catalog.sentence(sentence_id)

    if generation_mode not in GENERATION_MODES:
        raise EvaluationError(f"Unknown generation mode {generation_mode!r}")
    if generation_mode == CLONE_MODE and reference_audio is None:
        raise EvaluationError(
            "Voice cloning needs reference audio - upload a clip or give a URL."
        )
    if generation_mode == PRESET_MODE:
        # English house clips clone with the matching _refs transcript (zero-shot).
        reference_audio = load_house_prompt(catalog, speaker)
        if not reference_text:
            reference_text = speaker.reference_text

    if not language.available:
        raise EvaluationError(
            f"{language.label} is not available for evaluation yet "
            f"({language.unavailable_reason or 'withheld in languages.yaml'})."
        )
    if speaker.language != language.key:
        raise EvaluationError(f"Speaker {speaker_id!r} does not belong to {language.key!r}")
    if sentence.language != language.key:
        raise EvaluationError(f"Sentence {sentence_id!r} does not belong to {language.key!r}")

    accent = None
    if language.has_accents:
        resolved = accent_key or speaker.accent
        if not resolved:
            raise EvaluationError(f"{language.label} requires an accent selection")
        accent = language.accent(resolved)
        if speaker.accent != accent.key:
            raise EvaluationError(
                f"Speaker {speaker_id!r} is not a {accent.label} reference speaker"
            )
        if sentence.accent not in (None, accent.key):
            raise EvaluationError(f"Sentence {sentence_id!r} is not usable with {accent.label}")
    elif accent_key:
        raise EvaluationError(f"{language.label} does not define accents")

    return TestCondition(
        language=language,
        speaker=speaker,
        sentence=sentence,
        accent=accent,
        generation_mode=generation_mode,
        reference_audio=reference_audio,
        reference_text=_clean_reference_text(reference_text),
    )


def pick_random_condition(
    catalog: TestCatalog,
    *,
    exclude_keys: Iterable[str] = (),
    generation_mode: str = PRESET_MODE,
    reference_audio: ReferenceAudio | None = None,
    reference_text: str | None = None,
) -> TestCondition:
    """MODE 2 - randomised evaluation pool.

    Picks uniformly at random among conditions the tester has not seen yet in
    this session.
    """
    conditions = [
        build_condition(
            catalog,
            language_key=language.key,
            speaker_id=speaker.speaker_id,
            sentence=sentence,
            accent_key=accent.key if accent else None,
            generation_mode=generation_mode,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )
        for language, accent, speaker, sentence in catalog.iter_conditions()
    ]
    if not conditions:
        raise EvaluationError("The evaluation pool is empty - check your configuration")

    excluded = set(exclude_keys)
    remaining = [c for c in conditions if c.key not in excluded] or conditions
    return remaining[secrets.randbelow(len(remaining))]


# ---------------------------------------------------------------------------
# A/B assignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleAssignment:
    """Which internal system is behind Sample A and Sample B.

    Created once per trial with `secrets` and never changed afterwards, so
    Streamlit reruns cannot reshuffle a running evaluation.
    """

    model_for_A: str
    model_for_B: str
    #: False when the mapping was pinned by configuration instead of drawn.
    randomized: bool = True

    @classmethod
    def random(cls, systems: Sequence[str]) -> "SampleAssignment":
        if len(systems) != 2:
            raise EvaluationError("Exactly two systems are required for an A/B comparison")
        first, second = systems
        if secrets.randbits(1):
            first, second = second, first
        return cls(model_for_A=first, model_for_B=second, randomized=True)

    @classmethod
    def pinned(cls, systems: Sequence[str], sample_a_system: str) -> "SampleAssignment":
        """Fixed mapping - a temporary research setting, NOT bias-controlled.

        Used while the second system is still a placeholder, so the researcher
        knows which side is which.
        """
        if len(systems) != 2:
            raise EvaluationError("Exactly two systems are required for an A/B comparison")
        if sample_a_system not in systems:
            raise EvaluationError(f"Unknown system {sample_a_system!r} for Sample A")
        other = next(system for system in systems if system != sample_a_system)
        return cls(model_for_A=sample_a_system, model_for_B=other, randomized=False)

    @classmethod
    def resolve(
        cls,
        systems: Sequence[str],
        *,
        randomize: bool = True,
        sample_a_system: str | None = None,
    ) -> "SampleAssignment":
        if randomize or not sample_a_system:
            return cls.random(systems)
        return cls.pinned(systems, sample_a_system)

    def system_for(self, sample: str) -> str:
        return self.model_for_A if sample == "A" else self.model_for_B

    def sample_for(self, system: str) -> str:
        return "A" if self.model_for_A == system else "B"


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Chunked playback
# ---------------------------------------------------------------------------
#
# st.audio() takes a finished byte string, so Streamlit cannot play a growing
# stream the way a browser can. The endpoint itself streams well -- first byte in
# under a second -- but none of that reaches a listener here, who waits for the
# whole passage before anything sounds.
#
# What Streamlit CAN do is render as the script runs. So the text is split into
# sentences and synthesised one at a time: each finished sentence is played as
# soon as it exists, while the rest are still generating. First audio then arrives
# after one sentence rather than after the whole passage.
#
# Each chunk is cached under its own key (the sentence id gains a "#cN" suffix),
# so a repeat listen is instant and a half-finished trial is not wasted.
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_for_playback(text, min_chars=45):
    """One chunk per sentence, so the first one lands quickly.

    Sentences are the natural unit: they end on a pause, so consecutive clips join
    without an audible seam. An over-long sentence is left whole rather than cut
    mid-clause, which would sound worse than a slightly later start.

    min_chars is a FLOOR, not a target: a fragment shorter than it (a heading, "Yes.",
    a poem's title line) is joined to what precedes it, because a one-second clip of
    its own is both wasteful to request and jarring to hear. Anything at or above the
    floor stands alone -- building chunks UP to a character target would merge the
    whole passage back into one request and defeat the point.
    """
    parts = [p.strip() for p in _SENTENCE_END.split(text or "") if p and p.strip()]
    if not parts:
        return [text.strip()] if (text or "").strip() else []
    out = []
    for part in parts:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


@dataclass(frozen=True)
class PreparedSample:
    sample: str  # "A" / "B"
    system: str  # internal - never rendered
    clip: AudioClip
    cache_key: str
    from_cache: bool
    #: True when this side came from the offline placeholder client, so those
    #: rows can be filtered out of the analysis later.
    mocked: bool = False


@dataclass
class Trial:
    condition: TestCondition
    assignment: SampleAssignment
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    samples: dict[str, PreparedSample] = field(default_factory=dict)
    #: When True, Sample B is a placeholder system and is not generated or shown.
    hide_sample_b: bool = False
    #: Why a visible sample is missing, keyed by label. Shown to the listener.
    failures: dict[str, str] = field(default_factory=dict)
    #: The text split into playback chunks; chunk 0 is prepared before rendering.
    chunks: list = field(default_factory=list)
    #: Finished chunk audio, keyed by "<label>:<index>".
    chunk_audio: dict = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """True when there is something to listen to.

        Deliberately not "every sample succeeded". One endpoint waking from scale-to-zero
        used to withhold the sample that had already been generated, which wastes the work
        and leaves the listener with nothing while the other side boots.
        """
        return any(label in self.samples for label in self.visible_labels)

    @property
    def visible_labels(self) -> tuple[str, ...]:
        return ("A",) if self.hide_sample_b else SAMPLE_LABELS

    @property
    def available_labels(self) -> tuple[str, ...]:
        """Visible samples that actually have audio."""
        return tuple(l for l in self.visible_labels if l in self.samples)

    @property
    def is_complete(self) -> bool:
        """Every visible sample was produced -- the precondition for an A/B comparison."""
        return all(label in self.samples for label in self.visible_labels)

    def sample(self, label: str) -> PreparedSample:
        return self.samples[label]


def new_trial(
    condition: TestCondition,
    systems: Sequence[str] = SYSTEMS,
    *,
    randomize: bool = True,
    sample_a_system: str | None = None,
    hide_sample_b: bool = False,
) -> Trial:
    if hide_sample_b:
        randomize = False
        sample_a_system = sample_a_system or SYSTEMS[0]
    return Trial(
        condition=condition,
        assignment=SampleAssignment.resolve(
            systems, randomize=randomize, sample_a_system=sample_a_system
        ),
        hide_sample_b=hide_sample_b,
    )


def new_trial_for(settings: Settings, condition: TestCondition) -> Trial:
    """Trial whose A/B mapping follows the configured randomisation policy."""
    hide = settings.hide_sample_b()
    return new_trial(
        condition,
        SYSTEMS,
        randomize=settings.randomize_ab and not hide,
        sample_a_system=settings.sample_a_system,
        hide_sample_b=hide,
    )


def build_clients_for(
    settings: Settings, adapter: EndpointAdapter | None = None
) -> dict[str, TTSClient]:
    """Create one client per system.

    The UI calls this and therefore never handles system identifiers or the
    endpoint wire format.
    """
    if adapter is None:
        adapter = EndpointAdapter.load(settings.config_dir / "endpoint.yaml")
    return build_clients(settings, adapter, SYSTEMS)



def generate_chunk(trial, label, index, clients, cache):
    """Synthesise one playback chunk, reusing the cache.

    Each chunk is a separate request carrying only that sentence, and is cached
    under its own key so nothing is regenerated on a rerun. The system behind a
    label stays hidden: the caller only ever sees audio.
    """
    key_name = f"{label}:{index}"
    if key_name in trial.chunk_audio:
        return trial.chunk_audio[key_name]

    system = trial.assignment.system_for(label)
    client = clients.get(system)
    if client is None:
        raise EndpointError(f"no client available for system {system!r}")

    base = trial.condition.synthesis_request()
    chunk_text = trial.chunks[index]
    request = replace(base, text=chunk_text,
                      sentence_id=f"{base.sentence_id}#c{index}")
    cache_key = CacheKey(
        system=system,
        language=trial.condition.language.key,
        accent=trial.condition.accent_key,
        gender=trial.condition.gender,
        speaker_id=trial.condition.speaker.speaker_id,
        sentence_id=request.sentence_id,
        generation_mode=trial.condition.generation_mode,
        reference_id=trial.condition.reference_id,
    )
    mocked = bool(getattr(client, "is_mock", False))
    fingerprint = f"{request.fingerprint()}:{'mock' if mocked else 'live'}"
    cached = cache.get_or_generate(
        cache_key, fingerprint,
        lambda client=client, request=request: client.synthesize(request))
    trial.chunk_audio[key_name] = cached.clip
    return cached.clip

def prepare_samples(
    trial: Trial,
    clients: Mapping[str, TTSClient],
    cache: AudioCache,
) -> Trial:
    """Generate (or reuse cached) audio for the visible samples of a trial.

    Raises :class:`SampleGenerationError` if a required side fails. Sample B is
    skipped entirely while it is hidden.
    """
    request = trial.condition.synthesis_request()
    if not trial.chunks:
        trial.chunks = split_for_playback(request.text)
    fingerprint = request.fingerprint()
    failures: list[str] = []
    labels = list(trial.visible_labels)

    def _one(label: str):
        system = trial.assignment.system_for(label)
        client = clients.get(system)
        if client is None:
            raise EndpointError(f"no client available for system {system!r}")
        mocked = bool(getattr(client, "is_mock", False))
        cache_key = CacheKey(
            system=system,
            language=trial.condition.language.key,
            accent=trial.condition.accent_key,
            gender=trial.condition.gender,
            speaker_id=trial.condition.speaker.speaker_id,
            sentence_id=trial.condition.sentence.sentence_id,
            generation_mode=trial.condition.generation_mode,
            reference_id=trial.condition.reference_id,
        )
        sample_fingerprint = f"{fingerprint}:{'mock' if mocked else 'live'}"
        cached = cache.get_or_generate(
            cache_key,
            sample_fingerprint,
            lambda client=client, request=request: client.synthesize(request),
        )
        return label, system, cached, cache_key, mocked

    workers = min(2, max(1, len(labels)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, label): label for label in labels}
        for future in as_completed(futures):
            label = futures[future]
            try:
                sample_label, system, cached, cache_key, mocked = future.result()
            except EndpointLoadingError as exc:
                failures.append(f"{label}: {exc}")
                trial.failures[label] = (
                    f"The Sample {label} model is loading. Wait a minute, then press "
                    f"Play again."
                )
                continue
            except EndpointError as exc:
                failures.append(f"{label}: {exc}")
                trial.failures[label] = f"Sample {label} could not be generated. Try again."
                continue
            except OSError as exc:
                failures.append(f"{label}: cache I/O error: {exc}")
                trial.failures[label] = f"Sample {label} could not be saved. Try again."
                continue
            except Exception as exc:
                failures.append(f"{label}: {exc}")
                trial.failures[label] = f"Sample {label} could not be generated. Try again."
                continue
            trial.samples[sample_label] = PreparedSample(
                sample=sample_label,
                system=system,
                clip=cached.clip,
                cache_key=cache_key.as_string(cached.clip.extension),
                from_cache=cached.from_cache,
                mocked=mocked,
            )

    if failures:
        detail = f"trial {trial.trial_id} condition {trial.condition.key}: " + "; ".join(failures)
        logger.error("Sample generation failed - %s", detail)
        # Only give up when NOTHING was produced. If one side is ready it is kept and played;
        # the other reports why it is missing.
        if not trial.samples:
            trial.samples.clear()
            raise SampleGenerationError(detail)

    return trial


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------
@dataclass
class Ratings:
    preference: str | None = None  # "A" | "B" | "same"
    naturalness_a: int | None = None
    naturalness_b: int | None = None
    pronunciation_a: int | None = None
    pronunciation_b: int | None = None
    speaker_similarity_a: int | None = None
    speaker_similarity_b: int | None = None
    comments: str = ""
    listened_to_both: bool = False


def validate_ratings(
    ratings: Ratings,
    *,
    require_listen_confirmation: bool = True,
    voice_criterion_label: str = "Voice quality",
    hide_sample_b: bool = False,
    available_labels: tuple[str, ...] | None = None,
) -> list[str]:
    """Return a list of human-readable problems; empty means valid.

    `available_labels` names the samples that were actually produced. A side that failed to
    generate cannot be rated, so it is not required -- and a preference between two samples
    is only meaningful when both exist.
    """
    problems: list[str] = []
    avail = available_labels if available_labels is not None else (
        ("A",) if hide_sample_b else ("A", "B"))

    if not hide_sample_b and len(avail) > 1 and ratings.preference not in PREFERENCE_CHOICES:
        problems.append("Please choose which sample sounds better overall.")

    required = (
        ("Naturalness", ratings.naturalness_a, ratings.naturalness_b),
        ("Pronunciation / intelligibility", ratings.pronunciation_a, ratings.pronunciation_b),
        (voice_criterion_label, ratings.speaker_similarity_a, ratings.speaker_similarity_b),
    )
    for name, value_a, value_b in required:
        sides = (("A", value_a),) if hide_sample_b else (("A", value_a), ("B", value_b))
        sides = tuple((l, v) for l, v in sides if l in avail)
        missing = [label for label, value in sides if value not in RATING_SCALE]
        if missing:
            samples = " and ".join(f"Sample {label}" for label in missing)
            problems.append(f"Please rate {name} for {samples}.")

    if require_listen_confirmation and not ratings.listened_to_both:
        if hide_sample_b or len(avail) < 2:
            problems.append("Please confirm that you listened to the sample.")
        else:
            problems.append("Please confirm that you listened to both samples.")

    return problems


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------
def build_result_row(
    trial: Trial,
    ratings: Ratings,
    *,
    tester_id: str,
    session_id: str,
    selection_mode: str,
    app_version: str,
    mock_mode: bool | None = None,
    listening_seconds: float | None = None,
) -> dict[str, object]:
    """Flatten a completed trial into a storage row.

    `sample_A_model` / `sample_B_model` are INTERNAL decoding fields: they are
    written here for the researcher and never rendered in the tester UI.
    """
    condition = trial.condition
    preferred_system = {
        "A": trial.assignment.model_for_A,
        "B": trial.assignment.model_for_B,
        "same": "tie",
    }.get(ratings.preference or "", "")

    sample_a = trial.samples.get("A")
    sample_b = trial.samples.get("B")
    mocked_a = bool(sample_a.mocked) if sample_a else False
    mocked_b = bool(sample_b.mocked) if sample_b else bool(trial.hide_sample_b)
    # Per-trial accuracy: a row is "mock" when either side came from a
    # placeholder system, whatever the global setting says.
    trial_used_mock = mocked_a or mocked_b if sample_a else bool(mock_mode)

    return {
        "evaluation_id": uuid.uuid4().hex,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "tester_id": tester_id,
        "session_id": session_id,
        "trial_id": trial.trial_id,
        "language": condition.language.key,
        "accent": condition.accent_key or "",
        "gender": condition.gender,
        "speaker_id": condition.speaker.speaker_id,
        "sentence_id": condition.sentence.sentence_id,
        "sentence_text": condition.sentence.text,
        "sample_A_model": trial.assignment.model_for_A,
        "sample_B_model": trial.assignment.model_for_B,
        "preferred_sample": ratings.preference or "",
        "preferred_model": preferred_system,
        "naturalness_A": ratings.naturalness_a,
        "naturalness_B": ratings.naturalness_b,
        "pronunciation_A": ratings.pronunciation_a,
        "pronunciation_B": ratings.pronunciation_b,
        "speaker_similarity_A": ratings.speaker_similarity_a,
        "speaker_similarity_B": ratings.speaker_similarity_b,
        "comments": (ratings.comments or "").strip(),
        "sample_A_cache_key": sample_a.cache_key if sample_a else "",
        "sample_B_cache_key": sample_b.cache_key if sample_b else "",
        "generation_mode": condition.generation_mode,
        "reference_audio": condition.reference_id,
        "reference_text": condition.reference_text or "",
        "selection_mode": selection_mode,
        "mock_mode": int(bool(trial_used_mock)),
        "sample_A_mocked": int(mocked_a),
        "sample_B_mocked": int(mocked_b),
        "ab_randomized": int(bool(trial.assignment.randomized)),
        "app_version": app_version,
        "listening_seconds": round(listening_seconds, 1) if listening_seconds else None,
    }
