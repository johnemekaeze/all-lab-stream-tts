"""Multilingual Speech Evaluation - anonymous A/B listening test.

Two TTS systems render the SAME test condition (language + accent + gender +
speaker + sentence). The two results are presented as "Sample A" and "Sample B"
in a random order that is fixed for the duration of the trial. This file is
tester-facing and therefore never names, hints at, or displays the systems
under test; the mapping is written to the results database only.
"""

from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st

from services import evaluation, settings as settings_module
from services.audio_cache import AudioCache
from services.config_loader import ConfigError, Language, TestCatalog, make_custom_sentence
from services.evaluation import (
    CLONE_MODE,
    PREFERENCE_CHOICES,
    PRESET_MODE,
    RATING_SCALE,
    RATING_SCALE_HELP,
    Ratings,
    ReferenceAudio,
    SampleGenerationError,
    TestCondition,
    tester_instructions,
)
from services import researcher_view
from services.settings import Settings, load_settings
from services.storage import ResultsStore, StorageError

APP_TITLE = "Multilingual Speech Evaluation"
APP_SUBTITLE = "Side-by-side listening test"
STYLESHEET = Path(__file__).resolve().parent / "assets" / "styles.css"

SENTENCE_SOURCES = ("catalog", "custom")
SENTENCE_SOURCE_LABELS = {
    "catalog": "A ready-made sentence",
    "custom": "Type my own",
}
VOICE_MODES = (PRESET_MODE, CLONE_MODE)
VOICE_MODE_LABELS = {
    PRESET_MODE: "Preset voice",
    CLONE_MODE: "Clone a voice",
}
UPLOAD_TYPES = ["wav", "mp3", "flac", "ogg", "m4a", "webm"]
#: In-browser recordings are resampled to this rate before they are sent.
REFERENCE_SAMPLE_RATE = 16000

PREFERENCE_LABELS = {
    "A": "Sample A",
    "B": "Sample B",
    "same": "About the same",
}

CRITERIA_BASE = (
    (
        "Naturalness",
        "naturalness",
        "Does it sound like a real human speaking, with natural rhythm and intonation?",
    ),
    (
        "Pronunciation / intelligibility",
        "pronunciation",
        "Are the words pronounced correctly and easy to understand?",
    ),
)

CRITERIA_VOICE_PRESET = (
    "Voice quality",
    "similarity",
    "How clean and pleasant is the voice?",
)

CRITERIA_VOICE_CLONE = (
    "Match to your recording",
    "similarity",
    "How well does this sample match the voice you recorded, and how clean does it sound?",
)


def rating_criteria(is_clone: bool) -> tuple[tuple[str, str, str], ...]:
    voice = CRITERIA_VOICE_CLONE if is_clone else CRITERIA_VOICE_PRESET
    return CRITERIA_BASE + (voice,)

SCALE_LEGEND = " · ".join(RATING_SCALE_HELP[value] for value in RATING_SCALE)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppContext:
    settings: Settings
    catalog: TestCatalog
    cache: AudioCache
    store: ResultsStore
    clients: dict


def _runtime_stamp() -> str:
    summary = settings_module.runtime_config_summary()
    return (
        f"mock={int(summary['mock_mode'])}"
        f"|token={int(summary['token_set'])}"
        f"|ind={int(summary['individual_live'])}"
        f"|comb={int(summary['combined_live'])}"
    )


@st.cache_resource(show_spinner="Loading evaluation configuration...")
def get_context(config_stamp: str, runtime_stamp: str) -> AppContext:
    _ = config_stamp, runtime_stamp
    app_settings = load_settings()
    logger = settings_module.configure_logging(app_settings)
    summary = settings_module.runtime_config_summary()
    logger.info(
        "Runtime secrets: MOCK_MODE=%s HF_TOKEN set=%s individual live=%s combined live=%s",
        summary["mock_mode"],
        summary["token_set"],
        summary["individual_live"],
        summary["combined_live"],
    )
    for warning in app_settings.warnings():
        logger.warning(warning)

    catalog = TestCatalog.load(app_settings.config_dir)
    clients = evaluation.build_clients_for(app_settings)

    logger.info(
        "Configuration loaded: %s of %s languages available, %s speakers, %s sentences",
        len(catalog.available_languages),
        len(catalog.languages),
        len(catalog.speakers),
        len(catalog.sentences),
    )
    if catalog.unavailable_languages:
        logger.info(
            "Languages withheld from testers: %s",
            ", ".join(language.key for language in catalog.unavailable_languages),
        )
    for system in settings_module.SYSTEMS:
        logger.info("System %s: %s", system, app_settings.system_status(system))
    logger.info(
        "A/B mapping: %s",
        "randomised per trial" if app_settings.randomize_ab else f"pinned, A={app_settings.sample_a_system}",
    )
    return AppContext(
        settings=app_settings,
        catalog=catalog,
        cache=AudioCache(app_settings.audio_cache_dir),
        store=ResultsStore(app_settings.results_db),
        clients=clients,
    )


def init_session(context: AppContext) -> None:
    state = st.session_state
    # Deferred clean-up from the previous run. Widget-backed keys may only be
    # written before their widget is instantiated, which is why this happens
    # here and not at submission time.
    stale_prefix = state.pop("stale_rating_prefix", None)
    if stale_prefix:
        for key in [key for key in state if key.startswith(stale_prefix)]:
            del state[key]
    pending_sentence = state.pop("pending_sentence_id", None)
    if pending_sentence:
        state["sel_sentence"] = pending_sentence

    state.setdefault("session_id", uuid.uuid4().hex)
    state.setdefault("trial", None)
    state.setdefault("submitted_count", 0)
    state.setdefault("seen_conditions", [])
    state.setdefault("generation_error", None)
    state.setdefault("saved_message", None)
    state.setdefault("selection_mode", context.settings.default_selection_mode)
    state.setdefault("trial_started_at", None)

    if "tester_id" not in state:
        state["tester_id"] = (
            "" if context.settings.tester_id_mode == "required" else _anonymous_id()
        )


def _anonymous_id() -> str:
    return f"tester-{secrets.token_hex(3)}"


def _config_stamp(config_dir: Path) -> str:
    parts: list[str] = []
    for name in ("languages.yaml", "speakers.yaml", "test_sentences.csv", "endpoint.yaml"):
        path = config_dir / name
        if path.exists():
            parts.append(f"{name}:{path.stat().st_mtime_ns}")
    return "|".join(parts)


def _sentence_scope(language_key: str, accent_key: str | None) -> str:
    return f"{language_key}:{accent_key or ''}"


def _sync_sentence_for_language(
    catalog: TestCatalog, language_key: str, accent_key: str | None
) -> tuple[list[str], str]:
    """Keep one preset sentence selected per language. No dropdown widget."""
    scope = _sentence_scope(language_key, accent_key)
    sentences = catalog.sentences_for(language_key, accent=accent_key)
    ids = [sentence.sentence_id for sentence in sentences]
    if st.session_state.get("sentence_scope") != scope:
        st.session_state["sentence_scope"] = scope
        st.session_state["sel_sentence"] = ids[0] if ids else None
    elif st.session_state.get("sel_sentence") not in ids:
        st.session_state["sel_sentence"] = ids[0] if ids else None
    return ids, scope


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _stable_selectbox(container, label: str, options: list[str], key: str, **kwargs):
    """Selectbox whose remembered value is validated against current options."""
    if not options:
        return None
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return container.selectbox(label, options, key=key, **kwargs)


@st.cache_data(show_spinner=False)
def _stylesheet(mtime: float) -> str:
    try:
        return STYLESHEET.read_text(encoding="utf-8")
    except OSError:
        return ""


def apply_styles() -> None:
    mtime = STYLESHEET.stat().st_mtime if STYLESHEET.exists() else 0.0
    css = _stylesheet(mtime)
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _hide_sample_b(context: AppContext | None = None, trial=None) -> bool:
    if trial is not None:
        return bool(getattr(trial, "hide_sample_b", False))
    if context is not None:
        return context.settings.hide_sample_b()
    return False


def render_header(context: AppContext) -> None:
    hide_b = context.settings.hide_sample_b()
    lede = (
        "Listen to the sample and rate what you hear."
        if hide_b
        else f"{APP_SUBTITLE} — compare two speech samples and rate what you hear."
    )
    st.markdown(
        '<div class="app-hero">'
        '<div class="app-hero-mark" aria-hidden="true">'
        '<span class="app-hero-bars"><i></i><i></i><i></i><i></i><i></i></span>'
        "</div>"
        '<div class="app-hero-copy">'
        '<p class="app-kicker">Listening study</p>'
        f"<h1>{escape(APP_TITLE)}</h1>"
        f'<p class="app-lede">{escape(lede)}</p>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="eval-note"><span class="eval-note-label">How it works</span>'
        f"{escape(tester_instructions(hide_sample_b=hide_b))}</div>",
        unsafe_allow_html=True,
    )


def render_flow_steps(trial, *, hide_sample_b: bool = False) -> None:
    """Three-step progress: setup, listen, rate."""
    listen_active = trial is not None and getattr(trial, "is_ready", False)
    listen_label = "Listen to the sample" if hide_sample_b else "Listen to both samples"
    steps = (
        ("1", "Choose language & sentence", "done" if listen_active else "active"),
        ("2", listen_label, "active" if listen_active else "pending"),
        ("3", "Submit your ratings", "active" if listen_active else "pending"),
    )
    chips = "".join(
        f'<div class="flow-step flow-step--{state}">'
        f'<span class="flow-step-num">{num}</span>'
        f"<span>{escape(label)}</span></div>"
        for num, label, state in steps
    )
    st.markdown(f'<div class="flow-steps">{chips}</div>', unsafe_allow_html=True)


def _section_title(text: str, *, step: str | None = None) -> None:
    badge = f'<span class="section-step">{escape(step)}</span>' if step else ""
    st.markdown(
        f'<div class="eval-section-title">{badge}<span>{escape(text)}</span></div>',
        unsafe_allow_html=True,
    )


def render_tester_identity(context: AppContext) -> str:
    """Name field. Stored in the `tester_id` column either way."""
    mode = context.settings.tester_id_mode
    count = st.session_state["submitted_count"]
    name_col, count_col = st.columns([2, 1])

    with name_col:
        if mode == "auto":
            st.text_input("Session ID", value=st.session_state["tester_id"], disabled=True)
        else:
            st.session_state["tester_id"] = st.text_input(
                "Your name",
                value=st.session_state["tester_id"],
                help="Used to attribute your ratings.",
                placeholder="e.g. Amina O.",
            ).strip()
            if mode == "required" and not st.session_state["tester_id"]:
                st.caption(
                    "Add your name, then play the sample."
                    if context.settings.hide_sample_b()
                    else "Add your name, then play a pair."
                )
            elif mode == "prompt":
                st.caption("Use your name, or keep the generated ID to stay anonymous.")

    with count_col:
        if count:
            st.markdown(
                f'<div class="eval-note" style="margin:1.7rem 0 0">'
                f"{count} rating{'s' if count != 1 else ''} sent. You can stop anytime.</div>",
                unsafe_allow_html=True,
            )

    return st.session_state["tester_id"]


@dataclass(frozen=True)
class VoiceSelection:
    """Language and voice the listener picked. Speaker IDs stay off-screen."""

    mode: str
    language: Language | None = None
    accent_key: str | None = None
    speaker_id: str | None = None
    error: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.language is not None and self.speaker_id is not None and self.error is None


def render_voice_selection(context: AppContext) -> VoiceSelection:
    """Language and male/female voice. One speaker is chosen automatically."""
    catalog = context.catalog
    st.session_state["selection_mode"] = "researcher"

    language_col, voice_col = st.columns(2)
    with language_col:
        language_key = _stable_selectbox(
            st,
            "Language",
            list(catalog.language_keys()),
            "sel_language",
            format_func=lambda key: catalog.language(key).label,
        )
    language = catalog.language(language_key)

    accent_key = None
    with voice_col:
        if language.has_accents:
            accent_key = _stable_selectbox(
                st,
                "Accent",
                [accent.key for accent in language.accents],
                "sel_accent",
                format_func=lambda key: language.accent(key).label,
            )
            speakers = catalog.speakers_for(language.key, accent=accent_key)
        else:
            gender = _stable_selectbox(
                st,
                "Voice",
                list(catalog.genders(language.key)),
                "sel_gender",
                format_func=str.capitalize,
            )
            speakers = catalog.speakers_for(language.key, gender=gender)

    if not speakers:
        message = "No voice is configured for this language yet."
        st.error(message)
        return VoiceSelection(
            mode="researcher", language=language, accent_key=accent_key, error=message
        )

    return VoiceSelection(
        mode="researcher",
        language=language,
        accent_key=accent_key,
        speaker_id=speakers[0].speaker_id,
    )


# ---------------------------------------------------------------------------
# Main-area test setup: sentence + generation mode
# ---------------------------------------------------------------------------
def _render_sentence_controls(context: AppContext, selection: VoiceSelection):
    """One sentence at a time. Returns a Sentence or None."""
    catalog = context.catalog
    language = selection.language
    assert language is not None

    source = st.radio(
        "Sentence",
        options=list(SENTENCE_SOURCES),
        format_func=lambda value: SENTENCE_SOURCE_LABELS[value],
        horizontal=True,
        key="sentence_source",
    )

    if source == "custom":
        text = st.text_area(
            "What should they say?",
            key="custom_text",
            height=90,
            placeholder="Type one sentence in the language you picked.",
        )
        if not (text or "").strip():
            st.caption(
                "Type a sentence, then play the sample."
                if context.settings.hide_sample_b()
                else "Type a sentence, then play both samples."
            )
            return None
        try:
            return make_custom_sentence(text, language.key, selection.accent_key)
        except ConfigError as exc:
            st.warning(str(exc))
            return None

    sentences = catalog.sentences_for(language.key, accent=selection.accent_key)
    if not sentences:
        st.error("No sentence is configured for this language yet.")
        return None

    ids, _scope = _sync_sentence_for_language(catalog, language.key, selection.accent_key)
    sentence = catalog.sentence(st.session_state["sel_sentence"])
    index = ids.index(sentence.sentence_id)

    st.markdown(
        f'<div class="eval-sentence"><span class="eval-sentence-label">Sentence</span>'
        f"{escape(sentence.text)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sentence-nav-caption">Sentence {index + 1} of {len(ids)}. '
        f"{'Rate this sample' if context.settings.hide_sample_b() else 'Rate this pair'}, "
        "then try another sentence if you want to keep going.</p>",
        unsafe_allow_html=True,
    )
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("Previous sentence", use_container_width=True, disabled=len(ids) < 2):
            st.session_state["pending_sentence_id"] = ids[(index - 1) % len(ids)]
            st.rerun()
    with next_col:
        if st.button("Next sentence", use_container_width=True, disabled=len(ids) < 2):
            st.session_state["pending_sentence_id"] = ids[(index + 1) % len(ids)]
            st.rerun()
    return sentence


def _render_voice_mode_controls() -> tuple[str, ReferenceAudio | None, str, str | None]:
    """Preset vs clone, with the cloning inputs revealed inline.

    Returns (generation_mode, reference_audio, reference_text, error).
    """
    mode = st.radio(
        "Voice",
        options=list(VOICE_MODES),
        format_func=lambda value: VOICE_MODE_LABELS[value],
        horizontal=True,
        key="voice_mode",
    )

    if mode == PRESET_MODE:
        st.caption("Uses the voice you picked above. No recording is needed.")
        return PRESET_MODE, None, "", None

    st.caption(
        "Press record and read the sentence above (a few seconds of clean speech is enough). "
        "Your microphone is only used when you press record."
    )
    recorded = st.audio_input(
        "Record your voice",
        sample_rate=REFERENCE_SAMPLE_RATE,
        key="clone_recording",
    )

    uploaded = st.file_uploader(
        "Or upload an audio file instead",
        type=UPLOAD_TYPES,
        key="clone_upload",
    )
    transcript = st.text_input(
        "Transcript of the recording (optional)",
        key="clone_transcript",
        placeholder="Leave blank if you do not have one.",
    )

    # A fresh recording wins over a previously uploaded file.
    reference: ReferenceAudio | None = None
    if recorded is not None:
        reference = ReferenceAudio.from_upload(
            recorded.getvalue(),
            getattr(recorded, "name", None) or "recording.wav",
            getattr(recorded, "type", "") or "audio/wav",
        )
        if uploaded is not None:
            st.caption("Your recording is used.")
    elif uploaded is not None:
        reference = ReferenceAudio.from_upload(
            uploaded.getvalue(), uploaded.name, getattr(uploaded, "type", "") or "audio/wav"
        )

    if reference is None:
        return (
            CLONE_MODE,
            None,
            transcript or "",
            "Record your voice (or add a file) to clone a voice.",
        )
    return CLONE_MODE, reference, transcript or "", None


def render_test_setup(
    context: AppContext, selection: VoiceSelection
) -> tuple[TestCondition | None, bool]:
    """Main-area controls. Returns (condition, load requested)."""
    _section_title("Setup", step="1")
    with st.container(border=True):
        if not selection.is_usable:
            if selection.error is None:
                st.error("Pick a language and a voice.")
            return None, False

        sentence = _render_sentence_controls(context, selection)
        st.divider()

        generation_mode, reference, transcript, voice_error = _render_voice_mode_controls()
        if voice_error:
            st.caption(voice_error)

        blocked = voice_error is not None or sentence is None
        play_label = "Play sample" if context.settings.hide_sample_b() else "Play both samples"
        load = st.button(
            play_label, type="primary", use_container_width=True, disabled=blocked
        )

    if blocked:
        return None, False

    try:
        assert selection.language is not None
        condition = evaluation.build_condition(
            context.catalog,
            language_key=selection.language.key,
            speaker_id=selection.speaker_id or "",
            sentence=sentence,
            accent_key=selection.accent_key,
            generation_mode=generation_mode,
            reference_audio=reference,
            reference_text=transcript,
        )
    except (evaluation.EvaluationError, ConfigError) as exc:
        st.error(str(exc))
        return None, False

    return condition, load


# ---------------------------------------------------------------------------
# Trial handling
# ---------------------------------------------------------------------------
def load_trial(context: AppContext, condition: TestCondition) -> None:
    st.session_state["saved_message"] = None
    trial = evaluation.new_trial_for(context.settings, condition)
    spinner = (
        "Preparing the sample. This can take a moment..."
        if trial.hide_sample_b
        else "Preparing the samples. This can take a moment..."
    )
    with st.spinner(spinner):
        try:
            evaluation.prepare_samples(trial, context.clients, context.cache)
        except SampleGenerationError as exc:
            st.session_state["trial"] = None
            st.session_state["generation_error"] = exc.message
            return
    st.session_state["trial"] = trial
    st.session_state["generation_error"] = None
    st.session_state["saved_message"] = None
    st.session_state["trial_started_at"] = time.time()


def _rating_key(trial_id: str, name: str) -> str:
    return f"rating::{trial_id}::{name}"


def clear_trial_state(trial_id: str) -> None:
    """Drop the finished trial; its rating widgets are purged on the next run."""
    st.session_state["stale_rating_prefix"] = f"rating::{trial_id}::"
    st.session_state["trial"] = None
    st.session_state["trial_started_at"] = None


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
def render_condition_panel(trial) -> None:
    condition = trial.condition
    chips = [("Language", condition.language.label)]
    if condition.accent_label:
        chips.append(("Accent", condition.accent_label))
    if condition.is_clone:
        chips.append(("Voice", condition.mode_label))
        if condition.reference_audio is not None:
            chips.append(("Recording", condition.reference_audio.name))
    else:
        chips.append(("Preset voice", condition.speaker.neutral_label))
        chips.append(("Voice mode", condition.mode_label))

    _section_title("What you are rating", step="2")
    with st.container(border=True):
        chip_html = "".join(
            f'<span class="eval-chip"><span class="eval-chip-key">{escape(key)}</span>'
            f"{escape(value)}</span>"
            for key, value in chips
        )
        st.markdown(f'<div class="eval-chips">{chip_html}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="eval-sentence eval-sentence--compact">'
            f"{escape(condition.sentence.text)}</div>",
            unsafe_allow_html=True,
        )


def render_samples(trial) -> None:
    hide_b = trial.hide_sample_b
    labels = trial.visible_labels
    _section_title("Listen to the sample" if hide_b else "Listen to both samples", step="2")
    if not hide_b and any(sample.mocked for sample in trial.samples.values()):
        st.caption(
            "One sample may use placeholder audio (a synthetic tone) until the second TTS "
            "system is deployed. The other sample should be real speech when the live "
            "endpoint is configured."
        )
    st.markdown('<div class="sample-grid">', unsafe_allow_html=True)
    columns = st.columns(len(labels), gap="large")
    for column, label in zip(columns, labels):
        sample = trial.sample(label)
        with column:
            with st.container(border=True):
                st.markdown(
                    f'<div class="sample-card">'
                    f'<div class="sample-card-head">'
                    f'<span class="sample-badge">{label}</span>'
                    f'<div class="sample-card-copy">'
                    f'<span class="sample-card-kicker">Blinded sample</span>'
                    f'<span class="sample-card-title">Sample {label}</span>'
                    f"</div></div>"
                    f'<div class="sample-eq" aria-hidden="true">'
                    f"<i></i><i></i><i></i><i></i><i></i></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.audio(sample.clip.data, format=sample.clip.mime_type)
                st.caption("Replay as often as you like. Nothing plays automatically.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_rating_form(context: AppContext, trial, tester_id: str, mode: str) -> None:
    trial_id = trial.trial_id
    _section_title("Your ratings", step="3")

    hide_b = trial.hide_sample_b
    labels = trial.visible_labels

    with st.form(key=f"evaluation_form_{trial_id}", clear_on_submit=False, border=False):
        preference = None
        if not hide_b:
            with st.container(border=True):
                st.markdown(
                    '<div class="criterion-title">Which sample sounds better overall?</div>',
                    unsafe_allow_html=True,
                )
                preference = st.radio(
                    "Overall preference",
                    options=list(PREFERENCE_CHOICES),
                    format_func=lambda value: PREFERENCE_LABELS[value],
                    index=None,
                    horizontal=True,
                    label_visibility="collapsed",
                    key=_rating_key(trial_id, "preference"),
                )

        scores: dict[str, int | None] = {}
        for title, slug, help_text in rating_criteria(trial.condition.is_clone):
            with st.container(border=True):
                st.markdown(
                    f'<div class="criterion-title">{escape(title)}</div>'
                    f'<div class="criterion-help">{escape(help_text)}</div>'
                    f'<div class="criterion-scale">{escape(SCALE_LEGEND)}</div>',
                    unsafe_allow_html=True,
                )
                columns = st.columns(len(labels), gap="large")
                for column, label in zip(columns, labels):
                    with column:
                        st.markdown(
                            f'<div class="rating-side">Sample {label}</div>',
                            unsafe_allow_html=True,
                        )
                        scores[f"{slug}_{label}"] = st.radio(
                            f"{title} - Sample {label}",
                            options=list(RATING_SCALE),
                            index=None,
                            horizontal=True,
                            label_visibility="collapsed",
                            key=_rating_key(trial_id, f"{slug}_{label}"),
                        )

        comments = st.text_area(
            "Additional comments (optional)",
            key=_rating_key(trial_id, "comments"),
            placeholder=(
                "Anything you noticed about the sample."
                if hide_b
                else "Anything you noticed about either sample."
            ),
        )
        confirmed = st.checkbox(
            "I listened to the sample in full." if hide_b else "I listened to both samples in full.",
            key=_rating_key(trial_id, "confirmed"),
        )
        submitted = st.form_submit_button(
            "Submit Evaluation", type="primary", use_container_width=True
        )

    if not submitted:
        return

    ratings = Ratings(
        preference=preference,
        naturalness_a=scores.get("naturalness_A"),
        naturalness_b=scores.get("naturalness_B"),
        pronunciation_a=scores.get("pronunciation_A"),
        pronunciation_b=scores.get("pronunciation_B"),
        speaker_similarity_a=scores.get("similarity_A"),
        speaker_similarity_b=scores.get("similarity_B"),
        comments=comments or "",
        listened_to_both=bool(confirmed),
    )

    problems = evaluation.validate_ratings(
        ratings,
        voice_criterion_label=(
            CRITERIA_VOICE_CLONE[0] if trial.condition.is_clone else CRITERIA_VOICE_PRESET[0]
        ),
        hide_sample_b=trial.hide_sample_b,
    )
    if context.settings.tester_id_mode == "required" and not tester_id:
        problems.insert(0, "Please enter your name.")
    if problems:
        for problem in problems:
            st.error(problem)
        return

    started_at = st.session_state.get("trial_started_at")
    row = evaluation.build_result_row(
        trial,
        ratings,
        tester_id=tester_id or "anonymous",
        session_id=st.session_state["session_id"],
        selection_mode=mode,
        app_version=context.settings.app_version,
        listening_seconds=time.time() - started_at if started_at else None,
    )

    try:
        context.store.save(row)
    except StorageError as exc:
        settings_module.configure_logging(context.settings).error("Could not save evaluation: %s", exc)
        st.error("Your evaluation could not be saved. Please try submitting again.")
        return

    st.session_state["submitted_count"] += 1
    st.session_state["seen_conditions"].append(trial.condition.key)
    _advance_selection(context, trial)
    clear_trial_state(trial_id)
    st.session_state["saved_message"] = (
        "Saved. Try another sentence if you like, or stop whenever you're done."
    )
    st.rerun()


def _advance_selection(context: AppContext, trial) -> None:
    """Queue the next sentence of the same condition for the next run.

    The sentence picker is widget-backed, so it can only be updated before the
    widget is created - `init_session` applies the queued value.
    """
    if st.session_state.get("selection_mode") != "researcher":
        return
    if st.session_state.get("sentence_source") == "custom":
        return
    condition = trial.condition
    sentences = context.catalog.sentences_for(condition.language.key, accent=condition.accent_key)
    sentence_ids = [sentence.sentence_id for sentence in sentences]
    if condition.sentence.sentence_id not in sentence_ids:
        return
    index = sentence_ids.index(condition.sentence.sentence_id)
    st.session_state["pending_sentence_id"] = sentence_ids[(index + 1) % len(sentence_ids)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()

    try:
        context = get_context(_config_stamp(load_settings().config_dir), _runtime_stamp())
    except ConfigError as exc:
        st.title(APP_TITLE)
        st.error("The evaluation configuration could not be loaded.")
        st.code(str(exc), language="text")
        st.stop()
        return

    init_session(context)
    researcher_view.render_login(context.settings)

    hide_b = context.settings.hide_sample_b()
    render_header(context)
    render_flow_steps(st.session_state.get("trial"), hide_sample_b=hide_b)

    tester_id = render_tester_identity(context)
    selection = render_voice_selection(context)
    condition, load_requested = render_test_setup(context, selection)
    mode = selection.mode

    if load_requested and condition is not None:
        if context.settings.tester_id_mode == "required" and not tester_id:
            st.warning(
                "Please enter your name before playing the sample."
                if hide_b
                else "Please enter your name before playing the samples."
            )
        else:
            load_trial(context, condition)

    if st.session_state["saved_message"]:
        st.success(st.session_state["saved_message"])

    if st.session_state["generation_error"]:
        st.error(st.session_state["generation_error"])
        st.caption("The technical details were written to the application log.")

    trial = st.session_state["trial"]
    if trial is None:
        play_label = "Play sample" if hide_b else "Play both samples"
        idle_copy = (
            f"Press <strong>{play_label}</strong> above when your setup is complete. "
            + (
                "The clip appears here before you rate it."
                if hide_b
                else "Both clips appear here before you rate them."
            )
        )
        ghosts = '<div class="sample-ghost"><span>A</span><div class="ghost-wave"></div></div>'
        if not hide_b:
            ghosts += '<div class="sample-ghost"><span>B</span><div class="ghost-wave"></div></div>'
        st.markdown(
            '<div class="idle-panel">'
            '<div class="idle-icon" aria-hidden="true">'
            '<span class="app-hero-bars"><i></i><i></i><i></i><i></i><i></i></span>'
            "</div>"
            "<h2>Ready when you are</h2>"
            f"<p>{idle_copy}</p>"
            f'<div class="listen-preview">{ghosts}</div></div>',
            unsafe_allow_html=True,
        )
    elif not trial.is_ready:
        st.session_state["trial"] = None
        st.warning("The samples are no longer available. Please load the test again.")
    else:
        render_condition_panel(trial)
        render_samples(trial)
        if condition is not None and condition.selection_key != trial.condition.selection_key:
            st.warning(
                "Your setup above has changed. You are still rating the samples shown here - "
                "load the new test when you are ready."
            )
        render_rating_form(context, trial, tester_id, mode)

    researcher_view.render_dashboard(
        settings=context.settings,
        catalog=context.catalog,
        store=context.store,
        cache=context.cache,
    )


if __name__ == "__main__":
    main()
