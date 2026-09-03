"""Headless smoke test of the Streamlit UI (requires streamlit installed).

    python tools/ui_smoketest.py

Runs app.py through Streamlit's AppTest harness in MOCK_MODE against a
throwaway database and cache: loads a test, checks that no system-identifying
text is rendered, fills the rating form, submits it, and verifies the row was
written to the results database.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORKDIR = Path(tempfile.mkdtemp(prefix="cosyvoice-uitest-"))
os.environ.update(
    {
        "MOCK_MODE": "true",
        "TESTER_ID_MODE": "prompt",
        "ENABLE_RANDOM_MODE": "true",
        "RANDOMIZE_AB": "true",
        "ADMIN_PASSWORD": "smoketest-password",
        "RESULTS_DB": str(WORKDIR / "evaluations.db"),
        "AUDIO_CACHE_DIR": str(WORKDIR / "cache"),
        "LOG_FILE": str(WORKDIR / "app.log"),
        "INDIVIDUAL_ENDPOINT": "",
        "COMBINED_ENDPOINT": "",
        "HF_TOKEN": "",
    }
)

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from services.hf_endpoint import MockTTSClient, SynthesisRequest  # noqa: E402
from services.storage import ResultsStore  # noqa: E402

FORBIDDEN = ("individual", "combined", "model a", "model b", "endpoint", "checkpoint", "cosyvoice")

#: Stands in for a clip recorded in the browser - both arrive as WAV bytes.
REFERENCE_CLIP = MockTTSClient("reference").synthesize(
    SynthesisRequest(
        text="reference clip for the voice cloning test",
        language="igbo",
        language_label="Yoruba",
        language_code="YO",
        speaker_id="YO-F-01",
        gender="female",
        sentence_id="YO-001",
    )
).data

_failed: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [ok]   {description}")
    else:
        _failed.append(description)
        print(f"  [FAIL] {description}{(' - ' + detail) if detail else ''}")


def rendered_text(app: AppTest) -> str:
    chunks: list[str] = []
    for name in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success", "text"):
        try:
            elements = getattr(app, name)
        except Exception:  # pragma: no cover - element type absent in this version
            continue
        for element in elements:
            value = getattr(element, "value", None)
            if isinstance(value, str):
                chunks.append(value)
    for name in ("radio", "selectbox", "checkbox", "text_area", "text_input", "button"):
        try:
            elements = getattr(app, name)
        except Exception:  # pragma: no cover
            continue
        for element in elements:
            label = getattr(element, "label", None)
            if isinstance(label, str):
                chunks.append(label)
            for option in getattr(element, "options", []) or []:
                chunks.append(str(option))
    return " ".join(chunks).lower()


def widget_by_key_suffix(elements, suffix: str):
    for element in elements:
        key = getattr(element, "key", None)
        if key and str(key).endswith(suffix):
            return element
    raise AssertionError(f"No widget with key ending in {suffix!r}")


def optional_widget(app: AppTest, element_type: str, suffix: str):
    """Find a widget without failing if the element type is unsupported here."""
    try:
        elements = getattr(app, element_type)
    except Exception:  # pragma: no cover - element type absent in this version
        return None
    for element in elements:
        key = getattr(element, "key", None)
        if key and str(key).endswith(suffix):
            return element
    return None


def button_by_label(app: AppTest, label: str):
    for element in app.button:
        if label.lower() in str(getattr(element, "label", "")).lower():
            return element
    raise AssertionError(f"No button labelled {label!r}")


def main() -> int:
    print("Streamlit UI smoke test\n")
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180)
    app.run()
    check("app runs without exceptions", not app.exception, str(app.exception))
    if app.exception:
        return 1

    page = rendered_text(app)
    check("title is neutral", "multilingual speech evaluation" in page)
    check("instructions are shown", "listen to both samples before rating them" in page)
    check("no samples are loaded initially", app.session_state["trial"] is None)

    # --- The important controls live in the main area, not the sidebar -----
    def keys_of(elements) -> set[str]:
        return {str(getattr(element, "key", "") or "") for element in elements}

    sidebar_keys = keys_of(app.sidebar.selectbox) | keys_of(app.sidebar.radio) | keys_of(
        app.sidebar.text_area
    )
    all_keys = keys_of(app.selectbox) | keys_of(app.radio) | keys_of(app.text_area)
    check("the sentence picker is not a dropdown", "sel_sentence" not in all_keys)
    check("the sentence source choice is in the main area", "sentence_source" in all_keys and "sentence_source" not in sidebar_keys)
    check("the voice mode choice is in the main area", "voice_mode" in all_keys and "voice_mode" not in sidebar_keys)
    check("language is in the main area", "sel_language" in all_keys and "sel_language" not in sidebar_keys)
    check("the speaker id is not a listener control", "sel_speaker" not in all_keys)
    check(
        "the identity field asks for a name",
        any("your name" in str(getattr(el, "label", "")).lower() for el in app.text_input),
        str([getattr(el, "label", "") for el in app.text_input]),
    )
    check(
        "the old anonymity warning is gone",
        "do not enter your name" not in page,
    )
    check("preset voice is the default", app.session_state["voice_mode"] == "preset")

    # Select Igbo / female and play the samples.
    app.session_state["sel_language"] = "igbo"
    app.run()
    button_by_label(app, "Play both samples").click()
    app.run()

    check("no exception while generating samples", not app.exception, str(app.exception))
    trial = app.session_state["trial"]
    check("a trial is stored in session state", trial is not None)
    if trial is None:
        return 1
    check("both samples were prepared", trial.is_ready)
    check(
        "A/B mapping is randomised over the two systems",
        {trial.assignment.model_for_A, trial.assignment.model_for_B} == {"individual", "combined"},
    )
    check("the condition is Igbo", trial.condition.language.key == "igbo")
    check("preset mode sends no reference audio", trial.condition.reference_audio is None)
    check("preset mode sends no reference text", trial.condition.reference_text is None)
    check("both sides are flagged as placeholder audio in mock mode", all(s.mocked for s in trial.samples.values()))

    body = rendered_text(app)
    leaks = [word for word in FORBIDDEN if word in body]
    check("rendered page reveals no system identity", not leaks, str(leaks))
    check("Sample A and Sample B are both presented", "sample a" in body and "sample b" in body)

    assignment_before = (trial.assignment.model_for_A, trial.assignment.model_for_B)
    for _ in range(3):  # simulate incidental reruns
        app.run()
    trial_after = app.session_state["trial"]
    check(
        "A/B assignment survives reruns",
        (trial_after.assignment.model_for_A, trial_after.assignment.model_for_B) == assignment_before,
    )
    check("audio is not regenerated on rerun", trial_after.trial_id == trial.trial_id)

    # Submit without ratings -> validation errors, nothing saved.
    store = ResultsStore(Path(os.environ["RESULTS_DB"]))
    button_by_label(app, "Submit Evaluation").click()
    app.run()
    check("incomplete submission is rejected", len(app.error) >= 1)
    check("nothing was saved for an incomplete submission", store.count() == 0)
    check("the trial is still loaded after a failed submission", app.session_state["trial"] is not None)

    # Fill everything in and submit.
    widget_by_key_suffix(app.radio, "::preference").set_value("A")
    for slug in ("naturalness", "pronunciation", "similarity"):
        for label, value in (("A", 4), ("B", 3)):
            widget_by_key_suffix(app.radio, f"::{slug}_{label}").set_value(value)
    widget_by_key_suffix(app.text_area, "::comments").set_value("Smoke test comment.")
    widget_by_key_suffix(app.checkbox, "::confirmed").set_value(True)
    button_by_label(app, "Submit Evaluation").click()
    app.run()

    check("no exception on submit", not app.exception, str(app.exception))
    check("evaluation was saved", store.count() == 1, f"count={store.count()}")
    check("evaluation state was cleared", app.session_state["trial"] is None)
    check("submission counter increased", app.session_state["submitted_count"] == 1)
    check("confirmation is shown", any("saved" in s.value.lower() for s in app.success))

    row = store.rows()[0]
    check(
        "the internal mapping was recorded",
        (row["sample_A_model"], row["sample_B_model"]) == assignment_before,
        str((row["sample_A_model"], row["sample_B_model"])),
    )
    check("ratings were recorded", row["naturalness_A"] == 4 and row["naturalness_B"] == 3)
    check("preference was recorded", row["preferred_sample"] == "A")
    check("sentence advanced for the next test", app.session_state["sel_sentence"] != row["sentence_id"])

    # English is withheld until the endpoint supports it.
    language_box = widget_by_key_suffix(app.selectbox, "sel_language")
    check("English is not offered yet", "english" not in language_box.options, str(language_box.options))

    # --- Custom sentence typed in the main area ---------------------------
    app.session_state["sel_language"] = "igbo"
    app.run()
    widget_by_key_suffix(app.radio, "sentence_source").set_value("custom")
    app.run()
    custom_box = widget_by_key_suffix(app.text_area, "custom_text")
    check("a custom sentence box appears in the main area", custom_box is not None)
    custom_box.set_value("Mo fe gbo ohun tuntun ni ojo oni.")
    app.run()
    button_by_label(app, "Play both samples").click()
    app.run()
    custom_trial = app.session_state["trial"]
    check("no exception for a custom sentence", not app.exception, str(app.exception))
    check("a custom sentence loads", custom_trial is not None and custom_trial.is_ready)
    check(
        "the custom sentence gets a stable custom id",
        custom_trial.condition.sentence.sentence_id.startswith("custom-"),
        custom_trial.condition.sentence.sentence_id,
    )
    check(
        "the custom text is what gets synthesised",
        custom_trial.condition.sentence.text == "Mo fe gbo ohun tuntun ni ojo oni.",
    )
    check(
        "custom audio is cached under the custom id",
        custom_trial.condition.sentence.sentence_id in custom_trial.sample("A").cache_key,
        custom_trial.sample("A").cache_key,
    )
    custom_id = custom_trial.trial_id
    button_by_label(app, "Play both samples").click()
    app.run()
    reloaded = app.session_state["trial"]
    check(
        "reloading the same custom sentence reuses the cache",
        reloaded.trial_id != custom_id and all(s.from_cache for s in reloaded.samples.values()),
    )

    # --- Clone a voice, without a transcript ------------------------------
    widget_by_key_suffix(app.radio, "voice_mode").set_value("clone")
    app.run()
    # AppTest models no audio_input element, so look the widget up by key.
    try:
        recorder = app.get_by_key("clone_recording")
    except Exception as exc:  # pragma: no cover - widget missing
        recorder, exc_detail = None, str(exc)
    else:
        exc_detail = ""
    check("clone mode offers an in-browser recorder", recorder is not None, exc_detail)
    check(
        "the recorder is offered before the upload field",
        "record your voice" in rendered_text(app),
    )
    uploader = optional_widget(app, "file_uploader", "clone_upload")
    check("clone mode keeps a file upload fallback", uploader is not None)
    check(
        "no reference link field is offered",
        not any(str(getattr(f, "key", "")).endswith("clone_url") for f in app.text_input),
    )
    check(
        "clone mode reveals an optional transcript field",
        widget_by_key_suffix(app.text_input, "clone_transcript") is not None,
    )
    check(
        "clone inputs are in the main area",
        "clone_transcript" not in keys_of(app.sidebar.text_input),
    )
    # A browser recording and an upload both arrive as in-memory bytes, so an
    # upload is the closest thing AppTest can simulate for a recorded clip.
    uploader.set_value(("recording.wav", REFERENCE_CLIP, "audio/wav"))
    app.run()
    button_by_label(app, "Play both samples").click()
    app.run()
    clone_trial = app.session_state["trial"]
    check("no exception in clone mode", not app.exception, str(app.exception))
    check("clone mode loads a trial", clone_trial is not None and clone_trial.is_ready)
    check("clone mode is recorded on the condition", clone_trial.condition.generation_mode == "clone")
    check(
        "the reference audio reaches the request",
        clone_trial.condition.synthesis_request().reference_audio is not None,
    )
    check(
        "cloning works with no transcript",
        clone_trial.condition.reference_text is None,
    )
    request = clone_trial.condition.synthesis_request()
    check(
        "the recorded bytes are what get sent",
        request.reference_audio is not None
        and request.reference_audio.data == REFERENCE_CLIP,
    )
    check(
        "both samples share the identical reference audio",
        bool(clone_trial.condition.reference_id),
        str(clone_trial.condition.reference_id),
    )
    check(
        "cloned audio is cached separately from preset audio",
        "/clone-" in clone_trial.sample("A").cache_key,
        clone_trial.sample("A").cache_key,
    )
    check("clone mode hides system identity", not [w for w in FORBIDDEN if w in rendered_text(app)])

    widget_by_key_suffix(app.radio, "voice_mode").set_value("preset")
    widget_by_key_suffix(app.radio, "sentence_source").set_value("catalog")
    app.run()

    # Randomised selection mode.
    app.session_state["selection_mode"] = "random"
    app.run()
    check("randomised mode renders", not app.exception, str(app.exception))
    button_by_label(app, "Load next test").click()
    app.run()
    random_trial = app.session_state["trial"]
    check("randomised mode loads a trial", random_trial is not None and random_trial.is_ready)
    check(
        "randomised mode avoids already-seen conditions",
        random_trial.condition.key not in app.session_state["seen_conditions"],
    )
    check("randomised mode hides system identity", not [w for w in FORBIDDEN if w in rendered_text(app)])

    # Researcher dashboard is not part of the listener UI.
    check(
        "researcher dashboard is not shown",
        not any("Researcher dashboard" in element.value for element in app.subheader),
    )

    # A required-but-missing name blocks generation (the default mode).
    required_app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180)
    os.environ["TESTER_ID_MODE"] = "required"
    st.cache_resource.clear()  # the settings are cached per process
    try:
        required_app.run()
        check(
            "the name field starts empty when required",
            required_app.session_state["tester_id"] == "",
        )
        button_by_label(required_app, "Play both samples").click()
        required_app.run()
        check(
            "a missing name blocks loading",
            required_app.session_state["trial"] is None and len(required_app.warning) >= 1,
        )
        check(
            "the tester is asked for their name",
            "enter your name" in rendered_text(required_app),
        )
        required_app.sidebar.text_input[0].set_value("Ada Lovelace")
        required_app.run()
        button_by_label(required_app, "Play both samples").click()
        required_app.run()
        check(
            "loading proceeds once a name is given",
            required_app.session_state["trial"] is not None,
        )
    finally:
        os.environ["TESTER_ID_MODE"] = "prompt"
        st.cache_resource.clear()

    # A pinned A/B mapping keeps the configured system on Sample A.
    pinned_app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180)
    os.environ["RANDOMIZE_AB"] = "false"
    os.environ["SAMPLE_A_SYSTEM"] = "individual"
    st.cache_resource.clear()
    try:
        pinned_app.run()
        mappings = set()
        for _ in range(6):
            button_by_label(pinned_app, "Play both samples").click()
            pinned_app.run()
            pinned_trial = pinned_app.session_state["trial"]
            mappings.add((pinned_trial.assignment.model_for_A, pinned_trial.assignment.model_for_B))
        check(
            "a pinned mapping never swaps the samples",
            mappings == {("individual", "combined")},
            str(mappings),
        )
        check(
            "a pinned trial is marked as not randomised",
            not pinned_app.session_state["trial"].assignment.randomized,
        )
        check(
            "pinning still hides system identity from the tester",
            not [w for w in FORBIDDEN if w in rendered_text(pinned_app)],
        )
    finally:
        os.environ["RANDOMIZE_AB"] = "true"
        os.environ.pop("SAMPLE_A_SYSTEM", None)
        st.cache_resource.clear()

    print("\n" + ("All UI checks passed." if not _failed else f"{len(_failed)} UI check(s) failed."))
    return 1 if _failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(WORKDIR, ignore_errors=True)
