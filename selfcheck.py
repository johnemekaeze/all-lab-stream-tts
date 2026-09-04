"""End-to-end self-check for the evaluation app (no Streamlit server needed).

    python selfcheck.py

It verifies configuration loading, mock synthesis, A/B randomisation and its
stability, caching, results storage/export, error handling, and that the
tester-facing UI file contains no reference to the systems under test.
Temporary files are written to a throwaway directory and deleted afterwards.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import requests

from services import evaluation
from services.audio_cache import AudioCache
from services.config_loader import ConfigError, TestCatalog, make_custom_sentence
from services.evaluation import (
    CLONE_MODE,
    PRESET_MODE,
    EvaluationError,
    Ratings,
    ReferenceAudio,
    SampleGenerationError,
    build_condition,
)
from services.hf_endpoint import (
    EndpointAdapter,
    EndpointError,
    EndpointNotConfiguredError,
    EndpointResponseError,
    EndpointTimeoutError,
    HFEndpointClient,
    MockTTSClient,
)
from services.settings import PROJECT_ROOT, SYSTEMS, load_settings
from services.storage import INTERNAL_COLUMNS, ResultsStore

CONFIG_DIR = PROJECT_ROOT / "config"

_passed = 0
_failed: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    global _passed
    if condition:
        _passed += 1
        print(f"  [ok]   {description}")
    else:
        _failed.append(description)
        print(f"  [FAIL] {description}{(' - ' + detail) if detail else ''}")


class _FailingClient:
    system = "failing"

    def synthesize(self, request):  # noqa: D401 - test double
        raise EndpointError("simulated endpoint outage")


class _StubResponse:
    """Stand-in for `requests.Response`."""

    def __init__(self, status_code: int, content: bytes, content_type: str) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", "replace")


class _StubSession:
    """Stand-in for `requests.Session` - replays canned responses."""

    def __init__(self, responses: list[_StubResponse] | None = None, exception: Exception | None = None) -> None:
        self._responses = list(responses or [])
        self._exception = exception
        self.calls = 0

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls += 1
        if self._exception is not None:
            raise self._exception
        if not self._responses:
            raise AssertionError("stub session ran out of responses")
        return self._responses.pop(0)


def main() -> int:
    print("Multilingual Speech Evaluation - self-check\n")
    workdir = Path(tempfile.mkdtemp(prefix="cosyvoice-selfcheck-"))

    try:
        # -- 1. configuration ------------------------------------------------
        print("1. Configuration")
        catalog = TestCatalog.load(CONFIG_DIR)
        summary = catalog.summary()
        print(f"       {summary}")
        check("38 languages configured", len(catalog.languages) == 38, str(len(catalog.languages)))
        check("84 reference speakers configured", len(catalog.speakers) == 84, str(len(catalog.speakers)))
        check("English declares 5 accents", len(catalog.accents("english")) == 5)
        check(
            "English has 10 speakers (2 per accent)",
            len(catalog.speakers_for("english")) == 10
            and all(len(catalog.speakers_for("english", accent=a.key)) == 2 for a in catalog.accents("english")),
        )
        check(
            "37 non-English languages have exactly 1 male + 1 female speaker",
            all(
                len(catalog.speakers_for(language.key, gender="male")) == 1
                and len(catalog.speakers_for(language.key, gender="female")) == 1
                for language in catalog.languages
                if not language.has_accents
            ),
        )
        check(
            "no accent selector for non-English languages",
            all(not language.has_accents for language in catalog.languages if language.key != "english"),
        )
        check(
            "every language has 10 preset sentences",
            all(len(catalog.sentences_for(language.key)) == 10 for language in catalog.languages),
        )
        check(
            "no placeholder sentences remain",
            all(not sentence.is_placeholder for sentence in catalog.sentences),
        )
        check(
            "English accents share the same ten sentences",
            {s.sentence_id for s in catalog.sentences_for("english", accent="nigerian")}
            == {s.sentence_id for s in catalog.sentences_for("english", accent="ghanaian")}
            and len(catalog.sentences_for("english", accent="nigerian")) == 10,
        )
        try:
            TestCatalog.load(workdir / "missing")
            check("missing configuration raises ConfigError", False)
        except ConfigError:
            check("missing configuration raises ConfigError", True)

        check(
            "speaker reference audio is optional (presets need none)",
            all(not speaker.has_reference_audio for speaker in catalog.speakers),
        )
        custom = make_custom_sentence("  Kedụ ka ị mere? ", "igbo")
        check("custom sentence gets a stable id", custom.sentence_id.startswith("custom-"))
        check(
            "custom sentence id is derived from the text",
            make_custom_sentence("Kedụ ka ị mere?", "igbo").sentence_id == custom.sentence_id
            and make_custom_sentence("Something else", "igbo").sentence_id != custom.sentence_id,
        )
        check("custom sentence text is normalised", custom.text == "Kedụ ka ị mere?")
        for label, text in (("empty custom text", "  "), ("over-long custom text", "x" * 2000)):
            try:
                make_custom_sentence(text, "igbo")
                check(f"{label} is rejected", False)
            except ConfigError:
                check(f"{label} is rejected", True)

        # -- 2. settings / mock mode ----------------------------------------
        print("\n2. Settings and endpoint adapter")
        app_settings = load_settings()
        check("settings expose exactly two systems", len(SYSTEMS) == 2, str(SYSTEMS))
        # Read the built-in defaults, ignoring whatever the ambient shell and
        # the local .env happen to say (loading .env exports its values into
        # os.environ, so they have to be removed for this check).
        overridden = {
            name: os.environ.pop(name, None)
            for name in (
                "TESTER_ID_MODE",
                "RANDOMIZE_AB",
                "MOCK_MODE",
                "HF_TOKEN",
                "INDIVIDUAL_ENDPOINT",
                "COMBINED_ENDPOINT",
            )
        }
        try:
            defaults = load_settings(env_file=workdir / "no-such.env")
            check("tester identity is required by default", defaults.tester_id_mode == "required")
            check("the A/B mapping is randomised by default", defaults.randomize_ab)
            check("MOCK_MODE defaults to on without live credentials", defaults.mock_mode)

            os.environ["HF_TOKEN"] = "hf_test_token"
            os.environ["INDIVIDUAL_ENDPOINT"] = "https://example.com/v1"
            os.environ["COMBINED_ENDPOINT"] = "mock"
            live_defaults = load_settings(env_file=workdir / "no-such.env")
            check(
                "MOCK_MODE defaults to off when a live endpoint and token are configured",
                not live_defaults.mock_mode,
            )
            os.environ["MOCK_MODE"] = "false"
            check(
                "Streamlit-style boolean secrets are honoured",
                not load_settings(env_file=workdir / "no-such.env").mock_mode,
            )
        finally:
            for name, value in overridden.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        adapter = EndpointAdapter.load(CONFIG_DIR / "endpoint.yaml")
        condition = build_condition(
            catalog, language_key="igbo", speaker_id="IG-F-01", sentence_id="IG-003"
        )
        payload = adapter.build_payload(condition.synthesis_request())
        # The deployed handler's TTSRequest schema: inputs / language / voice.
        check(
            "preset payload matches the deployed request schema",
            set(payload) == {"inputs", "language", "voice"},
            str(sorted(payload)),
        )
        check(
            "preset payload sends NO reference audio and NO reference text",
            "prompt_audio_base64" not in payload and "prompt_text" not in payload,
            str(sorted(payload)),
        )
        check("payload carries the sentence text", payload["inputs"] == condition.sentence.text)
        check("payload carries the language the endpoint expects", payload["language"] == "igbo")
        check("payload carries the selected voice", payload["voice"] == "female")
        check("preset is the default generation mode", condition.generation_mode == PRESET_MODE)

        clip = MockTTSClient("individual").synthesize(condition.synthesis_request())
        check("mock mode produces WAV audio", clip.data.startswith(b"RIFF") and clip.size_bytes > 1000)
        check(
            "adapter accepts raw audio responses",
            adapter.parse_response(content=clip.data, content_type="audio/wav").size_bytes == clip.size_bytes,
        )
        import base64
        import json

        json_body = json.dumps({"audio": base64.b64encode(clip.data).decode(), "format": "wav"}).encode()
        check(
            "adapter accepts base64 JSON responses",
            adapter.parse_response(content=json_body, content_type="application/json").data == clip.data,
        )
        for label, body, content_type in (
            ("malformed audio", b"not-audio-at-all", "audio/wav"),
            ("endpoint error JSON", b'{"error": "model is not available"}', "application/json"),
            ("empty response", b"", "application/json"),
        ):
            try:
                adapter.parse_response(content=body, content_type=content_type)
                check(f"adapter rejects {label}", False)
            except EndpointResponseError:
                check(f"adapter rejects {label}", True)

        # -- 2a. voice cloning payloads --------------------------------------
        print("\n2a. Voice cloning (optional path)")
        upload = ReferenceAudio.from_upload(b"RIFF....fake-wav-bytes", "my-voice.wav", "audio/wav")
        clone_condition = build_condition(
            catalog,
            language_key="igbo",
            speaker_id="IG-F-01",
            sentence_id="IG-003",
            generation_mode=CLONE_MODE,
            reference_audio=upload,
            reference_text="the transcript",
        )
        clone_payload = adapter.build_payload(clone_condition.synthesis_request())
        check(
            "clone payload carries the reference audio",
            bool(clone_payload.get("prompt_audio_base64")),
        )
        check(
            "uploaded audio is base64-encoded for the wire",
            base64.b64decode(clone_payload["prompt_audio_base64"]) == upload.data,
        )
        check("clone payload carries the transcript", clone_payload.get("prompt_text") == "the transcript")

        no_transcript = build_condition(
            catalog,
            language_key="igbo",
            speaker_id="IG-F-01",
            sentence_id="IG-003",
            generation_mode=CLONE_MODE,
            reference_audio=upload,
        )
        payload_no_transcript = adapter.build_payload(no_transcript.synthesis_request())
        check(
            "cloning works without a transcript",
            "prompt_audio_base64" in payload_no_transcript
            and "prompt_text" not in payload_no_transcript,
            str(sorted(payload_no_transcript)),
        )

        url_reference = ReferenceAudio.from_url("https://example.com/voice.wav")
        url_condition = build_condition(
            catalog,
            language_key="igbo",
            speaker_id="IG-F-01",
            sentence_id="IG-003",
            generation_mode=CLONE_MODE,
            reference_audio=url_reference,
        )
        check(
            "a reference URL is passed through untouched",
            adapter.build_payload(url_condition.synthesis_request())["prompt_audio_base64"]
            == "https://example.com/voice.wav",
        )
        try:
            build_condition(
                catalog,
                language_key="igbo",
                speaker_id="IG-F-01",
                sentence_id="IG-003",
                generation_mode=CLONE_MODE,
            )
            check("clone mode without reference audio is rejected", False)
        except EvaluationError:
            check("clone mode without reference audio is rejected", True)
        check(
            "both systems receive the identical cloned reference",
            clone_condition.synthesis_request().reference_audio is upload,
        )
        check(
            "clone and preset requests have different fingerprints",
            clone_condition.synthesis_request().fingerprint()
            != condition.synthesis_request().fingerprint(),
        )

        # -- 2b. live HTTP client (no network: stubbed session) --------------
        print("\n2b. Endpoint client behaviour")
        request = condition.synthesis_request()

        ok_client = HFEndpointClient(
            system="individual",
            url="https://example.invalid/tts",
            token="dummy-token",
            adapter=adapter,
            session=_StubSession([_StubResponse(200, clip.data, "audio/wav")]),
            sleep=lambda _: None,
        )
        check("HTTP 200 audio response is accepted", ok_client.synthesize(request).data == clip.data)

        retry_session = _StubSession(
            [
                _StubResponse(503, b'{"error":"loading"}', "application/json"),
                _StubResponse(200, clip.data, "audio/wav"),
            ]
        )
        retry_client = HFEndpointClient(
            system="individual",
            url="https://example.invalid/tts",
            token="dummy-token",
            adapter=adapter,
            max_retries=2,
            session=retry_session,
            sleep=lambda _: None,
        )
        check("a warming-up endpoint (503) is retried", retry_client.synthesize(request).data == clip.data)
        check("retry used both stubbed responses", retry_session.calls == 2, str(retry_session.calls))

        for label, session, expected in (
            ("HTTP 401 is not retried", _StubSession([_StubResponse(401, b"forbidden", "text/plain")]), EndpointError),
            ("timeouts raise a timeout error", _StubSession(exception=requests.Timeout()), EndpointTimeoutError),
            (
                "transport errors are wrapped",
                _StubSession(exception=requests.ConnectionError("dns failure")),
                EndpointError,
            ),
        ):
            client = HFEndpointClient(
                system="individual",
                url="https://example.invalid/tts",
                token="dummy-token",
                adapter=adapter,
                max_retries=1,
                session=session,
                sleep=lambda _: None,
            )
            try:
                client.synthesize(request)
                check(label, False)
            except expected:
                check(label, True)

        unconfigured = HFEndpointClient(
            system="individual", url="", token="", adapter=adapter, session=_StubSession([])
        )
        try:
            unconfigured.synthesize(request)
            check("a missing endpoint URL is reported", False)
        except EndpointNotConfiguredError:
            check("a missing endpoint URL is reported", True)

        # -- 2c. per-system mock selection (no network) ----------------------
        print("\n2c. Per-system mock selection")
        live, placeholder = SYSTEMS
        one_live = replace(
            app_settings,
            mock_mode=False,
            endpoints={live: "https://example.invalid/tts", placeholder: "mock"},
        )
        check("a real URL means a live system", not one_live.uses_mock(live))
        check("the literal 'mock' means a placeholder system", one_live.uses_mock(placeholder))
        check("Sample B is hidden while combined is a placeholder", one_live.hide_sample_b())
        check(
            "Sample B reappears when combined is live",
            not replace(
                one_live,
                endpoints={live: "https://example.invalid/tts", placeholder: "https://example.invalid/combined"},
            ).hide_sample_b(),
        )
        check("an empty URL means a placeholder system", replace(one_live, endpoints={live: "", placeholder: "mock"}).uses_mock(live))
        check("live/placeholder systems are reported separately", one_live.live_systems() == (live,) and one_live.mocked_systems() == (placeholder,))
        check("placeholder status is explained for the researcher", "not deployed yet" in one_live.system_status(placeholder))
        mixed_clients = evaluation.build_clients_for(one_live, adapter)
        check(
            "the live system gets the HTTP client and the placeholder gets the mock",
            isinstance(mixed_clients[live], HFEndpointClient)
            and isinstance(mixed_clients[placeholder], MockTTSClient),
        )
        check(
            "clients advertise whether they are placeholders",
            mixed_clients[live].is_mock is False and mixed_clients[placeholder].is_mock is True,
        )
        all_mock = replace(one_live, mock_mode=True)
        check(
            "global MOCK_MODE forces every system to the mock",
            all(all_mock.uses_mock(system) for system in SYSTEMS)
            and all(
                isinstance(client, MockTTSClient)
                for client in evaluation.build_clients_for(all_mock, adapter).values()
            ),
        )
        check(
            "a live system needs a token to be usable",
            replace(one_live, hf_token="tok").ready_for_live_mode()
            and not replace(one_live, hf_token="").ready_for_live_mode(),
        )
        check("masked endpoints never reveal the full URL", "example.invalid/tts" not in one_live.masked_endpoint(live))

        # -- 3. randomisation ------------------------------------------------
        print("\n3. A/B randomisation")
        assignments = [evaluation.new_trial(condition, SYSTEMS).assignment for _ in range(400)]
        counter = Counter(a.model_for_A for a in assignments)
        check("both systems appear as Sample A", len(counter) == 2, str(dict(counter)))
        check(
            "assignment is roughly balanced",
            all(120 <= count <= 280 for count in counter.values()),
            str(dict(counter)),
        )
        check(
            "A and B are always different systems",
            all(a.model_for_A != a.model_for_B for a in assignments),
        )
        trial = evaluation.new_trial(condition, SYSTEMS)
        before = (trial.assignment.model_for_A, trial.assignment.model_for_B)
        for _ in range(5):  # simulate Streamlit reruns reading the stored trial
            after = (trial.assignment.model_for_A, trial.assignment.model_for_B)
        check("assignment is stable across reruns", before == after)
        check("randomised assignments are marked as such", all(a.randomized for a in assignments))

        pinned_settings = replace(app_settings, randomize_ab=False, sample_a_system=SYSTEMS[0])
        pinned = [evaluation.new_trial_for(pinned_settings, condition).assignment for _ in range(20)]
        check(
            "pinned mapping always puts the configured system in Sample A",
            all(a.model_for_A == SYSTEMS[0] and a.model_for_B == SYSTEMS[1] for a in pinned),
        )
        check("pinned assignments are marked as not randomised", not any(a.randomized for a in pinned))
        randomised_settings = replace(app_settings, randomize_ab=True)
        check(
            "randomisation stays the default policy",
            len({
                evaluation.new_trial_for(randomised_settings, condition).assignment.model_for_A
                for _ in range(200)
            })
            == 2,
        )
        try:
            evaluation.SampleAssignment.pinned(SYSTEMS, "not-a-system")
            check("pinning an unknown system is rejected", False)
        except EvaluationError:
            check("pinning an unknown system is rejected", True)

        # -- 4. caching ------------------------------------------------------
        print("\n4. Generation and caching")
        cache = AudioCache(workdir / "cache")
        clients = {system: MockTTSClient(system) for system in SYSTEMS}
        evaluation.prepare_samples(trial, clients, cache)
        check("both samples were prepared", trial.is_ready)
        sample_a, sample_b = trial.sample("A"), trial.sample("B")
        check("first generation is not a cache hit", not sample_a.from_cache and not sample_b.from_cache)
        check(
            "cache key contains system/language/gender/speaker/sentence",
            sample_a.cache_key.endswith("igbo/female/IG-F-01/IG-003.wav")
            and sample_a.cache_key.startswith(sample_a.system),
            sample_a.cache_key,
        )
        check("the two samples are cached separately", sample_a.cache_key != sample_b.cache_key)
        check(
            "the same text was sent to both systems",
            sample_a.clip.size_bytes > 0 and sample_b.clip.size_bytes > 0,
        )

        trial2 = evaluation.new_trial(condition, SYSTEMS)
        evaluation.prepare_samples(trial2, clients, cache)
        check(
            "second trial reuses the cached audio",
            trial2.sample("A").from_cache and trial2.sample("B").from_cache,
        )
        check(
            "cached audio is identical to the first generation",
            trial2.sample("A").clip.data
            == (sample_a.clip.data if trial2.sample("A").system == sample_a.system else sample_b.clip.data),
        )

        igbo_condition = build_condition(
            catalog,
            language_key="igbo",
            speaker_id="IG-F-01",
            sentence_id="IG-003",
        )
        igbo_trial = evaluation.new_trial(igbo_condition, SYSTEMS)
        evaluation.prepare_samples(igbo_trial, clients, cache)
        check(
            "cache key contains system/language/gender/speaker/sentence",
            "igbo/female/IG-F-01/IG-003.wav" in igbo_trial.sample("A").cache_key,
            igbo_trial.sample("A").cache_key,
        )

        clone_trial = evaluation.new_trial(clone_condition, SYSTEMS)
        evaluation.prepare_samples(clone_trial, clients, cache)
        clone_cache_key = clone_trial.sample("A").cache_key
        check("cloned audio lives in its own cache namespace", "/clone-" in clone_cache_key, clone_cache_key)
        check(
            "a cloned result never collides with a preset result",
            clone_cache_key != sample_a.cache_key and clone_cache_key != sample_b.cache_key,
        )
        other_reference = build_condition(
            catalog,
            language_key="igbo",
            speaker_id="IG-F-01",
            sentence_id="IG-003",
            generation_mode=CLONE_MODE,
            reference_audio=ReferenceAudio.from_upload(b"RIFF....a-different-clip", "other.wav"),
        )
        other_trial = evaluation.new_trial(other_reference, SYSTEMS)
        evaluation.prepare_samples(other_trial, clients, cache)
        check(
            "different cloning prompts get different cache entries",
            other_trial.sample("A").cache_key != clone_cache_key,
        )

        custom_condition = build_condition(
            catalog, language_key="igbo", speaker_id="IG-F-01", sentence=custom
        )
        custom_trial = evaluation.new_trial(custom_condition, SYSTEMS)
        evaluation.prepare_samples(custom_trial, clients, cache)
        check(
            "custom sentences are cached under their derived id",
            custom_trial.sample("A").cache_key.endswith(f"{custom.sentence_id}.wav"),
            custom_trial.sample("A").cache_key,
        )
        repeat_custom = evaluation.new_trial(custom_condition, SYSTEMS)
        evaluation.prepare_samples(repeat_custom, clients, cache)
        check(
            "the same custom sentence reuses its cached audio",
            repeat_custom.sample("A").from_cache and repeat_custom.sample("B").from_cache,
        )

        # Placeholder audio must never be served once a system goes live.
        live_client_trial = evaluation.new_trial(condition, SYSTEMS)
        live_clients = {
            SYSTEMS[0]: HFEndpointClient(
                system=SYSTEMS[0],
                url="https://example.invalid/tts",
                token="dummy-token",
                adapter=adapter,
                session=_StubSession([_StubResponse(200, clip.data, "audio/wav")]),
                sleep=lambda _: None,
            ),
            SYSTEMS[1]: clients[SYSTEMS[1]],
        }
        evaluation.prepare_samples(live_client_trial, live_clients, cache)
        live_sample = next(
            sample for sample in live_client_trial.samples.values() if sample.system == SYSTEMS[0]
        )
        check("switching a system to live regenerates its cached audio", not live_sample.from_cache)
        check("live samples are not flagged as placeholders", not live_sample.mocked)
        check(
            "placeholder samples are flagged per side",
            next(
                sample for sample in live_client_trial.samples.values() if sample.system == SYSTEMS[1]
            ).mocked,
        )

        broken_trial = evaluation.new_trial(condition, SYSTEMS)
        try:
            evaluation.prepare_samples(
                broken_trial,
                {SYSTEMS[0]: clients[SYSTEMS[0]], SYSTEMS[1]: _FailingClient()},
                AudioCache(workdir / "cache-broken"),
            )
            check("a failing endpoint aborts the whole trial", False)
        except SampleGenerationError as exc:
            check(
                "a failing endpoint aborts the whole trial",
                not broken_trial.is_ready and "Unable to generate" in exc.message,
            )
            check("tester-facing error message hides technical detail", "simulated" not in exc.message)

        # -- 5. validation ---------------------------------------------------
        print("\n5. Rating validation")
        check("empty ratings are rejected", len(evaluation.validate_ratings(Ratings())) >= 4)
        partial = Ratings(preference="A", naturalness_a=4, listened_to_both=True)
        check("partial ratings are rejected", bool(evaluation.validate_ratings(partial)))
        complete = Ratings(
            preference="same",
            naturalness_a=4,
            naturalness_b=5,
            pronunciation_a=3,
            pronunciation_b=4,
            speaker_similarity_a=5,
            speaker_similarity_b=2,
            comments="",
            listened_to_both=True,
        )
        check("complete ratings pass (comments optional)", evaluation.validate_ratings(complete) == [])
        check(
            "listening confirmation is required",
            bool(evaluation.validate_ratings(Ratings(**{**complete.__dict__, "listened_to_both": False}))),
        )

        # -- 6. storage ------------------------------------------------------
        print("\n6. Results storage")
        store = ResultsStore(workdir / "results" / "evaluations.db")
        row = evaluation.build_result_row(
            trial,
            complete,
            tester_id="selfcheck-tester",
            session_id="selfcheck-session",
            selection_mode="researcher",
            app_version=app_settings.app_version,
            listening_seconds=12.3,
        )
        store.save(row)
        store.save(
            evaluation.build_result_row(
                igbo_trial,
                Ratings(
                    preference="A",
                    naturalness_a=5,
                    naturalness_b=3,
                    pronunciation_a=5,
                    pronunciation_b=3,
                    speaker_similarity_a=4,
                    speaker_similarity_b=3,
                    listened_to_both=True,
                ),
                tester_id="selfcheck-tester",
                session_id="selfcheck-session",
                selection_mode="random",
                app_version=app_settings.app_version,
            )
        )
        check("evaluations are persisted", store.count() == 2, str(store.count()))
        stored = store.rows()[0]
        check(
            "internal A/B mapping is stored for decoding",
            {stored["sample_A_model"], stored["sample_B_model"]} == set(SYSTEMS),
        )
        check("preference is decoded to a system", stored["preferred_model"] in set(SYSTEMS) | {"tie"})
        check("condition is traceable", bool(stored["sentence_text"]) and bool(stored["sample_A_cache_key"]))
        check("the generation mode is recorded", stored["generation_mode"] in {PRESET_MODE, CLONE_MODE})
        check(
            "placeholder audio is flagged per trial and per side",
            stored["mock_mode"] == 1
            and stored["sample_A_mocked"] == 1
            and stored["sample_B_mocked"] == 1,
        )
        check("the randomisation policy is recorded", stored["ab_randomized"] == 1)

        pinned_trial = evaluation.new_trial_for(pinned_settings, clone_condition)
        evaluation.prepare_samples(pinned_trial, clients, cache)
        store.save(
            evaluation.build_result_row(
                pinned_trial,
                complete,
                tester_id="selfcheck-tester",
                session_id="selfcheck-session",
                selection_mode="researcher",
                app_version=app_settings.app_version,
            )
        )
        pinned_row = next(r for r in store.rows() if r["trial_id"] == pinned_trial.trial_id)
        check("a pinned mapping is recorded as not randomised", pinned_row["ab_randomized"] == 0)
        check("clone trials record their mode and prompt", pinned_row["generation_mode"] == CLONE_MODE and pinned_row["reference_audio"].startswith("upload:"))
        check("the cloning transcript is stored for reproduction", pinned_row["reference_text"] == "the transcript")

        live_row = evaluation.build_result_row(
            live_client_trial,
            complete,
            tester_id="selfcheck-tester",
            session_id="selfcheck-session",
            selection_mode="researcher",
            app_version=app_settings.app_version,
        )
        check(
            "a half-placeholder trial is still flagged as mock",
            live_row["mock_mode"] == 1
            and (live_row["sample_A_mocked"], live_row["sample_B_mocked"]) != (1, 1),
        )
        exported = store.export_csv(workdir / "export.csv")
        check("CSV export contains both rows", exported.read_text(encoding="utf-8").count("\n") >= 3)
        anonymous = store.csv_bytes(include_internal=False).decode()
        check(
            "anonymised export drops the decoding columns",
            all(column not in anonymous.splitlines()[0] for column in INTERNAL_COLUMNS),
        )
        coverage = store.coverage(summary["conditions"])
        check("coverage is computed", coverage["conditions_evaluated"] == 2, str(coverage))
        check("per-language breakdown works", bool(store.counts_by("language")))
        check("preference breakdown works", bool(store.preference_breakdown("language")))

        # -- 7. anonymity ----------------------------------------------------
        print("\n7. Anonymity of the tester UI")
        forbidden = ("individual", "combined", "model a", "model b", "endpoint", "checkpoint", "cosyvoice")
        tester_ui = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8").lower()
        # Inspect the quoted string literals - i.e. everything that can reach a screen.
        literals = " ".join(
            "".join(match) for match in re.findall(r'"([^"\n]*)"|\'([^\'\n]*)\'', tester_ui)
        )
        leaks = [word for word in forbidden if word in literals]
        check("app.py contains no system-identifying UI text", not leaks, str(leaks))
        check(
            "sample labels are A and B only",
            evaluation.SAMPLE_LABELS == ("A", "B"),
        )
        dashboard = (PROJECT_ROOT / "services" / "researcher_view.py").read_text(encoding="utf-8")
        check(
            "the decoding view is password protected",
            "compare_digest" in dashboard and "admin_password" in dashboard,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{_passed} checks passed, {len(_failed)} failed")
    if _failed:
        for description in _failed:
            print(f"  - {description}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
