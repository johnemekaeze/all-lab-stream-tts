"""Password-protected researcher dashboard.

This is the ONLY Streamlit surface that is allowed to mention the internal
system identifiers. It is rendered exclusively after a successful password
check and is hidden entirely when ADMIN_PASSWORD is not set.
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from .audio_cache import AudioCache
from .config_loader import TestCatalog
from .settings import SYSTEMS, Settings
from .storage import ResultsStore

_STATE_KEY = "researcher_unlocked"


def is_available(settings: Settings) -> bool:
    return bool(settings.admin_password)


def is_unlocked() -> bool:
    return bool(st.session_state.get(_STATE_KEY))


def render_login(settings: Settings) -> None:
    """Sidebar login. Renders nothing unless an admin password is configured."""
    if not is_available(settings):
        return

    with st.sidebar.expander("Researcher access", expanded=False):
        if is_unlocked():
            st.caption("Dashboard unlocked.")
            if st.button("Lock dashboard", use_container_width=True):
                st.session_state[_STATE_KEY] = False
                st.rerun()
            return

        password = st.text_input("Password", type="password", key="researcher_password")
        if st.button("Unlock", use_container_width=True):
            import hmac

            if password and hmac.compare_digest(password, settings.admin_password):
                st.session_state[_STATE_KEY] = True
                st.session_state.pop("researcher_password", None)
                st.rerun()
            else:
                st.error("Incorrect password.")


def render_dashboard(
    *,
    settings: Settings,
    catalog: TestCatalog,
    store: ResultsStore,
    cache: AudioCache,
) -> None:
    if not (is_available(settings) and is_unlocked()):
        return

    st.divider()
    st.subheader("Researcher dashboard")

    summary = catalog.summary()
    coverage = store.coverage(summary["conditions"])
    testers = len({row["tester_id"] for row in store.rows()})

    columns = st.columns(4)
    columns[0].metric("Evaluations", coverage["evaluations"])
    columns[1].metric("Testers", testers)
    columns[2].metric(
        "Conditions covered",
        f"{coverage['conditions_evaluated']} / {coverage['conditions_total']}",
    )
    columns[3].metric("Coverage", f"{coverage['coverage_percent']}%")

    progress_tab, preference_tab, coverage_tab, export_tab, diagnostics_tab = st.tabs(
        ["Progress", "Preference", "Coverage", "Export", "Diagnostics"]
    )

    with progress_tab:
        mock_rows = sum(1 for row in store.rows() if row.get("mock_mode"))
        if mock_rows:
            st.warning(
                f"{mock_rows} of {coverage['evaluations']} evaluations involved a placeholder "
                "system. Filter on `mock_mode = 0` before analysing preferences."
            )
        for title, column in (
            ("By language", "language"),
            ("By gender", "gender"),
            ("By English accent", "accent"),
            ("By speaker", "speaker_id"),
            ("By voice mode", "generation_mode"),
            ("By tester", "tester_id"),
        ):
            st.markdown(f"**{title}**")
            rows = store.counts_by(column)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No evaluations yet.")

    with preference_tab:
        st.caption(
            "Decoded results. `preferred_model` maps each anonymous choice back to the "
            "system that produced it."
        )
        for title, group_by in (
            ("Overall", None),
            ("By language", "language"),
            ("By gender", "gender"),
            ("By English accent", "accent"),
            ("By speaker", "speaker_id"),
        ):
            st.markdown(f"**{title}**")
            rows = store.preference_breakdown(group_by)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No evaluations yet.")

    with coverage_tab:
        covered = store.covered_condition_keys()
        per_language: dict[str, dict[str, Any]] = {}
        for language, accent, speaker, sentence in catalog.iter_conditions():
            key = "/".join(
                [
                    language.key,
                    accent.key if accent else "-",
                    speaker.gender,
                    speaker.speaker_id,
                    sentence.sentence_id,
                ]
            )
            bucket = per_language.setdefault(
                language.key, {"language": language.label, "conditions": 0, "evaluated": 0}
            )
            bucket["conditions"] += 1
            bucket["evaluated"] += 1 if key in covered else 0

        table = []
        for bucket in per_language.values():
            percent = round(100.0 * bucket["evaluated"] / bucket["conditions"], 1)
            table.append({**bucket, "coverage_percent": percent})
        st.dataframe(
            sorted(table, key=lambda row: row["coverage_percent"]),
            use_container_width=True,
            hide_index=True,
        )

    with export_tab:
        st.caption("The CSV export includes the internal A/B decoding columns.")
        st.download_button(
            "Download results as CSV",
            data=store.csv_bytes(),
            file_name=f"evaluations_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Write CSV into results/", use_container_width=True):
            path = store.export_csv()
            st.success(f"Written to {path}")
        st.markdown("**Recent evaluations**")
        recent = store.rows(limit=25)
        if recent:
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.caption("No evaluations yet.")

    with diagnostics_tab:
        st.markdown("**Systems under test**")
        st.caption(
            "Which system is live and which is still a placeholder. Placeholder audio is "
            "synthetic and must not be treated as a model result."
        )
        st.dataframe(
            [
                {
                    "system": system,
                    "audio source": "LIVE endpoint" if not settings.uses_mock(system) else "placeholder (mock)",
                    "status": settings.system_status(system),
                    "endpoint": settings.masked_endpoint(system),
                }
                for system in SYSTEMS
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**A/B mapping**")
        if settings.randomize_ab:
            st.success("Randomised per trial (bias-controlled).")
        else:
            st.warning(
                f"PINNED: Sample A is always `{settings.sample_a_system}`. "
                "This is not bias-controlled - set RANDOMIZE_AB=true for the real study. "
                "Every row records this in `ab_randomized`."
            )

        st.markdown("**Runtime**")
        st.write(
            {
                "app_version": settings.app_version,
                "global_mock_mode": settings.mock_mode,
                "hf_token_present": bool(settings.hf_token),
                "live_systems": len(settings.live_systems()),
                "request_timeout_s": settings.request_timeout,
                "max_retries": settings.max_retries,
                "tester_id_mode": settings.tester_id_mode,
                "results_db": str(settings.results_db),
                "log_file": str(settings.log_file),
            }
        )
        st.markdown("**Configuration**")
        st.write({**summary, "config_dir": str(catalog.source_dir)})
        st.markdown("**Audio cache**")
        stats = cache.stats()
        st.write(
            {
                "cache_dir": str(cache.root),
                "files": stats["files"],
                "size_mb": round(stats["bytes"] / (1024 * 1024), 2),
            }
        )
        for warning in settings.warnings():
            st.warning(warning)
        with_prompts = [s.speaker_id for s in catalog.speakers if s.has_reference_audio]
        st.info(
            f"{len(with_prompts)} of {len(catalog.speakers)} speakers have an optional default "
            "cloning prompt configured. Preset generation does not need one."
        )
        st.markdown("**Log tail**")
        st.code(_log_tail(settings), language="text")


def _log_tail(settings: Settings, lines: int = 40) -> str:
    try:
        content = settings.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no log file yet)"
    return "\n".join(content[-lines:]) or "(empty)"
