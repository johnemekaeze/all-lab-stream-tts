"""Process configuration, read from environment variables / `.env`.

Nothing in this module ever prints a token or a full endpoint URL.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from . import __version__

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Internal system identifiers. These MUST NEVER be rendered in the tester UI.
SYSTEMS: tuple[str, ...] = ("individual", "combined")

TESTER_ID_MODES = ("auto", "prompt", "required")
SELECTION_MODES = ("researcher", "random")

#: Endpoint URL value that explicitly requests the offline mock client. Used
#: for a system that has not been deployed yet.
MOCK_URL = "mock"


class ConfigurationError(RuntimeError):
    """Raised when the environment configuration cannot be used."""


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Invalid integer for %s=%r, using %s", name, raw, default)
        return default


def _resolve(path_value: str, default: str) -> Path:
    path = Path(path_value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    mock_mode: bool
    hf_token: str
    endpoints: dict[str, str]
    request_timeout: int
    max_retries: int
    tester_id_mode: str
    enable_random_mode: bool
    default_selection_mode: str
    randomize_ab: bool
    sample_a_system: str
    admin_password: str
    config_dir: Path
    audio_cache_dir: Path
    results_db: Path
    results_dir: Path
    log_file: Path
    log_level: str
    app_version: str = __version__
    _warnings: tuple[str, ...] = field(default=(), repr=False)

    # -- endpoint helpers --------------------------------------------------
    def endpoint_url(self, system: str) -> str:
        try:
            return self.endpoints[system]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"Unknown system {system!r}") from exc

    def endpoint_configured(self, system: str) -> bool:
        """True when the system points at a real http(s) endpoint."""
        url = (self.endpoints.get(system) or "").strip()
        return bool(url) and url.lower() != MOCK_URL and url.lower().startswith("http")

    def uses_mock(self, system: str) -> bool:
        """Per-system client choice.

        A system is mocked when the global MOCK_MODE forces it, when its URL is
        empty, or when its URL is the literal ``mock`` - which is how a system
        that has not been deployed yet is represented.
        """
        return self.mock_mode or not self.endpoint_configured(system)

    def live_systems(self) -> tuple[str, ...]:
        return tuple(s for s in SYSTEMS if not self.uses_mock(s))

    def mocked_systems(self) -> tuple[str, ...]:
        return tuple(s for s in SYSTEMS if self.uses_mock(s))

    def system_status(self, system: str) -> str:
        """Researcher-facing description of where a system's audio comes from."""
        if self.mock_mode:
            return "placeholder (MOCK_MODE forces all systems to mock)"
        if self.endpoint_configured(system):
            return "live endpoint"
        url = (self.endpoints.get(system) or "").strip()
        if url.lower() == MOCK_URL:
            return "placeholder (endpoint set to 'mock', not deployed yet)"
        return "placeholder (no endpoint URL configured)"

    def ready_for_live_mode(self) -> bool:
        """True when at least one system is live and can be authenticated."""
        return bool(self.live_systems()) and bool(self.hf_token)

    def masked_endpoint(self, system: str) -> str:
        """Host-only, token-free rendering for the researcher dashboard."""
        url = (self.endpoints.get(system) or "").strip()
        if not url:
            return "(not configured)"
        if not self.endpoint_configured(system):
            return "(placeholder)"
        parts = urlsplit(url)
        host = parts.netloc or url
        if len(host) > 12:
            host = f"{host[:6]}...{host[-6:]}"
        return f"{parts.scheme or 'https'}://{host}"

    def warnings(self) -> tuple[str, ...]:
        return self._warnings


def _bootstrap_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into os.environ when not already set."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return
        for key in secrets:
            value = secrets[key]
            if isinstance(value, str) and not _env(key):
                os.environ[key] = value
    except Exception:
        # Local runs and offline checks use `.env` instead.
        pass


def load_settings(*, env_file: Path | None = None, override: bool = False) -> Settings:
    """Load settings from `.env` (if present) plus the real environment."""
    dotenv_path = env_file or (PROJECT_ROOT / ".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=override)

    _bootstrap_streamlit_secrets()

    warnings: list[str] = []

    mock_mode = _env_bool("MOCK_MODE", True)
    endpoints = {
        "individual": _env("INDIVIDUAL_ENDPOINT"),
        "combined": _env("COMBINED_ENDPOINT"),
    }
    hf_token = _env("HF_TOKEN")

    tester_id_mode = _env("TESTER_ID_MODE", "required").lower()
    if tester_id_mode not in TESTER_ID_MODES:
        warnings.append(f"TESTER_ID_MODE={tester_id_mode!r} is invalid; falling back to 'required'.")
        tester_id_mode = "required"

    sample_a_system = _env("SAMPLE_A_SYSTEM", SYSTEMS[0]).lower()
    if sample_a_system not in SYSTEMS:
        warnings.append(
            f"SAMPLE_A_SYSTEM={sample_a_system!r} is not a known system; falling back to the first one."
        )
        sample_a_system = SYSTEMS[0]
    randomize_ab = _env_bool("RANDOMIZE_AB", True)
    if not randomize_ab:
        warnings.append(
            "RANDOMIZE_AB is false: the A/B mapping is PINNED, so this run is not bias-controlled. "
            "Set it back to true for the real study."
        )

    default_selection_mode = _env("DEFAULT_SELECTION_MODE", "researcher").lower()
    if default_selection_mode not in SELECTION_MODES:
        warnings.append(
            f"DEFAULT_SELECTION_MODE={default_selection_mode!r} is invalid; falling back to 'researcher'."
        )
        default_selection_mode = "researcher"

    results_db = _resolve(_env("RESULTS_DB"), "results/evaluations.db")

    settings = Settings(
        mock_mode=mock_mode,
        hf_token=hf_token,
        endpoints=endpoints,
        request_timeout=_env_int("REQUEST_TIMEOUT", 120),
        max_retries=max(0, _env_int("MAX_RETRIES", 2)),
        tester_id_mode=tester_id_mode,
        enable_random_mode=_env_bool("ENABLE_RANDOM_MODE", True),
        default_selection_mode=default_selection_mode,
        randomize_ab=randomize_ab,
        sample_a_system=sample_a_system,
        admin_password=_env("ADMIN_PASSWORD"),
        config_dir=_resolve(_env("CONFIG_DIR"), "config"),
        audio_cache_dir=_resolve(_env("AUDIO_CACHE_DIR"), "data/audio_cache"),
        results_db=results_db,
        results_dir=results_db.parent,
        log_file=_resolve(_env("LOG_FILE"), "data/app.log"),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
        _warnings=tuple(warnings),
    )

    # Endpoint warnings need the finished object so they can use `uses_mock`.
    if not mock_mode:
        placeholders = settings.mocked_systems()
        if placeholders:
            warnings.append(
                f"{len(placeholders)} of {len(SYSTEMS)} systems have no live endpoint and will "
                "serve placeholder audio. Results from those trials are flagged in the database."
            )
        if settings.live_systems() and not hf_token:
            warnings.append(
                "A live endpoint is configured but HF_TOKEN is empty - synthesis will fail. "
                "Paste your token into the HF_TOKEN line of the .env file."
            )
    return replace(settings, _warnings=tuple(warnings))


_LOGGING_CONFIGURED = False


def configure_logging(settings: Settings) -> logging.Logger:
    """Send technical detail to a log file; testers only ever see friendly text."""
    global _LOGGING_CONFIGURED
    logger = logging.getLogger("cosyvoice_eval")
    if _LOGGING_CONFIGURED:
        return logger

    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    logger.propagate = False
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    _LOGGING_CONFIGURED = True
    return logger
