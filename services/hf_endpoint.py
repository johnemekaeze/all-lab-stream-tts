"""Hugging Face Inference Endpoint client + payload/response adapter + mock.

The UI never imports `requests` and never knows the endpoint wire format.
Everything endpoint-specific lives here and in `config/endpoint.yaml`.
"""

from __future__ import annotations

import array
import base64
import binascii
import hashlib
import json
import logging
import math
import random
import re
import time
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Protocol

import requests
import yaml

logger = logging.getLogger("cosyvoice_eval.endpoint")

_TEMPLATE_FIELD = re.compile(r"^\{([a-z_]+)\}$")

_MIME_EXTENSIONS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/webm": "webm",
}

_AUDIO_MAGIC = (b"RIFF", b"ID3", b"OggS", b"fLaC", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class EndpointError(RuntimeError):
    """Base class for all synthesis failures (safe to log, not to display)."""


class EndpointTimeoutError(EndpointError):
    pass


class EndpointResponseError(EndpointError):
    pass


class EndpointNotConfiguredError(EndpointError):
    pass


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
#: Generation modes. "preset" uses the reference speakers already baked into
#: the model; "clone" sends a reference clip supplied by the user.
PRESET_MODE = "preset"
CLONE_MODE = "clone"
GENERATION_MODES = (PRESET_MODE, CLONE_MODE)

#: Template fields that only exist for voice cloning. A payload key built
#: solely from these is dropped when they are absent, which is what guarantees
#: that preset generation sends no reference audio and no reference text.
REFERENCE_FIELDS = frozenset(
    {"reference_audio", "reference_audio_list", "reference_audio_name", "reference_text"}
)


@dataclass(frozen=True)
class ReferenceAudio:
    """A voice-cloning prompt, either an uploaded clip or a URL/path.

    Uploads carry their bytes so the identical clip can be sent to both
    systems; URLs are passed through untouched for the endpoint to resolve.
    """

    location: str = ""
    data: bytes = b""
    filename: str = ""
    mime_type: str = "audio/wav"

    @classmethod
    def from_url(cls, location: str) -> "ReferenceAudio":
        return cls(location=location.strip())

    @classmethod
    def from_upload(cls, data: bytes, filename: str, mime_type: str = "audio/wav") -> "ReferenceAudio":
        return cls(data=bytes(data), filename=filename, mime_type=mime_type or "audio/wav")

    @property
    def is_upload(self) -> bool:
        return bool(self.data)

    @property
    def identifier(self) -> str:
        """Stable, log-safe identity used for cache keys and results rows."""
        if self.is_upload:
            digest = hashlib.sha256(self.data).hexdigest()[:12]
            return f"upload:{self.filename or 'reference'}#{digest}"
        return self.location

    @property
    def name(self) -> str:
        return self.filename or self.location.rsplit("/", 1)[-1]

    def as_base64(self, *, data_uri: bool = False) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}" if data_uri else encoded


@dataclass(frozen=True)
class SynthesisRequest:
    """Everything a TTS system needs to render one test condition.

    The *identical* request is sent to both systems; only the target endpoint
    differs.
    """

    text: str
    language: str
    language_label: str
    language_code: str
    speaker_id: str
    gender: str
    sentence_id: str
    accent: str | None = None
    accent_label: str | None = None
    generation_mode: str = PRESET_MODE
    reference_audio: ReferenceAudio | None = None
    reference_text: str | None = None

    @property
    def is_clone(self) -> bool:
        return self.generation_mode == CLONE_MODE

    def template_fields(self) -> dict[str, Any]:
        """Values available to the payload templates in endpoint.yaml.

        Reference fields are always None outside clone mode.
        """
        reference = self.reference_audio if self.is_clone else None
        return {
            "text": self.text,
            "language": self.language,
            "language_label": self.language_label,
            "language_code": self.language_code,
            "speaker_id": self.speaker_id,
            "gender": self.gender,
            "sentence_id": self.sentence_id,
            "accent": self.accent,
            "accent_label": self.accent_label,
            "generation_mode": self.generation_mode,
            "reference_audio": reference,
            "reference_audio_list": [reference] if reference else [],
            "reference_audio_name": reference.name if reference else None,
            "reference_text": (self.reference_text or None) if self.is_clone else None,
        }

    def fingerprint(self) -> str:
        """Stable hash of the request content (used for cache invalidation)."""
        reference = self.reference_audio if self.is_clone else None
        payload = json.dumps(
            {
                "text": self.text,
                "language": self.language,
                "speaker_id": self.speaker_id,
                "gender": self.gender,
                "accent": self.accent,
                "sentence_id": self.sentence_id,
                "generation_mode": self.generation_mode,
                "reference_audio": reference.identifier if reference else None,
                "reference_text": (self.reference_text or None) if self.is_clone else None,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def reference_identifier(self) -> str:
        reference = self.reference_audio if self.is_clone else None
        return reference.identifier if reference else ""


@dataclass(frozen=True)
class AudioClip:
    data: bytes
    mime_type: str = "audio/wav"

    @property
    def extension(self) -> str:
        return _MIME_EXTENSIONS.get(self.mime_type.lower().split(";")[0].strip(), "wav")

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# Adapter: payload building + response parsing
# ---------------------------------------------------------------------------
DEFAULT_PRESET_PAYLOAD: dict[str, Any] = {
    "inputs": "{text}",
    "language": "{language}",
    "voice": "{gender}",
}

DEFAULT_CLONE_PAYLOAD: dict[str, Any] = {
    "inputs": "{text}",
    "language": "{language}",
    "prompt_audio_base64": "{reference_audio}",
    "prompt_text": "{reference_text}",
}


@dataclass(frozen=True)
class EndpointAdapter:
    payload_template: dict[str, Any] = field(default_factory=dict)
    clone_payload_template: dict[str, Any] = field(default_factory=dict)
    reference_audio_upload: str = "base64"
    extra_payload: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    content_type: str = "application/json"
    method: str = "POST"
    wrap_in_inputs: bool = False
    omit_none: bool = False
    response_mode: str = "auto"
    json_audio_keys: tuple[str, ...] = ("audio", "audio_base64", "data")
    json_format_keys: tuple[str, ...] = ("mime_type", "content_type", "format")
    default_mime: str = "audio/wav"
    min_bytes: int = 512

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "EndpointAdapter":
        if not path.exists():
            logger.warning("Endpoint adapter config %s not found; using defaults", path)
            return cls()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise EndpointError(f"{path.name} is not valid YAML: {exc}") from exc

        request_cfg = raw.get("request") or {}
        response_cfg = raw.get("response") or {}
        return cls(
            payload_template=dict(request_cfg.get("payload") or {}),
            clone_payload_template=dict(request_cfg.get("payload_clone") or {}),
            reference_audio_upload=str(
                request_cfg.get("reference_audio_upload") or "base64"
            ).lower(),
            extra_payload=dict(request_cfg.get("extra_payload") or {}),
            parameters=dict(request_cfg.get("parameters") or {}),
            content_type=str(request_cfg.get("content_type") or "application/json"),
            method=str(request_cfg.get("method") or "POST").upper(),
            wrap_in_inputs=bool(request_cfg.get("wrap_in_inputs", False)),
            omit_none=bool(request_cfg.get("omit_none", False)),
            response_mode=str(response_cfg.get("mode") or "auto").lower(),
            json_audio_keys=tuple(response_cfg.get("json_audio_keys") or ("audio", "data")),
            json_format_keys=tuple(response_cfg.get("json_format_keys") or ("format",)),
            default_mime=str(response_cfg.get("default_mime") or "audio/wav"),
            min_bytes=int(response_cfg.get("min_bytes") or 0),
        )

    # -- request -----------------------------------------------------------
    def template_for(self, request: SynthesisRequest) -> dict[str, Any]:
        if request.is_clone:
            return self.clone_payload_template or self.payload_template or DEFAULT_CLONE_PAYLOAD
        return self.payload_template or DEFAULT_PRESET_PAYLOAD

    def build_payload(self, request: SynthesisRequest) -> dict[str, Any]:
        fields = self._encode_reference(request.template_fields())
        template = self.template_for(request)

        payload: dict[str, Any] = {}
        for key, value in template.items():
            rendered = self._render(value, fields)
            # Preset generation relies on this: any key built purely from
            # reference fields disappears when there is nothing to clone, so no
            # reference audio and no reference text is ever sent. It also drops
            # an omitted transcript in clone mode, which is optional.
            if _only_reference_fields(value) and _is_absent(rendered):
                continue
            if rendered is None and self.omit_none:
                continue
            payload[key] = rendered
        payload.update(self.extra_payload)

        if self.wrap_in_inputs:
            body: dict[str, Any] = {"inputs": payload}
            if self.parameters:
                body["parameters"] = dict(self.parameters)
            return body
        return payload

    def _encode_reference(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Turn a ReferenceAudio into something JSON-serialisable.

        Uploaded clips become base64 (or a data URI); URLs pass through so the
        endpoint can fetch them itself.
        """

        def encode(reference: ReferenceAudio | None) -> str | None:
            if reference is None:
                return None
            if not reference.is_upload:
                return reference.location or None
            if self.reference_audio_upload == "data_uri":
                return reference.as_base64(data_uri=True)
            return reference.as_base64()

        encoded = dict(fields)
        encoded["reference_audio"] = encode(fields.get("reference_audio"))
        encoded["reference_audio_list"] = [
            value for value in (encode(item) for item in fields.get("reference_audio_list") or []) if value
        ]
        return encoded

    def _render(self, value: Any, fields: dict[str, Any]) -> Any:
        if isinstance(value, str):
            match = _TEMPLATE_FIELD.match(value.strip())
            if match:
                # Pass the raw python value through (keeps None / lists intact).
                if match.group(1) not in fields:
                    raise EndpointError(
                        f"endpoint.yaml references unknown field {{{match.group(1)}}}"
                    )
                return fields[match.group(1)]
            try:
                return value.format(**fields)
            except KeyError as exc:
                raise EndpointError(f"endpoint.yaml references unknown field {exc}") from exc
        if isinstance(value, dict):
            return {k: self._render(v, fields) for k, v in value.items()}
        if isinstance(value, list):
            return [self._render(v, fields) for v in value]
        return value

    # -- response ----------------------------------------------------------
    def parse_response(self, *, content: bytes, content_type: str) -> AudioClip:
        mode = self.response_mode
        content_type = (content_type or "").lower()

        if mode == "auto":
            if content_type.startswith("audio/") or "octet-stream" in content_type:
                mode = "audio"
            elif "json" in content_type:
                mode = "json"
            elif content_type.startswith("text/"):
                mode = "base64"
            else:
                mode = "audio" if _looks_like_audio(content) else "json"

        if mode == "audio":
            clip = AudioClip(content, self._mime_from_header(content_type))
        elif mode == "base64":
            clip = AudioClip(_decode_base64(content.decode("utf-8", "ignore")), self.default_mime)
        elif mode == "json":
            clip = self._parse_json(content)
        else:
            raise EndpointResponseError(f"Unsupported response mode {self.response_mode!r}")

        self._validate(clip)
        return clip

    def _mime_from_header(self, content_type: str) -> str:
        base = content_type.split(";")[0].strip()
        return base if base.startswith("audio/") else self.default_mime

    def _parse_json(self, content: bytes) -> AudioClip:
        try:
            document = json.loads(content.decode("utf-8", "ignore"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EndpointResponseError(f"Response is not valid JSON: {exc}") from exc

        if isinstance(document, dict) and ("error" in document or "message" in document):
            if not any(_find_key(document, key) for key in self.json_audio_keys):
                detail = document.get("error") or document.get("message")
                raise EndpointResponseError(f"Endpoint returned an error: {detail}")

        raw_audio = None
        for key in self.json_audio_keys:
            raw_audio = _find_key(document, key)
            if raw_audio is not None:
                break
        if raw_audio is None:
            keys = list(document)[:8] if isinstance(document, dict) else type(document).__name__
            raise EndpointResponseError(f"No audio field in JSON response (top level: {keys})")

        mime = self.default_mime
        for key in self.json_format_keys:
            declared = _find_key(document, key)
            if isinstance(declared, str) and declared:
                mime = declared if "/" in declared else f"audio/{declared.lstrip('.')}"
                break

        if isinstance(raw_audio, str):
            return AudioClip(_decode_base64(raw_audio), mime)
        if isinstance(raw_audio, (bytes, bytearray)):
            return AudioClip(bytes(raw_audio), mime)
        if isinstance(raw_audio, list) and raw_audio and all(isinstance(v, int) for v in raw_audio):
            return AudioClip(bytes(bytearray(raw_audio)), mime)
        raise EndpointResponseError(
            f"Unsupported audio field type in JSON response: {type(raw_audio).__name__}"
        )

    def _validate(self, clip: AudioClip) -> None:
        if not clip.data:
            raise EndpointResponseError("Endpoint returned an empty audio payload")
        if self.min_bytes and clip.size_bytes < self.min_bytes:
            raise EndpointResponseError(
                f"Audio payload is suspiciously small ({clip.size_bytes} bytes)"
            )
        if clip.extension == "wav" and not clip.data.startswith(b"RIFF"):
            raise EndpointResponseError("Audio payload is not a valid WAV stream")


def _template_fields_in(value: Any) -> set[str]:
    """Field names referenced by a payload template value, recursively."""
    if isinstance(value, str):
        return set(re.findall(r"\{([a-z_]+)\}", value))
    if isinstance(value, dict):
        return set().union(*(_template_fields_in(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_template_fields_in(item) for item in value)) if value else set()
    return set()


def _only_reference_fields(value: Any) -> bool:
    referenced = _template_fields_in(value)
    return bool(referenced) and referenced <= REFERENCE_FIELDS


def _is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, (list, dict, tuple)) and len(value) == 0


def _looks_like_audio(content: bytes) -> bool:
    return content.startswith(_AUDIO_MAGIC)


def _decode_base64(value: str) -> bytes:
    payload = value.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    payload = re.sub(r"\s+", "", payload)
    padding = (-len(payload)) % 4
    try:
        return base64.b64decode(payload + "=" * padding, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise EndpointResponseError(f"Could not base64-decode the audio payload: {exc}") from exc


def _find_key(document: Any, key: str) -> Any:
    """Depth-first search for `key` in nested dicts/lists."""
    if isinstance(document, dict):
        if key in document and document[key] is not None:
            return document[key]
        for value in document.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(document, list):
        for item in document:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
class TTSClient(Protocol):
    system: str
    is_mock: bool

    def synthesize(self, request: SynthesisRequest) -> AudioClip: ...


class HFEndpointClient:
    """Calls one Hugging Face Inference Endpoint."""

    is_mock = False

    def __init__(
        self,
        *,
        system: str,
        url: str,
        token: str,
        adapter: EndpointAdapter,
        timeout: int = 120,
        max_retries: int = 2,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.system = system
        self._url = url
        self._token = token
        self._adapter = adapter
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._session = session or requests.Session()
        self._sleep = sleep

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        if not self._url:
            raise EndpointNotConfiguredError(f"No endpoint URL configured for system {self.system!r}")

        payload = self._adapter.build_payload(request)
        headers = {
            "Accept": "audio/wav, application/json",
            "Content-Type": self._adapter.content_type,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.request(
                    self._adapter.method,
                    self._url,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
            except requests.Timeout:
                last_error = EndpointTimeoutError(
                    f"[{self.system}] request timed out after {self._timeout}s"
                )
                logger.warning("%s (attempt %s)", last_error, attempt + 1)
            except requests.RequestException as exc:
                last_error = EndpointError(f"[{self.system}] transport error: {exc}")
                logger.warning("%s (attempt %s)", last_error, attempt + 1)
            else:
                if response.status_code == 200:
                    return self._adapter.parse_response(
                        content=response.content,
                        content_type=response.headers.get("Content-Type", ""),
                    )
                detail, retry_hint = _error_detail(response.text)
                last_error = EndpointError(
                    f"[{self.system}] HTTP {response.status_code}: {detail}"
                )
                # 503 usually means a scale-to-zero endpoint is still starting.
                retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                if retry_hint is False:
                    retryable = False
                logger.warning("%s (attempt %s)", last_error, attempt + 1)
                if not retryable:
                    break

            if attempt < self._max_retries:
                self._sleep(min(2 ** attempt * 2, 15))

        raise last_error or EndpointError(f"[{self.system}] synthesis failed")


class MockTTSClient:
    """Offline stand-in that synthesises a deterministic placeholder tone.

    Used for any system without a live endpoint - either because MOCK_MODE is
    on, or because that system has not been deployed yet. The whole evaluation
    workflow (randomisation, caching, storage, UI) can therefore be exercised
    without any Hugging Face calls. The two systems produce audibly different
    audio so A/B behaviour is visible, but nothing about the audio identifies
    a system.
    """

    is_mock = True

    def __init__(self, system: str, *, duration_seconds: float = 2.6, sample_rate: int = 16000) -> None:
        self.system = system
        self._duration = duration_seconds
        self._sample_rate = sample_rate

    def synthesize(self, request: SynthesisRequest) -> AudioClip:
        seed_source = f"{self.system}|{request.fingerprint()}"
        seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:12], 16)
        rng = random.Random(seed)

        base_f0 = 95.0 + rng.random() * 60.0
        if request.gender == "female":
            base_f0 *= 1.6
        syllable_rate = 2.6 + rng.random() * 1.6
        brightness = 0.15 + rng.random() * 0.25
        # Keep the clip length text-dependent so it feels like speech.
        duration = max(1.2, min(6.0, self._duration + len(request.text) / 90.0))

        total = int(duration * self._sample_rate)
        samples = array.array("h")
        for index in range(total):
            t = index / self._sample_rate
            envelope = 0.5 * (1.0 - math.cos(2 * math.pi * syllable_rate * t))
            fade = min(1.0, t / 0.05, max(0.0, (duration - t) / 0.2))
            f0 = base_f0 * (1.0 + 0.012 * math.sin(2 * math.pi * 5.0 * t))
            value = (
                0.60 * math.sin(2 * math.pi * f0 * t)
                + 0.25 * math.sin(4 * math.pi * f0 * t)
                + brightness * math.sin(6 * math.pi * f0 * t)
            )
            value *= envelope * fade * 0.32
            samples.append(int(max(-1.0, min(1.0, value)) * 32767))

        buffer = BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self._sample_rate)
            handle.writeframes(samples.tobytes())
        return AudioClip(buffer.getvalue(), "audio/wav")


def _short(text: str, limit: int = 300) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _error_detail(body: str) -> tuple[str, bool | None]:
    """Pull a human-readable message (and any retry hint) out of an error body.

    The handler answers with {"detail": {"error": ..., "retry": false}} and
    FastAPI validation errors with {"detail": [{"msg": ..., "loc": [...]}]}.
    """
    try:
        document = json.loads(body or "")
    except (json.JSONDecodeError, TypeError):
        return _short(body), None

    detail = document.get("detail", document) if isinstance(document, dict) else document
    if isinstance(detail, dict):
        message = detail.get("error") or detail.get("message") or json.dumps(detail)
        hint = detail.get("retry")
        return _short(str(message)), hint if isinstance(hint, bool) else None
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if not isinstance(item, dict):
                continue
            location = ".".join(str(piece) for piece in item.get("loc", []) if piece != "body")
            parts.append(f"{location}: {item.get('msg')}" if location else str(item.get("msg")))
        if parts:
            return _short("; ".join(parts)), None
    return _short(str(detail)), None


def build_clients(settings, adapter: EndpointAdapter, systems: tuple[str, ...]) -> dict[str, TTSClient]:
    """Create one client per internal system, deciding mock/live per system.

    A system without a live endpoint gets the mock client, so the study can run
    with one deployed system and one placeholder.
    """
    session: requests.Session | None = None
    clients: dict[str, TTSClient] = {}
    for system in systems:
        if settings.uses_mock(system):
            logger.info("System %s uses the offline mock client (%s)", system, settings.system_status(system))
            clients[system] = MockTTSClient(system)
            continue
        if session is None:
            session = requests.Session()
        logger.info("System %s uses a live Hugging Face endpoint", system)
        clients[system] = HFEndpointClient(
            system=system,
            url=settings.endpoint_url(system),
            token=settings.hf_token,
            adapter=adapter,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            session=session,
        )
    return clients
