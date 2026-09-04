"""On-disk cache for generated audio.

Layout (the accent segment only appears for languages that use accents):

    <cache_root>/individual/yoruba/female/YO-F-01/YO-003.wav
    <cache_root>/combined/yoruba/female/YO-F-01/YO-003.wav
    <cache_root>/individual/english/nigerian/male/EN-NG-01/EN-NG-001.wav

Each audio file is accompanied by `<sentence_id>.meta.json` holding the mime
type and a fingerprint of the request. If the sentence text (or any other
request field) changes, the fingerprint no longer matches and the audio is
regenerated instead of silently serving stale speech.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .hf_endpoint import AudioClip

logger = logging.getLogger("stream_tts.cache")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str | None, fallback: str = "na") -> str:
    cleaned = _SAFE.sub("-", (value or "").strip()).strip("-")
    return cleaned or fallback


@dataclass(frozen=True)
class CacheKey:
    system: str
    language: str
    gender: str
    speaker_id: str
    sentence_id: str
    accent: str | None = None
    #: "preset" (voices baked into the model) or "clone".
    generation_mode: str = "preset"
    #: Identity of the cloning prompt; only used in clone mode.
    reference_id: str = ""

    @property
    def variant(self) -> str:
        """Extra path segment that keeps cloned audio away from preset audio."""
        if self.generation_mode == "preset":
            return ""
        digest = hashlib.sha256(self.reference_id.encode("utf-8")).hexdigest()[:8]
        return f"{_slug(self.generation_mode)}-{digest}"

    @property
    def parts(self) -> tuple[str, ...]:
        segments = [self.system, self.language]
        if self.accent:
            segments.append(self.accent)
        segments.extend([self.gender, self.speaker_id])
        if self.variant:
            segments.append(self.variant)
        return tuple(_slug(segment) for segment in segments)

    @property
    def directory(self) -> Path:
        return Path(*self.parts)

    def as_string(self, extension: str = "wav") -> str:
        return "/".join(self.parts) + f"/{_slug(self.sentence_id)}.{extension}"

    def audio_path(self, root: Path, extension: str = "wav") -> Path:
        return root / self.directory / f"{_slug(self.sentence_id)}.{extension}"

    def meta_path(self, root: Path) -> Path:
        return root / self.directory / f"{_slug(self.sentence_id)}.meta.json"


@dataclass(frozen=True)
class CachedAudio:
    clip: AudioClip
    path: Path
    from_cache: bool

    @property
    def cache_key(self) -> str:
        return self.path.name


class AudioCache:
    """Filesystem cache. Safe to share between Streamlit sessions."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- read/write --------------------------------------------------------
    def lookup(self, key: CacheKey, fingerprint: str) -> tuple[AudioClip, Path] | None:
        meta_path = key.meta_path(self.root)
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Unreadable cache metadata %s: %s", meta_path, exc)
            return None

        if meta.get("fingerprint") != fingerprint:
            logger.info("Cache fingerprint changed for %s - regenerating", key.as_string())
            return None

        audio_path = meta_path.parent / str(meta.get("filename") or "")
        if not audio_path.is_file():
            return None
        try:
            data = audio_path.read_bytes()
        except OSError as exc:
            logger.warning("Could not read cached audio %s: %s", audio_path, exc)
            return None
        if not data:
            return None
        return AudioClip(data, str(meta.get("mime_type") or "audio/wav")), audio_path

    def store(self, key: CacheKey, clip: AudioClip, fingerprint: str) -> Path:
        audio_path = key.audio_path(self.root, clip.extension)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file first so a crash cannot leave a truncated clip.
        temp_path = audio_path.with_suffix(audio_path.suffix + ".part")
        temp_path.write_bytes(clip.data)
        temp_path.replace(audio_path)

        key.meta_path(self.root).write_text(
            json.dumps(
                {
                    "filename": audio_path.name,
                    "mime_type": clip.mime_type,
                    "fingerprint": fingerprint,
                    "size_bytes": clip.size_bytes,
                    "created_at": time.time(),
                    "system": key.system,
                    "language": key.language,
                    "accent": key.accent,
                    "gender": key.gender,
                    "speaker_id": key.speaker_id,
                    "sentence_id": key.sentence_id,
                    "generation_mode": key.generation_mode,
                    "reference_id": key.reference_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return audio_path

    def get_or_generate(
        self,
        key: CacheKey,
        fingerprint: str,
        generate: Callable[[], AudioClip],
    ) -> CachedAudio:
        hit = self.lookup(key, fingerprint)
        if hit is not None:
            clip, path = hit
            logger.debug("Cache hit %s", key.as_string(clip.extension))
            return CachedAudio(clip=clip, path=path, from_cache=True)

        clip = generate()
        path = self.store(key, clip, fingerprint)
        logger.info("Cached new audio %s (%s bytes)", key.as_string(clip.extension), clip.size_bytes)
        return CachedAudio(clip=clip, path=path, from_cache=False)

    # -- introspection -----------------------------------------------------
    def stats(self) -> dict[str, int]:
        files = [p for p in self.root.rglob("*") if p.is_file() and p.suffix != ".json"]
        return {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        }

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()
