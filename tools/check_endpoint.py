"""Send one real synthesis request to each live endpoint and report back.

    python tools/check_endpoint.py                  # short Yoruba sentence
    python tools/check_endpoint.py --language igbo --gender male
    python tools/check_endpoint.py --text "Custom sentence to synthesise."

Use this after setting HF_TOKEN or after changing an endpoint URL. It ignores
placeholder systems, never prints your token, and writes any successful audio
to `data/endpoint_check/` so you can listen to it.

A cold Hugging Face endpoint can take a minute or two to answer the first
request; the client retries 503s while it starts up.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services import evaluation  # noqa: E402
from services.config_loader import TestCatalog, make_custom_sentence  # noqa: E402
from services.hf_endpoint import (  # noqa: E402
    CLONE_MODE,
    PRESET_MODE,
    EndpointError,
    ReferenceAudio,
)
from services.settings import SYSTEMS, configure_logging, load_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="yoruba")
    parser.add_argument("--gender", default="female", choices=["male", "female"])
    parser.add_argument("--accent", default=None, help="English only")
    parser.add_argument("--text", default=None, help="Defaults to the first configured sentence")
    parser.add_argument(
        "--clone",
        metavar="WAV",
        default=None,
        help="Test voice cloning with this WAV file as the reference clip",
    )
    parser.add_argument("--prompt-text", default=None, help="Optional transcript of --clone")
    args = parser.parse_args()

    settings = load_settings()
    configure_logging(settings)
    catalog = TestCatalog.load(settings.config_dir)

    print("Endpoint status")
    for system in SYSTEMS:
        state = "placeholder" if settings.uses_mock(system) else "live"
        print(f"  {system:12} {state:12} {settings.masked_endpoint(system)}")
    print(f"  HF token set: {bool(settings.hf_token)}")
    print(f"  MOCK_MODE:    {settings.mock_mode}")

    live = [s for s in SYSTEMS if not settings.uses_mock(s)]
    if not live:
        print("\nNothing to test: every system is a placeholder.")
        return 0
    if not settings.hf_token:
        print("\nHF_TOKEN is empty - a live endpoint will reject the request.")
        return 1

    speakers = catalog.speakers_for(args.language, accent=args.accent, gender=args.gender)
    if not speakers:
        print(f"\nNo speaker configured for {args.language}/{args.gender}.")
        return 1
    if args.text:
        sentence = make_custom_sentence(args.text, args.language, args.accent)
    else:
        sentence = catalog.sentences_for(args.language, accent=args.accent)[0]
    reference = None
    mode = PRESET_MODE
    if args.clone:
        clip_path = Path(args.clone)
        if not clip_path.exists():
            print(f"\nReference clip not found: {clip_path}")
            return 1
        reference = ReferenceAudio.from_upload(clip_path.read_bytes(), clip_path.name)
        mode = CLONE_MODE

    condition = evaluation.build_condition(
        catalog,
        language_key=args.language,
        speaker_id=speakers[0].speaker_id,
        sentence=sentence,
        accent_key=args.accent,
        generation_mode=mode,
        reference_audio=reference,
        reference_text=args.prompt_text,
    )

    request = condition.synthesis_request()
    print(f"\nRequesting: {condition.language.label} / {condition.speaker.speaker_id} ({mode})")
    print(f'  text: "{request.text[:70]}"')
    if reference is not None:
        print(f"  reference: {reference.name} ({len(reference.data):,} bytes)")
        print(f"  transcript: {args.prompt_text or '(none)'}")

    clients = evaluation.build_clients_for(settings)
    output_dir = PROJECT_ROOT / "data" / "endpoint_check"
    failures = 0

    for system in live:
        print(f"\n[{system}] calling endpoint (timeout {settings.request_timeout}s)...")
        started = time.time()
        try:
            clip = clients[system].synthesize(request)
        except EndpointError as exc:
            failures += 1
            print(f"[{system}] FAILED after {time.time() - started:.1f}s")
            print(f"           {exc}")
            continue
        elapsed = time.time() - started
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{system}-{mode}.{clip.extension}"
        target.write_bytes(clip.data)
        print(
            f"[{system}] OK in {elapsed:.1f}s - {clip.size_bytes:,} bytes "
            f"({clip.mime_type})\n           saved to {target.relative_to(PROJECT_ROOT)}"
        )

    if failures:
        print(f"\n{failures} of {len(live)} live endpoint(s) failed. See the log for detail.")
        return 1
    print(f"\nAll {len(live)} live endpoint(s) responded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
