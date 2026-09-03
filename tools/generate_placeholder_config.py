"""Regenerate the PLACEHOLDER speakers.yaml and test_sentences.csv.

Run this only if you want to reset the placeholder configuration back to a
clean skeleton derived from `config/languages.yaml`:

    python tools/generate_placeholder_config.py

WARNING: it overwrites `config/speakers.yaml` and `config/test_sentences.csv`.
Once you have filled in your real 84 speakers and real sentences, do not run
it again.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config"


SPEAKERS_HEADER = """# ---------------------------------------------------------------------------
# Reference speaker configuration - 84 speakers in total.
#
#   37 non-English languages x 2 speakers (male + female) = 74
#   English: 5 African English accents x 2 speakers        = 10
#                                                          ----
#                                                            84
#
# These speakers are PRESETS that already exist inside the model, so the
# default generation path sends nothing but the speaker selection: NO
# reference audio and NO reference text.
#
# Replace the placeholders:
#   speaker_id      -> your real reference-speaker identifier (REQUIRED)
#   label           -> researcher-facing description (never shown to testers)
#
# OPTIONAL, and empty by default:
#   reference_audio -> a list of local paths or http(s)/hf URLs, used ONLY as a
#                      convenience default for the "Clone a voice" mode. Leave
#                      it empty for normal preset text-to-speech. Values that
#                      start with PLACEHOLDER are ignored at load time.
#   reference_text  -> transcript of that clip. Optional even when cloning.
#
# Non-English languages use:
#     <language>:
#       male:   { speaker_id: ... }
#       female: { speaker_id: ... }
#
# English uses one list of speakers per accent:
#     english:
#       nigerian:
#         - speaker_id: EN-NG-01
#         - speaker_id: EN-NG-02
#
# A gender entry may also hold a LIST if you later add more than one speaker
# per gender - the loader accepts both a single mapping and a list.
# ---------------------------------------------------------------------------

"""


def build_speakers_yaml(languages: list[dict]) -> str:
    blocks: list[str] = []
    for lang in languages:
        key, code, label = lang["key"], lang["code"], lang["label"]
        lines = [f"{key}:"]
        if lang.get("accents"):
            for accent in lang["accents"]:
                lines.append(f"  # {accent['label']}")
                lines.append(f"  {accent['key']}:")
                for idx in (1, 2):
                    speaker_id = f"{code}-{accent['code']}-{idx:02d}"
                    gender = "male" if idx == 1 else "female"
                    lines.append(f"    - speaker_id: {speaker_id}")
                    lines.append(
                        f"      gender: {gender}        # PLACEHOLDER - set the real speaker gender"
                    )
                    lines.append(f"      label: \"{accent['label']} reference speaker {idx}\"")
                    lines.append("      reference_audio: []       # optional, cloning only")
                    lines.append("      reference_text: null      # optional, cloning only")
                lines.append("")
            blocks.append("\n".join(lines).rstrip() + "\n")
            continue

        for gender, gender_code in (("male", "M"), ("female", "F")):
            speaker_id = f"{code}-{gender_code}-01"
            lines.append(f"  {gender}:")
            lines.append(f"    speaker_id: {speaker_id}")
            lines.append(f'    label: "{label} {gender} reference speaker"')
            lines.append("    reference_audio: []            # optional, cloning only")
            lines.append("    reference_text: null           # optional, cloning only")
        blocks.append("\n".join(lines) + "\n")
    return SPEAKERS_HEADER + "\n".join(blocks)


def build_sentence_rows(languages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for lang in languages:
        key, code, label = lang["key"], lang["code"], lang["label"]
        if lang.get("accents"):
            # Accent-agnostic sentences are offered for every accent.
            for idx in (1, 2):
                rows.append(
                    {
                        "sentence_id": f"{code}-{idx:03d}",
                        "language": key,
                        "accent": "",
                        "text": f"[PLACEHOLDER] Shared English test sentence {idx} "
                        "- replace with the real evaluation sentence.",
                    }
                )
            for accent in lang["accents"]:
                for idx in (1, 2):
                    rows.append(
                        {
                            "sentence_id": f"{code}-{accent['code']}-{idx:03d}",
                            "language": key,
                            "accent": accent["key"],
                            "text": f"[PLACEHOLDER] {accent['label']} test sentence {idx} "
                            "- replace with the real evaluation sentence.",
                        }
                    )
            continue

        for idx in range(1, 11):
            rows.append(
                {
                    "sentence_id": f"{code}-{idx:03d}",
                    "language": key,
                    "accent": "",
                    "text": f"[PLACEHOLDER] {label} test sentence {idx} "
                    "- replace with tools/write_test_sentences.py.",
                }
            )
    return rows


def main() -> None:
    languages = yaml.safe_load((CONFIG / "languages.yaml").read_text(encoding="utf-8"))["languages"]

    (CONFIG / "speakers.yaml").write_text(build_speakers_yaml(languages), encoding="utf-8")

    rows = build_sentence_rows(languages)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["sentence_id", "language", "accent", "text"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    (CONFIG / "test_sentences.csv").write_text(buffer.getvalue(), encoding="utf-8")

    print(f"languages: {len(languages)}")
    print("wrote config/speakers.yaml")
    print(f"wrote config/test_sentences.csv ({len(rows)} sentences)")


if __name__ == "__main__":
    main()
