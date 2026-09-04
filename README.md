# Multilingual Speech Evaluation (anonymous A/B listening test)

A Streamlit application for running a blind human evaluation of two TTS
systems across 38 languages and 84 reference speakers.

For every test condition (language + accent + gender + speaker + sentence) the
**same** synthesis request is sent to **two** Hugging Face Inference Endpoints.
The two resulting audio clips are shown side by side as **Sample A** and
**Sample B** in a random order. The tester never learns which system produced
which sample; the mapping is written only to the results database so
researchers can decode the data later.

The application is a self-contained Streamlit app. **It does not require
Cineca**, Docker, or Google Drive: it talks to your two HTTP endpoints, caches
the audio on local disk and stores results in SQLite.

---

## 1. What the application does

- Loads all languages, speakers and test sentences from configuration files.
- Lets a researcher pick a test condition (or picks one at random), pick a
  sentence from the test set **or type a custom one**, and choose between the
  **preset voices** baked into the systems and **cloning a voice** recorded
  straight from the browser microphone.
- Sends one identical request to each of the two systems.
- Randomises which system becomes Sample A and which becomes Sample B, using
  `secrets`, once per trial. Streamlit reruns cannot reshuffle it.
- Caches every generated clip on disk so the same condition is never
  synthesised twice.
- Collects overall preference, naturalness, pronunciation/intelligibility,
  speaker similarity/voice quality and free-text comments.
- Stores every submission in SQLite, with CSV export.
- Offers a password-protected researcher dashboard (progress, coverage,
  decoded preferences, export, diagnostics, per-system live/placeholder status).
- Works **per system**: each system independently uses its real endpoint or an
  offline placeholder, so the study can start while only one model is deployed.

### Where the controls live

| Sidebar | Main area (the wide space) |
| --- | --- |
| Your name, language, English accent / gender, reference speaker | Test sentence (from the set or typed), preset-vs-clone choice with the cloning inputs, the load button, the samples and the rating form |

### Blinding rules enforced by the code

- The tester UI (`app.py`) contains no system, model, checkpoint or endpoint
  names. `selfcheck.py` fails the build if any appear in a UI string.
- Sample A is always on the left and Sample B on the right, in identically
  styled cards - no colour, size, ordering or emphasis differences. Every rule
  in `assets/styles.css` that touches a sample applies to both, and the theme
  accent in `.streamlit/config.toml` is deliberately neutral.
- The internal identifiers (`individual` / `combined`) appear only in
  `services/researcher_view.py`, which is hidden unless `ADMIN_PASSWORD` is set
  *and* the correct password is entered.
- Audio never autoplays and can be replayed as often as the tester wants.
- Nothing in the UI reveals that a system is a placeholder, or that the A/B
  mapping has been pinned - that information is in the researcher dashboard and
  in the database only.

---

## 2. Project structure

```
all-lab-stream-tts/
├── app.py                        Streamlit UI (tester-facing only)
├── selfcheck.py                  offline verification of the whole pipeline
├── requirements.txt
├── .env                          YOUR settings + token (git-ignored)
├── .env.example                  committed template of the same variables
├── .gitignore
├── README.md
│
├── assets/
│   └── styles.css                visual styling of the evaluation UI
├── .streamlit/
│   └── config.toml               theme (neutral accent colour)
│
├── config/
│   ├── languages.yaml            38 languages (+ English accent metadata)
│   ├── speakers.yaml             84 reference speaker presets (PLACEHOLDER ids)
│   ├── test_sentences.csv        sentence_id, language, accent, text
│   └── endpoint.yaml             payload/response adapter (preset + clone)
│
├── services/
│   ├── __init__.py
│   ├── settings.py               environment configuration + logging
│   ├── config_loader.py          loads and validates languages/speakers/sentences
│   ├── hf_endpoint.py            HF endpoint client, payload adapter, mock client
│   ├── audio_cache.py            on-disk audio cache
│   ├── evaluation.py             test conditions, A/B randomisation, validation
│   ├── storage.py                SQLite results store + CSV export
│   └── researcher_view.py        password-protected researcher dashboard
│
├── tools/
│   ├── generate_placeholder_config.py   regenerates the placeholder config
│   └── ui_smoketest.py                  headless Streamlit UI test
│
├── data/                         audio cache + application log (git-ignored)
└── results/                      SQLite database + CSV exports (git-ignored)
```

Separation of concerns: test configuration (`config_loader`), endpoint
communication (`hf_endpoint`), caching (`audio_cache`), evaluation logic
(`evaluation`), results storage (`storage`) and UI (`app.py`) are independent
modules. Only `app.py` and `researcher_view.py` import Streamlit.

---

## 3. Local installation

Python 3.10 or newer.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

`.env` already exists in this project. Open it and paste your Hugging Face
token on the `HF_TOKEN=` line (see section 5). If it is ever lost, recreate it
with `copy .env.example .env` on Windows (`cp .env.example .env` elsewhere).

Verify the installation without any Hugging Face calls:

```bash
python selfcheck.py          # configuration, randomisation, caching, storage
python tools/ui_smoketest.py # headless run of the Streamlit UI
```

## 4. How to run locally

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. With the shipped `.env` the first
system calls its real Inference Endpoint (as soon as you have pasted your
token) and the second serves placeholder audio until it is deployed. Set
`MOCK_MODE=true` to work entirely offline on synthetic demo audio.

---

## 5. Environment variables

Copy `.env.example` to `.env` and edit it. `.env` is git-ignored; never commit
it and never paste your token into a configuration file or the UI.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | – | Hugging Face token used as `Authorization: Bearer ...`. **Paste it on the `HF_TOKEN=` line of `.env`.** The token must have access to the organisation that owns the endpoint. |
| `INDIVIDUAL_ENDPOINT` | – | First system: an inference URL, or `mock` / empty for a placeholder |
| `COMBINED_ENDPOINT` | – | Second system: an inference URL, or `mock` / empty for a placeholder |
| `MOCK_MODE` | `true` | `true` forces **both** systems to synthetic demo audio with no network. `false` decides per system (see below) |
| `REQUEST_TIMEOUT` | `120` | Seconds allowed per synthesis request |
| `MAX_RETRIES` | `2` | Retries per endpoint (503 while an endpoint warms up, 429, 5xx) |
| `RANDOMIZE_AB` | `true` | `true` draws the A/B mapping per trial. `false` pins it - only for temporary use while a system is a placeholder |
| `SAMPLE_A_SYSTEM` | `individual` | Which system is Sample A when `RANDOMIZE_AB=false` |
| `TESTER_ID_MODE` | `required` | `required` (tester must type their name), `prompt` (pre-filled anonymous ID, editable), `auto` (generated, read-only) |
| `ENABLE_RANDOM_MODE` | `true` | Show the randomised test-selection mode |
| `DEFAULT_SELECTION_MODE` | `researcher` | `researcher` or `random` |
| `ADMIN_PASSWORD` | empty | Researcher dashboard password. **Empty hides the dashboard entirely.** |
| `RESULTS_DB` | `results/evaluations.db` | SQLite database path |
| `AUDIO_CACHE_DIR` | `data/audio_cache` | Audio cache directory |
| `LOG_FILE` | `data/app.log` | Technical error log (never shown to testers) |
| `LOG_LEVEL` | `INFO` | Python logging level |

### Where the Hugging Face token goes

Open `.env` (in the project root, next to `app.py`) and paste the token on the
`HF_TOKEN=` line - directly after the `=`, with no quotes and no spaces:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Then save the file and restart Streamlit. `.env` is git-ignored, so the token
stays on your machine. The token needs access to the Inference Endpoint of the
organisation that owns it; a fine-grained "Read" token scoped to that
organisation is enough. Nothing in the app ever prints or logs the token.

### One live system and one placeholder

Each system decides independently how it produces audio:

| `*_ENDPOINT` value | Behaviour |
| --- | --- |
| `https://...` | Calls that real Inference Endpoint |
| `mock` | Placeholder: synthetic audio, no network |
| empty | Same as `mock` |

`MOCK_MODE=true` overrides all of it and forces both systems to placeholders.

This is how the study runs while only the first model is deployed: point
`INDIVIDUAL_ENDPOINT` at the live URL and leave `COMBINED_ENDPOINT=mock` until
the second model has been pushed to Hugging Face and deployed. Testers cannot
tell which side is which, but:

- the researcher dashboard shows, per system, whether it is live or a
  placeholder;
- every result row records `sample_A_mocked`, `sample_B_mocked` and a per-trial
  `mock_mode`, so those rows can be excluded from the analysis with
  `WHERE mock_mode = 0`;
- placeholder audio is cached separately from live audio, so no placeholder
  clip is ever served after a system goes live.

While one system is a placeholder you may also pin the mapping with
`RANDOMIZE_AB=false` and `SAMPLE_A_SYSTEM=individual`, which makes Sample A
always the live system. This is **not** bias-controlled - the dashboard warns
about it and each row stores `ab_randomized=0`. Set `RANDOMIZE_AB=true` before
collecting real comparative data.

---

## 6. How to configure languages

`config/languages.yaml` holds the 38 languages:

```yaml
languages:
  - key: yoruba      # internal id, used by speakers.yaml and the CSV
    label: Yoruba    # shown in the UI
    code: YO         # used in speaker IDs and cache paths
```

Only English declares accents, which is what makes the accent selector appear:

```yaml
  - key: english
    label: English
    code: EN
    accents:
      - { key: nigerian, label: Nigerian English, code: NG }
      - { key: ghanaian, label: Ghanaian English, code: GH }
      - { key: east_african, label: East African English, code: EA }
      - { key: north_african, label: North African English, code: NA }
      - { key: south_african, label: South African English, code: SA }
```

If you add accents to another language it automatically gets an accent
selector instead of a gender selector. The loader rejects a language that has
no speakers or no sentences, so the three files always agree.

## 7. How to configure speakers

`config/speakers.yaml`. Non-English languages use gender keys:

```yaml
yoruba:
  male:
    speaker_id: YO-M-01
    label: "Yoruba male reference speaker"    # researcher-facing only
    reference_audio: []       # OPTIONAL - see section 9
    reference_text: null      # OPTIONAL
  female:
    speaker_id: YO-F-01
    ...
```

These speakers are **presets that already exist inside the systems**, so
`speaker_id` is the only thing that has to be right: the default generation
path sends no reference audio and no reference text.

English uses a list of speakers per accent:

```yaml
english:
  nigerian:
    - speaker_id: EN-NG-01
      gender: male
      label: "Nigerian English reference speaker 1"
      reference_audio: []
      reference_text: null
    - speaker_id: EN-NG-02
      gender: female
      ...
```

Notes:

- A gender key may also hold a **list**, if you later add more than one
  speaker per gender.
- `gender` is optional; it defaults to `unspecified` and is recorded as such in
  the results.
- `label` is only ever shown in the researcher sidebar. Testers see
  `speaker_id (gender)`.

## 8. How to configure test sentences

`config/test_sentences.csv` with the columns `sentence_id, language, accent,
text`:

```csv
sentence_id,language,accent,text
YO-001,yoruba,,Ẹ ku aarọ, ...
EN-001,english,,A sentence usable with every English accent.
EN-NG-001,english,nigerian,A Nigerian-English-specific sentence.
```

- `accent` is only valid for languages that declare accents.
- An empty `accent` on an English row means the sentence is offered for **all**
  five accents.
- Any number of sentences per language is supported; the main area lists all
  sentences valid for the current selection.

### Custom sentences

The tester or researcher can also type a sentence directly in the main area
instead of choosing one from the CSV. Custom text gets a stable, content-derived
ID of the form `custom-<hash>`, so:

- the same custom sentence always reuses its cached audio;
- repeated evaluations of it group together in the results;
- the full text is stored in the `sentence_text` column either way.

Custom sentences are normalised (collapsed whitespace) and limited to 600
characters.

## 9. Reference audio and voice cloning

The reference speakers are **presets inside the models**, so the default path -
"Preset voice" - is plain text-to-speech: it sends the sentence and the speaker
selection, and **no reference audio and no reference text**. Nothing in
`config/speakers.yaml` has to be populated for this to work, and placeholder
values there are discarded at load time so they can never be sent.

"Clone a voice" is the optional path, chosen in the main area. The user
supplies:

- a **reference recording** - required. The primary way is to **record it in
  the browser**: the tester presses record and reads the sentence, and the clip
  is captured at 16 kHz via `st.audio_input`. No file handling, no hosting and
  no link needed. The microphone is only accessed when they press record, and
  the browser asks their permission first. Uploading a file (`wav`, `mp3`,
  `flac`, `ogg`, `m4a`, `webm`) is offered as a fallback, mainly for
  researchers who already have clips; if both are given, the fresh recording
  wins. There is deliberately no "paste a link" field - a tester should never
  have to host a file somewhere;
- a **transcript** of that recording - optional, and simply omitted from the
  request when left blank.

Both systems always receive the identical request, including the identical
reference clip. Browser recordings and uploads are both handled as in-memory
bytes, so the same clip reaches both endpoints even though the tester recorded
it once. Such clips are sent as base64 (configurable in
`config/endpoint.yaml` via `reference_audio_upload: base64 | data_uri`). A
reference that comes from `config/speakers.yaml` as a URL or path is passed
through untouched for the endpoint to fetch itself, which is the only remaining
way a link can reach the request. Cloned results are cached under their own
path segment
(`.../clone-<hash>/<sentence>.wav`), so a cloned clip can never collide with a
preset one or with a clone made from a different recording.

`reference_audio` in `config/speakers.yaml` remains available as an optional
default cloning prompt per speaker; leave it empty for normal use.

## 10. How to configure the Hugging Face endpoints

1. Put the URLs and your token in `.env`, and set `MOCK_MODE=false`. A system
   that is not deployed yet stays as `mock`.
2. Adjust `config/endpoint.yaml` to match your inference handler. That file is
   the **only** place that knows the wire format. It holds **two** payload
   templates, one per generation mode.

`payload` - the default, preset voices (no reference audio, no reference text):

```json
{
  "text": "...",
  "language": "yoruba",
  "speaker_id": "YO-F-01",
  "accent": null,
  "gender": "female"
}
```

`payload_clone` - only used by "Clone a voice":

```json
{
  "text": "...",
  "language": "yoruba",
  "speaker_id": "YO-F-01",
  "accent": null,
  "gender": "female",
  "reference_audio": "<base64 of the upload, or the URL as given>",
  "reference_text": "optional transcript"
}
```

Available template fields: `{text}`, `{language}`, `{language_label}`,
`{language_code}`, `{accent}`, `{accent_label}`, `{gender}`, `{speaker_id}`,
`{sentence_id}`, `{generation_mode}`, and - in clone mode only -
`{reference_audio}`, `{reference_audio_list}`, `{reference_audio_name}`,
`{reference_text}`. A value of exactly `"{field}"` passes the raw Python value
through, so `null` stays `null` and lists stay lists.

A payload key built **only** from reference fields is dropped when those fields
are empty. That is what guarantees that preset generation sends no reference
audio and no reference text, and that an omitted transcript is absent rather
than `null`.

Set `wrap_in_inputs: true` if your handler expects the standard Hugging Face
envelope `{"inputs": {...}, "parameters": {...}}`, and use `extra_payload` for
static generation settings.

Responses are handled automatically (`response.mode: auto`) for:

- raw audio bytes (`Content-Type: audio/*` or `application/octet-stream`),
- a base64 string (with or without a `data:` URI prefix),
- JSON containing the audio under any of `response.json_audio_keys`
  (searched recursively), as base64 or as a byte array.

Malformed, empty and error responses are rejected, and HTTP 408/429/5xx are
retried - a 503 from a scale-to-zero endpoint that is still starting will
resolve itself.

---

## 11. How results are stored

Every submission is one row in the `evaluations` table of
`results/evaluations.db` (SQLite, WAL mode, safe for concurrent testers):

| Column | Notes |
| --- | --- |
| `evaluation_id`, `timestamp`, `tester_id`, `session_id`, `trial_id` | identity and traceability |
| `language`, `accent`, `gender`, `speaker_id`, `sentence_id`, `sentence_text` | the exact condition |
| `sample_A_model`, `sample_B_model` | **INTERNAL** - decodes the blinding |
| `preferred_sample` | `A`, `B` or `same` (what the tester chose) |
| `preferred_model` | **INTERNAL** - the decoded winner, or `tie` |
| `naturalness_A/B`, `pronunciation_A/B`, `speaker_similarity_A/B` | 1-5 |
| `comments` | free text, optional |
| `sample_A_cache_key`, `sample_B_cache_key` | reproduce the exact audio |
| `generation_mode`, `reference_audio`, `reference_text` | `preset` or `clone`, plus the cloning prompt used |
| `mock_mode`, `sample_A_mocked`, `sample_B_mocked` | per-trial placeholder flags - `mock_mode = 0` selects fully real comparisons |
| `ab_randomized` | `1` when the mapping was drawn, `0` when it was pinned |
| `selection_mode`, `app_version`, `listening_seconds` | run metadata |

The `sample_*_model` and `preferred_model` columns are never rendered in the
tester UI.

`tester_id` holds whatever the tester typed. With the default
`TESTER_ID_MODE=required` that is their **name**, so results are attributable
rather than anonymous - tell your testers, and treat the database and exports
as personal data. Set `TESTER_ID_MODE=auto` (or `prompt`) to go back to
generated anonymous IDs; the column name and every analysis stay the same.

New columns are added to an existing database automatically on startup, so
older result files keep working (the added columns are simply empty).

Ready-made analyses from this schema: overall preference rate, and preference
or mean score broken down by language, gender, English accent or speaker. The
researcher dashboard already shows these; `ResultsStore.dataframe()` gives you
a pandas DataFrame for anything else.

## 12. How to export results

- **Dashboard:** unlock the researcher section and use *Export → Download
  results as CSV*, or *Write CSV into results/*.
- **Script:**

```python
from pathlib import Path
from services.storage import ResultsStore

store = ResultsStore(Path("results/evaluations.db"))
store.export_csv(Path("results/evaluations.csv"))                    # with decoding columns
store.export_csv(Path("results/blind.csv"), include_internal=False)  # without them
```

- **SQL:** `sqlite3 results/evaluations.db ".mode csv" ".output out.csv" "SELECT * FROM evaluations;"`

## 13. Audio caching

Generated audio is stored as:

```
data/audio_cache/individual/yoruba/female/YO-F-01/YO-003.wav
data/audio_cache/combined/yoruba/female/YO-F-01/YO-003.wav
data/audio_cache/individual/english/nigerian/male/EN-NG-01/EN-NG-001.wav
data/audio_cache/individual/yoruba/female/YO-F-01/clone-3f9c1a2b/custom-1a2b3c4d5e.wav
```

The key is system + language + accent (when applicable) + gender + speaker +
sentence, so many testers evaluating the same condition trigger exactly two
generations in total. Cloned audio gets an extra `clone-<hash of the reference>`
segment, which keeps it apart from preset audio and from clones of other
recordings. A sidecar `*.meta.json` records a fingerprint of the request; if you
edit a sentence's text the affected audio is regenerated automatically instead
of serving stale speech. The fingerprint also distinguishes placeholder audio
from live audio, so nothing synthetic survives a system going live.

To force a full regeneration, delete `data/audio_cache/`.

## 14. How to deploy

The app is a normal Streamlit app and needs no GPU - all synthesis happens on
your Hugging Face endpoints.

**Streamlit Community Cloud:** push the repository (without `.env`), then set
`HF_TOKEN`, `INDIVIDUAL_ENDPOINT`, `COMBINED_ENDPOINT`, `MOCK_MODE=false` and
`ADMIN_PASSWORD` as *Secrets* in the app settings. Note that Community Cloud
storage is ephemeral: download the CSV export regularly, or point
`RESULTS_DB` at a mounted volume.

**Any VM / container platform:** install the requirements and run
`streamlit run app.py --server.port 8501 --server.address 0.0.0.0` behind a
reverse proxy with TLS. Persist `results/` (evaluation data) and `data/`
(audio cache) on a volume.

Before inviting testers, confirm that `ADMIN_PASSWORD` is set to something
strong, that `MOCK_MODE=false`, and - once both models are deployed - that
`RANDOMIZE_AB=true`.

---

## 15. Verification

`python selfcheck.py` (103 checks, no network required) covers configuration
loading, the 38/84 dataset shape, English accent handling, custom-sentence IDs,
preset payloads (proving no reference audio or text is sent), clone payloads
with and without a transcript, all response formats, endpoint
retries/timeouts/failures, per-system mock selection against a stubbed HTTP
session, randomised and pinned A/B mappings, caching (including clone/preset
separation and placeholder-versus-live invalidation), rating validation,
storage, CSV export, coverage, and that the tester UI leaks no system identity.

`python tools/ui_smoketest.py` (81 checks) drives the real Streamlit script
headlessly: loading a test, blinding, rerun stability, that the sentence and
voice-mode controls live in the main area rather than the sidebar, custom
sentences and their caching, voice cloning from a recorded clip with no
transcript,
rejected incomplete submissions, a successful submission, the English accent
flow, randomised mode, the required-name gate, a pinned A/B mapping, and the
locked/unlocked researcher dashboard.

Neither suite makes a network call: the live HTTP client is exercised through a
stubbed session.
