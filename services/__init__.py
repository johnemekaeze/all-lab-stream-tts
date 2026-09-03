"""Service layer for the multilingual speech evaluation app.

Modules are deliberately independent of Streamlit so they can be tested and
reused from scripts:

    settings       process configuration read from the environment / .env
    config_loader  languages, speakers and test sentences (YAML / CSV)
    hf_endpoint    Hugging Face endpoint client, payload adapter, mock client
    audio_cache    on-disk cache of generated audio
    evaluation     test conditions, A/B randomisation, trials, validation
    storage        SQLite results store and CSV export
"""

__version__ = "1.0.0"
