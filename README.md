# 🎨 Text-to-Image Studio

A Streamlit web app that turns a text prompt into an image using open-source
models (FLUX.1 [schnell], Stable Diffusion XL) served through
**Hugging Face Inference Providers**.

## Architecture

```
User ─► Streamlit app (app.py, runs on CPU) ─► Hugging Face Inference Providers
                                                 (FLUX / SDXL on remote GPUs)
     ◄── image displayed + download button ◄─── PIL image returned
```

The model never runs inside the app, so it deploys on Streamlit Community Cloud
(1 CPU-class container, no GPU) without changes.

## Features

- Prompt and optional negative prompt (SDXL)
- Image size presets or custom size, inference steps, guidance scale, seed
- Progress indicator, image preview, PNG/JPEG download
- Input validation and plain-language error messages

## Project structure

```
text-to-image-app/
├── app.py                          # the Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # this file
├── .gitignore                      # keeps secrets & junk out of Git
└── .streamlit/
    ├── config.toml                 # theme (safe to commit)
    └── secrets.toml.example        # template — real secrets.toml is git-ignored
```

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your token
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (do **not** commit `secrets.toml`).
2. Go to https://share.streamlit.io → **Create app** → pick the repo, branch `main`, file `app.py`.
3. **Advanced settings → Secrets**, paste: `HF_TOKEN = "hf_..."`
4. Click **Deploy**.

## Token

Create a fine-grained token at https://huggingface.co/settings/tokens with the
permission **"Make calls to Inference Providers"**. Free accounts get a small
monthly credit allowance; see https://huggingface.co/settings/billing for usage.

## Licenses

FLUX.1 [schnell] is Apache-2.0. SDXL base 1.0 is CreativeML Open RAIL++-M.
Check each model card before commercial use.
