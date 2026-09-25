"""
app.py — Text-to-Image Studio
================================================================
A Streamlit web app that turns a text prompt into an image.

How it works (the architecture):

    User  ->  Streamlit app (this file)  ->  Hugging Face Inference Providers API
          <-  image shown + download     <-  image generated on a remote GPU

The heavy AI model does NOT run inside this app. It runs on GPU servers
reached through Hugging Face's API. That is why the exact same file works
in Google Colab AND on Streamlit Community Cloud (which has no GPU).

The Hugging Face token is read from:
  1. Streamlit Secrets (st.secrets["HF_TOKEN"])  -> used on Streamlit Cloud
  2. An environment variable named HF_TOKEN      -> used in Google Colab
It is never written into this file.
"""

# ----------------------------------------------------------------
# 1. IMPORTS
# ----------------------------------------------------------------
import io
import os
import random
import time
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# ----------------------------------------------------------------
# 2. PAGE SETUP (must be the first Streamlit command in the script)
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Text-to-Image Studio",
    page_icon="🎨",
    layout="wide",
)

# ----------------------------------------------------------------
# 3. CONFIGURATION
# Everything you might want to tweak later lives here, in one place.
# ----------------------------------------------------------------

# Each model has different abilities, so we describe them in a dictionary.
# "steps" and "guidance" are (minimum, maximum, default) values for sliders.
MODELS = {
    "FLUX.1 [schnell] (fast, recommended)": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "about": "Apache-2.0 licensed, high quality in 1–4 steps. "
                 "Ignores negative prompt and guidance scale.",
        "supports_negative_prompt": False,
        "steps": (1, 4, 4),
        "guidance": None,  # None = this model does not use guidance scale
    },
    "Stable Diffusion XL (supports negative prompt)": {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "about": "Classic open model with full control: negative prompt, "
                 "guidance scale and more steps. Slower than FLUX schnell. "
                 "Provider availability can change over time.",
        "supports_negative_prompt": True,
        "steps": (10, 50, 30),
        "guidance": (1.0, 15.0, 7.0),
    },
}

# Common image sizes. Both numbers must be multiples of 64.
SIZE_PRESETS = {
    "Square — 1024 × 1024": (1024, 1024),
    "Square — 768 × 768 (cheaper, faster)": (768, 768),
    "Landscape — 1216 × 832": (1216, 832),
    "Portrait — 832 × 1216": (832, 1216),
    "Custom": None,
}

MAX_PROMPT_CHARS = 1000
MAX_SEED = 2_147_483_647          # largest seed all providers accept
REQUEST_TIMEOUT_SECONDS = 120     # give up if the API takes longer than this


# ----------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ----------------------------------------------------------------
def get_hf_token():
    """Find the Hugging Face token without ever hard-coding it.

    Order of lookup:
      1. Streamlit Secrets (Streamlit Cloud, or a local .streamlit/secrets.toml)
      2. The HF_TOKEN environment variable (how we pass it in Google Colab)
    """
    token = None
    try:
        # st.secrets raises an error if no secrets file exists at all,
        # so we wrap it in try/except and simply fall back to the env var.
        token = st.secrets.get("HF_TOKEN")
    except Exception:
        token = None
    token = token or os.environ.get("HF_TOKEN")
    return token.strip() if token else None


@st.cache_resource(show_spinner=False)
def get_client(token):
    """Create the API client once and reuse it.

    @st.cache_resource means Streamlit keeps this object between reruns,
    instead of rebuilding it every time the user clicks something.
    provider="auto" lets Hugging Face pick an available GPU provider
    (fal-ai, Replicate, Together, ...) for the chosen model.
    """
    return InferenceClient(
        provider="auto",
        api_key=token,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def validate_inputs(prompt, width, height):
    """Return a list of human-readable problems. Empty list = all good."""
    problems = []
    if not prompt.strip():
        problems.append("Enter a description of the image you want to create.")
    elif len(prompt.strip()) < 3:
        problems.append("The prompt is too short — describe the image in a few words.")
    if len(prompt) > MAX_PROMPT_CHARS:
        problems.append(
            f"The prompt is {len(prompt)} characters long. "
            f"Shorten it to {MAX_PROMPT_CHARS} characters or fewer."
        )
    if width % 64 or height % 64:
        problems.append("Width and height must both be multiples of 64.")
    return problems


def generate_image(client, model_cfg, prompt, negative_prompt,
                   width, height, steps, guidance, seed):
    """Send the request to the hosted model and return a PIL image.

    We only include optional settings the chosen model actually supports,
    because some providers reject parameters they do not understand.
    """
    request = {
        "prompt": prompt.strip(),
        "model": model_cfg["id"],
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "seed": seed,
    }
    if model_cfg["supports_negative_prompt"] and negative_prompt.strip():
        request["negative_prompt"] = negative_prompt.strip()
    if model_cfg["guidance"] is not None and guidance is not None:
        request["guidance_scale"] = guidance

    return client.text_to_image(**request)


def explain_error(error):
    """Translate technical API errors into clear advice for the user."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    text = str(error).lower()

    if status == 401 or "unauthorized" in text or "invalid credentials" in text:
        return ("Your Hugging Face token was rejected (401). Check that HF_TOKEN is "
                "copied correctly and has the 'Make calls to Inference Providers' permission.")
    if status == 402 or "payment required" in text or ("exceeded" in text and "credit" in text):
        return ("You have used up your monthly free Inference Providers credits (402). "
                "Wait for the monthly reset, add credits, or upgrade to Hugging Face PRO.")
    if status == 403 or "gated" in text or "forbidden" in text:
        return ("Access denied (403). The model may be gated — open its page on "
                "huggingface.co and accept its license — or your token lacks permission.")
    if status == 404 or "not supported by any provider" in text or "no provider" in text:
        return ("This model is not currently available through any Inference Provider. "
                "Choose a different model in the sidebar.")
    if status == 429 or "rate limit" in text:
        return "Too many requests right now (429). Wait a minute, then try again."
    if status in (500, 502, 503, 504) or "unavailable" in text or "loading" in text:
        return ("The model server is busy or warming up. Wait 20–60 seconds and try again.")
    if "timed out" in text or "timeout" in text:
        return (f"The request took longer than {REQUEST_TIMEOUT_SECONDS} seconds. "
                "Try a smaller image size or fewer steps.")
    if status == 400 or "invalid" in text:
        return ("The provider rejected one of the settings (400). Try the default size "
                "and step count for this model.")
    return "Image generation failed for an unexpected reason. See the technical details below."


def image_to_bytes(image, file_format):
    """Convert a PIL image into bytes so it can be downloaded."""
    buffer = io.BytesIO()
    if file_format == "JPEG":
        # JPEG has no transparency channel, so convert to plain RGB first.
        image.convert("RGB").save(buffer, format="JPEG", quality=95)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


# ----------------------------------------------------------------
# 5. SIDEBAR — generation settings
# ----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox("Model", list(MODELS.keys()))
    model_cfg = MODELS[model_name]
    st.caption(model_cfg["about"])

    size_name = st.selectbox("Image size", list(SIZE_PRESETS.keys()))
    if SIZE_PRESETS[size_name] is None:
        width = st.slider("Width (px)", 256, 1536, 1024, step=64)
        height = st.slider("Height (px)", 256, 1536, 1024, step=64)
    else:
        width, height = SIZE_PRESETS[size_name]

    min_steps, max_steps, default_steps = model_cfg["steps"]
    steps = st.slider(
        "Inference steps", min_steps, max_steps, default_steps,
        help="More steps = more refinement but slower. FLUX schnell needs only 1–4.",
    )

    if model_cfg["guidance"] is not None:
        g_min, g_max, g_default = model_cfg["guidance"]
        guidance = st.slider(
            "Guidance scale", g_min, g_max, g_default, step=0.5,
            help="How strictly the image follows your prompt. 5–9 is a good range.",
        )
    else:
        guidance = None
        st.caption("Guidance scale is not used by this model.")

    random_seed = st.checkbox(
        "Use a random seed", value=True,
        help="A seed makes results reproducible: same prompt + same seed = same image.",
    )
    chosen_seed = None
    if not random_seed:
        chosen_seed = st.number_input("Seed", min_value=0, max_value=MAX_SEED,
                                      value=42, step=1)

    file_format = st.radio("Download format", ["PNG", "JPEG"], horizontal=True)


# ----------------------------------------------------------------
# 6. MAIN AREA — header and token check
# ----------------------------------------------------------------
st.title("🎨 Text-to-Image Studio")
st.write("Describe an image in plain words and an open-source AI model will create it.")

hf_token = get_hf_token()
if not hf_token:
    st.error(
        "No Hugging Face token found. Add `HF_TOKEN` to Streamlit Secrets "
        "(Streamlit Cloud → your app → Settings → Secrets) or set it as an "
        "environment variable in Colab, then reload this page."
    )
    st.stop()  # nothing else can work without a token, so stop the script here

client = get_client(hf_token)

# The session state is Streamlit's memory between reruns. We use it to keep
# the last generated image on screen after other widgets are clicked.
if "result" not in st.session_state:
    st.session_state.result = None

left, right = st.columns([1, 1], gap="large")

# ----------------------------------------------------------------
# 7. LEFT COLUMN — prompt inputs and the Generate button
# ----------------------------------------------------------------
with left:
    prompt = st.text_area(
        "Prompt",
        placeholder="A cozy reading nook by a rainy window, warm lamp light, "
                    "watercolor illustration",
        height=140,
        max_chars=MAX_PROMPT_CHARS,
    )
    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        placeholder="blurry, low quality, distorted hands, text, watermark",
        height=80,
        disabled=not model_cfg["supports_negative_prompt"],
        help="Things you do NOT want in the image. Only some models support this.",
    )
    if not model_cfg["supports_negative_prompt"]:
        st.caption("The selected model ignores negative prompts. "
                   "Switch to Stable Diffusion XL to use one.")

    generate_clicked = st.button("Generate image", type="primary", width="stretch")

    with st.expander("Tips for better prompts"):
        st.markdown(
            "- Say **what** is in the image, **where**, and **in what style**.\n"
            "- Add lighting and mood: *golden hour*, *soft studio light*, *foggy*.\n"
            "- Name a medium: *oil painting*, *35mm photo*, *3D render*, *pencil sketch*.\n"
            "- Turn off the random seed to recreate or fine-tune an image you liked."
        )

# ----------------------------------------------------------------
# 8. GENERATION — runs only when the button was clicked
# ----------------------------------------------------------------
if generate_clicked:
    problems = validate_inputs(prompt, width, height)
    if problems:
        for problem in problems:
            left.warning(problem)
    else:
        seed = random.randint(0, MAX_SEED) if random_seed else int(chosen_seed)
        with right:
            # st.status shows an animated "working" box while the code inside runs.
            with st.status("Generating your image…", expanded=True) as status:
                st.write(f"Sending your prompt to **{model_cfg['id']}**…")
                start = time.time()
                try:
                    image = generate_image(client, model_cfg, prompt, negative_prompt,
                                           width, height, steps, guidance, seed)
                    elapsed = time.time() - start
                    st.session_state.result = {
                        "image": image,
                        "prompt": prompt.strip(),
                        "model": model_cfg["id"],
                        "seed": seed,
                        "size": f"{image.width} × {image.height}",
                        "seconds": elapsed,
                    }
                    status.update(label=f"Done in {elapsed:.1f} s", state="complete",
                                  expanded=False)
                except Exception as error:  # show a friendly message, never a crash
                    status.update(label="Generation failed", state="error", expanded=True)
                    st.error(explain_error(error))
                    with st.expander("Technical details"):
                        st.code(f"{type(error).__name__}: {error}")

# ----------------------------------------------------------------
# 9. RIGHT COLUMN — show the latest result and the download button
# ----------------------------------------------------------------
with right:
    result = st.session_state.result
    if result is None:
        st.info("Your image will appear here. Write a prompt and click **Generate image**.")
    else:
        st.image(result["image"], caption=result["prompt"], width="stretch")
        st.caption(
            f"Model: {result['model']}  |  Size: {result['size']}  |  "
            f"Seed: {result['seed']}  |  Time: {result['seconds']:.1f} s"
        )
        extension = "png" if file_format == "PNG" else "jpg"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        st.download_button(
            label=f"Download {file_format}",
            data=image_to_bytes(result["image"], file_format),
            file_name=f"generated-{timestamp}.{extension}",
            mime="image/png" if file_format == "PNG" else "image/jpeg",
            on_click="ignore",   # downloading should not rerun the app
            width="stretch",
        )

st.divider()
st.caption("Images are generated by open-source models via Hugging Face Inference Providers. "
           "Do not enter personal or sensitive information in prompts.")
