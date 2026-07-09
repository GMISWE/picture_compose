# Picture Compose

Tastefully place one image into another using **GMI Cloud**.

Upload a destination photo (a room, a balcony, an outdoor scene) and up to 13 source object photos. Gemini 3.1 Pro analyzes the destination's lighting and perspective and writes a precise composition prompt. Gemini 3.1 Flash Image (Nano Banana 2) renders the objects into the scene, matching the existing light so the result reads as a single in-camera photograph.

## Pipeline

```
destination  +  up to 13 source objects  +  optional instruction
         │
         ▼
  Stage 1 — Gemini 3.1 Pro (vision LLM)
    • Analyzes the destination scene
    • Analyzes EACH source object individually
    • Drafts a composition prompt instructing the
      image model to include ALL objects, subtly lit
         │
         ▼
  Stage 2 — Upload images to GMI
    • POST /upload-url  → signed GCS URL + public URL
    • PUT bytes to signed URL
         │
         ▼
  Stage 3 — Gemini 3.1 Flash Image (Nano Banana 2)
    • POST /requests  → request_id
    • GET  /requests/{id} (poll until success)
    • Download outcome.media_urls[0].url
         │
         ▼
       final composite
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GMI_API_KEY=sk-...        # from console.gmicloud.ai → API Keys
python app.py
```

Open http://localhost:7860 in a browser.

## Deploy on GMI AgentBox

This app is AgentBox-ready. It reads the standard MaaS env vars (`GMI_MAAS_BASE_URL`, `GMI_MAAS_API_KEY`, `GMI_MODELS`) when those are injected, and falls back to the local-dev names (`GMI_API_KEY`, `GMI_LLM_BASE`, `GMI_LLM_MODEL`) when running outside AgentBox.

### Build & push the image

```bash
docker build -t <registry>/<your-org>/picture-compose:latest .

# Smoke-test (use a real GMI key)
docker run --rm -p 7860:7860 \
    -e GMI_MAAS_API_KEY=sk-... \
    <registry>/<your-org>/picture-compose:latest

docker push <registry>/<your-org>/picture-compose:latest
```

Or skip the registry step and use **Upload Image** in the AgentBox wizard.

### Register on AgentBox

Walk through the four-step wizard at *Register & List* in the GMI console.

**Step 1 — Basic Info**
- Internal project name: `picture-compose`
- Display name + description: what users see on the catalog card

**Step 2 — Infrastructure**
- **Docker image source:** Registry URL of your pushed image, or upload the local image
- **Compute tier:** Standard (2 vCPU · 4 GB RAM · 10 GiB ephemeral · 30 GiB data) — this app is stateless; most work runs on GMI inference
- **Region:** closest to your users
- **MaaS integration:** **Toggle ON**. In the model selector, select **both**:
    - `google/gemini-3.1-pro-preview` (vision LLM)
    - `gemini-3.1-flash-image-preview` (image generation)

With MaaS on, GMI injects `GMI_MAAS_API_KEY` at runtime — no key in the image. The same key authenticates both the chat-completions endpoint and the image-gen request-queue endpoint.

**Step 3 — Env Variables**

Nothing required — the app picks the right two models out of whatever `GMI_MODELS` AgentBox injects. Optional overrides if you want to pin them explicitly:

| Variable           | Value                                                  |
|--------------------|--------------------------------------------------------|
| `GMI_LLM_MODEL`    | `google/gemini-3.1-pro-preview`                        |
| `GMI_IMAGE_MODEL`  | `gemini-3.1-flash-image-preview`                       |

(Plain values, not secrets. Leave `GMI_MAAS_API_KEY` blank — AgentBox sets it.)

**Step 4 — Review & Register**

Confirm and register. AgentBox builds the container on demand and exposes a public URL. Open it in a browser to use the Gradio UI exactly like local dev. The API KEY field at the top is optional — leave blank to use the AgentBox-injected key, or paste your own to override.

### What runs where

| Layer              | Endpoint                                                                       | Auth                       |
|--------------------|--------------------------------------------------------------------------------|----------------------------|
| Vision LLM         | `${GMI_MAAS_BASE_URL}/v1/chat/completions`                                     | `Bearer $GMI_MAAS_API_KEY` |
| Image upload       | `console.gmicloud.ai/.../upload-url` → signed GCS                              | Same key                   |
| Image generation   | `console.gmicloud.ai/.../requests`                                             | Same key                   |
| Image download     | GCS public URL returned by the job                                             | None (signed URL)          |

## Environment variables (full reference)

| Variable             | AgentBox-injected? | Default                                                                  | Purpose                                           |
|----------------------|--------------------|--------------------------------------------------------------------------|---------------------------------------------------|
| `GMI_MAAS_API_KEY`   | yes                | —                                                                        | Bearer token for both GMI APIs                    |
| `GMI_MAAS_BASE_URL`  | yes                | `https://api.gmi-serving.com`                                            | Chat-completions base                             |
| `GMI_MODELS`         | yes                | —                                                                        | Models selected in Step 2; app picks by substring |
| `GMI_API_KEY`        | local dev          | —                                                                        | Fallback for `GMI_MAAS_API_KEY`                   |
| `GMI_LLM_BASE`       | local dev          | `https://api.gmi-serving.com`                                            | Fallback for `GMI_MAAS_BASE_URL`                  |
| `GMI_RQ_BASE`        | no                 | `https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey`              | Image-gen / upload base                           |
| `GMI_LLM_MODEL`      | optional           | `google/gemini-3.1-pro-preview`                                          | Override vision LLM                               |
| `GMI_IMAGE_MODEL`    | optional           | `gemini-3.1-flash-image-preview`                                         | Override image-gen model                          |
| `PORT`               | optional           | `7860`                                                                   | Port Gradio binds to                              |

## Files

- `app.py` — the whole application (Gradio UI + GMI client + pipeline)
- `requirements.txt` — `gradio`, `requests`, `Pillow`
- `Dockerfile` — Python 3.11-slim container
- `.dockerignore` — keeps the image lean
