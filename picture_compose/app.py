"""
Picture Compose — Tastefully place one image into another, end to end on GMI Cloud.

Two GMI APIs are involved:

  1) LLM (vision) — OpenAI-compatible chat completions
        POST https://api.gmi-serving.com/v1/chat/completions
        Model: google/gemini-3.1-pro-preview
        Inline data-URL images are accepted. Returns text (we ask for JSON).

  2) Image generation — async request-queue API
        POST {RQ_BASE}/upload-url                  -> signed GCS URL + public URL
        PUT  <signed_url>                          -> raw image bytes go to GCS
        POST {RQ_BASE}/requests?source_product=studio
        GET  {RQ_BASE}/requests/{request_id}       -> poll until status == "success"

     where RQ_BASE = https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey
     Model: gemini-3.1-flash-image-preview (supports 0-14 reference images)

Pipeline:
  destination (+ optional source + optional instruction)
    -> Stage 1: Gemini 3.1 Pro analyzes both images and drafts a precise
                composition prompt (returned as JSON for reliability).
    -> Stage 2: Upload destination & source to GMI's GCS bucket, then submit
                an image-gen job with those URLs + the prompt, then poll.
    -> Download the resulting composite and show it in the UI.

Run:
  export GMI_API_KEY=sk-...
  pip install -r requirements.txt
  python app.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import gradio as gr
import requests
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
#
# AgentBox-compatible: when this container runs on GMI AgentBox with MaaS
# integration enabled, GMI injects GMI_MAAS_BASE_URL / GMI_MAAS_API_KEY /
# GMI_MODELS at runtime. For local development we fall back to GMI_API_KEY /
# GMI_LLM_BASE / GMI_LLM_MODEL so existing dev workflows keep working.
# ---------------------------------------------------------------------------

GMI_API_KEY = (os.getenv("GMI_MAAS_API_KEY")
               or os.getenv("GMI_API_KEY")
               or "")

LLM_BASE = (os.getenv("GMI_MAAS_BASE_URL")
            or os.getenv("GMI_LLM_BASE")
            or "https://api.gmi-serving.com").rstrip("/")

# Image generation / async request-queue base (separate API; no AgentBox var
# for this one — it stays on console.gmicloud.ai).
RQ_BASE = os.getenv(
    "GMI_RQ_BASE",
    "https://console.gmicloud.ai/api/v1/ie/requestqueue/apikey",
).rstrip("/")


def _pick_model_from_env(prefer_substring: str, fallback: str) -> str:
    """AgentBox sets GMI_MODELS to a (possibly comma-separated) list of model
    IDs the user selected in Step 2 of the register wizard. This app needs two
    models (a vision LLM and an image-gen model), so we pick by substring."""
    raw = (os.getenv("GMI_MODELS") or "").strip()
    if not raw:
        return fallback
    models = [m.strip() for m in raw.split(",") if m.strip()]
    for m in models:
        if prefer_substring.lower() in m.lower():
            return m
    # If GMI_MODELS contains a single model that doesn't match our substring,
    # don't silently mis-route — fall back to the configured default.
    return fallback


LLM_MODEL = _pick_model_from_env(
    "pro-preview",
    os.getenv("GMI_LLM_MODEL", "google/gemini-3.1-pro-preview"),
)
IMAGE_MODEL = _pick_model_from_env(
    "flash-image",
    os.getenv("GMI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"),
)

HTTP_TIMEOUT = 90      # per request
POLL_INTERVAL = 2.5    # seconds between polls
POLL_TIMEOUT = 300     # max wait for an image-gen job

ALLOWED_ASPECTS = ["16:9", "9:16", "5:4", "4:5", "4:3", "3:4", "3:2", "2:3", "1:1", "21:9"]


class GMIError(RuntimeError):
    """Raised when GMI returns an error or an unexpected payload."""


def _resolve_api_key(ui_key: str | None) -> str:
    """UI value beats env var. Raises a friendly GMIError if neither is set."""
    key = (ui_key or "").strip() or GMI_API_KEY
    if not key:
        raise GMIError(
            "No GMI API key provided. Paste it in the API KEY field at the top, "
            "or set the GMI_API_KEY environment variable before launching."
        )
    return key.encode("ascii", errors="ignore").decode("ascii")


def _auth_headers(api_key: str, json_body: bool = True) -> dict:
    h = {"Authorization": f"Bearer {api_key}"}
    if json_body:
        h["Content-Type"] = "application/json"
    return h


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _pil_to_bytes(img: Image.Image, fmt: str) -> bytes:
    if fmt.upper() == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    return buf.getvalue()


def _pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    data = _pil_to_bytes(img, fmt)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _choose_aspect_ratio(img: Image.Image) -> str:
    """Pick the supported aspect ratio closest to the destination image."""
    w, h = img.size
    if h == 0:
        return "1:1"
    target = w / h
    best, best_diff = "1:1", float("inf")
    for ratio in ALLOWED_ASPECTS:
        a, b = ratio.split(":")
        diff = abs(target - (int(a) / int(b)))
        if diff < best_diff:
            best, best_diff = ratio, diff
    return best


def _format_for(img: Image.Image) -> str:
    """PNG if there's transparency, else JPEG (smaller payload)."""
    return "PNG" if img.mode in ("RGBA", "LA", "P") else "JPEG"


# ---------------------------------------------------------------------------
# Stage 1 — LLM vision analysis -> composition prompt
# ---------------------------------------------------------------------------

ANALYSIS_INSTRUCTIONS = """\
You are a photo-compositing analyst. Your goal is REALISM through SUBTLETY —
the inserted object(s) should look like they were simply photographed in the
scene, not relit or restyled. Avoid drama.

Two outcomes determine quality:
  (A) The destination scene must remain pixel-identical OUTSIDE the inserted
      objects. No restyling, recoloring, recropping, or regenerating of the
      surrounding scene.
  (B) Every inserted object must blend so naturally that a viewer would not
      notice an edit. Match the existing light gently. Do not add rim lights,
      god rays, dramatic shadows, glow, halos, vignette, or any stylization.

You will be shown a DESTINATION image and {n_source_label}.

User's note (optional — may be empty, may describe placement, may describe the
object(s) to add, or both): {user_instruction}

CRITICAL RULES FOR MULTIPLE SOURCE IMAGES:
  • If N source images are provided, you MUST analyze EACH ONE INDIVIDUALLY.
  • The "object_analyses" array MUST have EXACTLY N entries (one per source
    image, in the same order they were provided).
  • The composition_prompt MUST instruct the image model to include ALL N
    objects in the result — do NOT omit any source object.
  • Each object must get its own concrete placement, scale, and orientation
    in the composition_prompt.

If a SOURCE image is provided, insert THAT specific object, preserving its
identity (shape, material, color, key details). If no SOURCE is provided,
infer the object from the user's note. If the user's note says nothing about
placement, choose a placement that is natural and uncluttered — somewhere a
person would actually put that object in real life.

Return ONLY a single JSON OBJECT (not an array, no markdown fences) with
EXACTLY these three keys:

{{
  "scene_analysis": "<3-4 sentences in plain natural language. Describe the \
scene: what it is, where the light is coming from in everyday terms (e.g. \
'soft daylight from a tall window on the left' or 'warm overhead pendant \
light, fairly even across the room' or 'overcast outdoor light, no strong \
direction'). Note how the existing shadows behave — short or long, soft or \
crisp — in natural words, not Kelvin numbers. Mention camera angle and the \
overall mood/color cast. Identify where there is open, uncluttered space for \
placing the new object(s).>",

  "object_analyses": [
    "<2 sentences on the FIRST source object: what it is, what it's made of, \
its color and rough scale. Note any baked-in lighting in the source image \
that should be softened so the destination's light can take over naturally.>",
    "<2 sentences on the SECOND source object (if provided), same format.>",
    "<...one entry per source image, in the order provided. If no source \
images, exactly one entry describing the object inferred from the user's \
instruction.>"
  ],

  "composition_prompt": "<A single paragraph that will be sent verbatim to an \
image-editing model. Begin with: 'EDIT the destination image by adding the \
following object(s): <enumerate EVERY object with a concrete placement, \
scale, and orientation in plain words — e.g. \"a green upholstered armchair \
in the front-left, facing slightly toward the window; a tall floor lamp \
behind it to the right at standing height\">. PRESERVE the destination image \
exactly — do NOT regenerate, restyle, recolor, recrop, or alter any pixel \
outside the added object(s).' Then in plain natural language describe the \
integration: light each object gently to match the scene's existing light \
(described in everyday terms — soft, warm, cool, diffuse, directional from \
<where>), cast a SUBTLE shadow for each that matches the other shadows in \
the scene in length, direction, and softness — do not make the new shadows \
stronger or longer than what already exists, sit each object correctly in \
perspective at a believable scale, add a small soft contact shadow where it \
touches the floor or surface, inherit the scene's slight color cast, and \
match the existing photo's grain/sharpness/depth of field. End with: 'IMPORTANT: \
EVERY ONE of the source objects must appear in the final image — do not omit \
any. The integration must be UNDERSTATED — the objects should look as if they \
were simply present when the photo was taken. Avoid dramatic relighting, rim \
lights, halos, exaggerated shadows, added glow, or anything stylized that \
would draw attention to the edit. Output only the edited image.'>"
}}
"""


@dataclass
class AnalysisResult:
    scene_analysis: str
    object_analyses: list[str]   # one per source image (or one inferred if no source)
    composition_prompt: str
    raw: str


def _extract_object_analyses(obj: dict) -> list[str]:
    """Accept either object_analyses (list) or object_analysis (string)."""
    val = obj.get("object_analyses")
    if val is None:
        val = obj.get("object_analysis")
    if val is None:
        return []
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, list):
        return [str(a).strip() for a in val if isinstance(a, str) and a.strip()]
    return []


def analyze_and_prompt(
    destination: Image.Image,
    sources: list[Image.Image],
    user_instruction: str,
    api_key: str,
) -> AnalysisResult:
    """Call Gemini 3.1 Pro on the LLM endpoint and parse the JSON it returns."""
    if user_instruction.strip():
        instruction = user_instruction.strip()
    elif sources:
        instruction = (
            "(no specific placement requested — choose the most natural, "
            "uncluttered position that complements the scene's composition.)"
        )
    else:
        instruction = "(no instruction)"

    # Multimodal user message. (For multimodal calls on GMI's Gemini, we
    # keep everything inside a single user message — no separate system role.)
    user_content: list[dict] = [
        {"type": "text", "text": "DESTINATION image (the scene to edit):"},
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(destination, "JPEG")}},
    ]
    if sources:
        if len(sources) == 1:
            user_content.append(
                {"type": "text", "text": "SOURCE image (the object to insert):"}
            )
        else:
            user_content.append({
                "type": "text",
                "text": (
                    f"SOURCE images ({len(sources)} provided). These may be "
                    "multiple different objects to insert into the scene, "
                    "or multiple views/angles of the same object to help you "
                    "preserve its identity. Use the user's instruction (if any) "
                    "to decide."
                ),
            })
        for i, src in enumerate(sources, 1):
            if len(sources) > 1:
                user_content.append({"type": "text", "text": f"Source #{i}:"})
            user_content.append(
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(src, "JPEG")}}
            )
    else:
        user_content.append({
            "type": "text",
            "text": "No source image provided — infer the object(s) from the user instruction.",
        })
    user_content.append(
        {"type": "text", "text": ANALYSIS_INSTRUCTIONS.format(
            user_instruction=instruction,
            n_source_label=(
                f"{len(sources)} SOURCE object image(s) (analyze EACH separately and "
                f"include ALL of them in the output)" if sources
                else "no SOURCE image (infer the object from the user note)"
            ),
        )}
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.4,
        # Bumped from 1500: Gemini 3.1 Pro spends tokens on hidden "thinking"
        # before emitting visible content. A long instructions block + JSON
        # mode + thinking can exhaust a small budget, leaving content="".
        "max_tokens": 8000,
        "response_format": {"type": "json_object"},
    }

    url = f"{LLM_BASE}/v1/chat/completions"
    resp = requests.post(url, headers=_auth_headers(api_key), json=payload, timeout=HTTP_TIMEOUT)
    if resp.status_code >= 400:
        raise GMIError(f"LLM call failed {resp.status_code}: {resp.text[:500]}")

    body = resp.json()
    try:
        choice = body["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as e:
        raise GMIError(f"Unexpected LLM response shape: {body}") from e

    # Some reasoning models on GMI put the visible answer in `content` and
    # the chain-of-thought in `reasoning_content`; others put EVERYTHING in
    # `reasoning_content` and leave `content` empty (or wrap it in <think>
    # tags). Mirror the GMI ComfyUI LLM-node parser: prefer cleaned content,
    # then raw content, then reasoning_content.
    raw_content = (message.get("content") or "")
    raw_reasoning = (message.get("reasoning_content") or "")
    content_no_think = re.sub(
        r"<think>[\s\S]*?</think>", "", raw_content, flags=re.DOTALL
    ).strip()
    raw = content_no_think or raw_content.strip() or raw_reasoning.strip()

    if not raw:
        finish = choice.get("finish_reason") or "?"
        usage = body.get("usage") or {}
        raise GMIError(
            f"LLM returned empty content (finish_reason={finish}, usage={usage}). "
            f"This usually means the model exhausted its token budget on internal "
            f"thinking. Try raising max_tokens or simplifying the instruction."
        )

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"[\{\[][\s\S]*[\}\]]", cleaned)
        if not m:
            raise GMIError(
                f"LLM did not return parseable JSON.\n"
                f"finish_reason={choice.get('finish_reason')!r}, "
                f"len(content)={len(raw_content)}, len(reasoning)={len(raw_reasoning)}.\n"
                f"First 800 chars of what we got:\n{raw[:800]}"
            )
        obj = json.loads(m.group(0))

    # Gemini sometimes returns [{...}] instead of {...} -- unwrap.
    if isinstance(obj, list):
        if not obj or not isinstance(obj[0], dict):
            raise GMIError(f"LLM returned non-dict JSON: {obj!r}\nRaw:\n{raw[:1000]}")
        obj = obj[0]
    if not isinstance(obj, dict):
        raise GMIError(
            f"LLM returned unexpected JSON type {type(obj).__name__}. Raw:\n{raw[:1000]}"
        )

    return AnalysisResult(
        scene_analysis=(obj.get("scene_analysis") or "").strip(),
        object_analyses=_extract_object_analyses(obj),
        composition_prompt=(obj.get("composition_prompt") or "").strip(),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Stage 2 — upload images to GMI's signed-URL bucket
# ---------------------------------------------------------------------------

def upload_image(img: Image.Image, api_key: str) -> str:
    """
    Two-step upload to GMI:
      1) POST {RQ_BASE}/upload-url with {"file_type": "...", "type": "image"}
         -> {"upload_url": <signed GCS URL>, "public_url": <stable URL>}
      2) PUT the raw bytes to upload_url with the matching image/* Content-Type
    Returns the stable public URL.
    """
    fmt = _format_for(img)                   # "PNG" or "JPEG"
    file_type = "png" if fmt == "PNG" else "jpeg"

    # Step 1: ask GMI for a signed upload URL.
    r1 = requests.post(
        f"{RQ_BASE}/upload-url",
        headers=_auth_headers(api_key),
        json={"file_type": file_type, "type": "image"},
        timeout=HTTP_TIMEOUT,
    )
    if r1.status_code >= 400:
        raise GMIError(f"upload-url failed {r1.status_code}: {r1.text[:300]}")

    body = r1.json()
    signed_url = body.get("upload_url")
    public_url = body.get("public_url")
    if not signed_url or not public_url:
        raise GMIError(f"upload-url returned unexpected payload: {body}")

    # Step 2: PUT bytes to GCS. (No Authorization header here — the signature
    # in the URL is what authorizes the write.)
    r2 = requests.put(
        signed_url,
        data=_pil_to_bytes(img, fmt),
        headers={"Content-Type": f"image/{file_type}"},
        timeout=HTTP_TIMEOUT,
    )
    if r2.status_code >= 400:
        raise GMIError(f"GCS upload failed {r2.status_code}: {r2.text[:300]}")

    return public_url


# ---------------------------------------------------------------------------
# Stage 3 — submit image-gen job, poll, download result
# ---------------------------------------------------------------------------

def submit_image_job(
    composition_prompt: str,
    reference_urls: list[str],
    image_size: str,
    aspect_ratio: str,
    api_key: str,
) -> str:
    """POST a generation request and return its request_id."""
    payload = {
        "model": IMAGE_MODEL,
        "payload": {
            "prompt": composition_prompt,
            "image": reference_urls,        # 0-14 public URLs
            "image_size": image_size,       # "512" | "1K" | "2K" | "4K"
            "aspect_ratio": aspect_ratio,
        },
    }
    r = requests.post(
        f"{RQ_BASE}/requests?source_product=studio",
        headers=_auth_headers(api_key),
        json=payload,
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code >= 400:
        raise GMIError(f"image-gen submit failed {r.status_code}: {r.text[:500]}")
    body = r.json()
    rid = body.get("request_id")
    if not rid:
        raise GMIError(f"submit returned no request_id: {body}")
    return rid


def poll_until_done(request_id: str, api_key: str, on_tick=None) -> dict:
    """Poll /requests/{id} until status is success or failed/error/cancelled."""
    url = f"{RQ_BASE}/requests/{request_id}"
    deadline = time.monotonic() + POLL_TIMEOUT
    last_status = None
    while time.monotonic() < deadline:
        r = requests.get(url, headers=_auth_headers(api_key, json_body=False), timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise GMIError(f"poll failed {r.status_code}: {r.text[:300]}")
        body = r.json()
        status = (body.get("status") or "").lower()
        if status != last_status and on_tick is not None:
            on_tick(status)
            last_status = status
        if status == "success":
            return body
        if status in ("failed", "error", "cancelled"):
            err = (body.get("outcome") or {}).get("error") or body
            raise GMIError(f"image-gen job {status}: {err}")
        time.sleep(POLL_INTERVAL)
    raise GMIError(f"image-gen job timed out after {POLL_TIMEOUT}s (request_id={request_id})")


def poll_until_done_iter(request_id: str, api_key: str):
    """Generator version of poll_until_done — yields {'status': str} on every change,
    finally yields {'final': body} when the job succeeds. Used by pipeline generators
    so they can re-yield UI status updates while waiting."""
    url = f"{RQ_BASE}/requests/{request_id}"
    deadline = time.monotonic() + POLL_TIMEOUT
    last_status = None
    while time.monotonic() < deadline:
        r = requests.get(url, headers=_auth_headers(api_key, json_body=False), timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise GMIError(f"poll failed {r.status_code}: {r.text[:300]}")
        body = r.json()
        status = (body.get("status") or "").lower()
        if status != last_status:
            yield {"status": status}
            last_status = status
        if status == "success":
            yield {"final": body}
            return
        if status in ("failed", "error", "cancelled"):
            err = (body.get("outcome") or {}).get("error") or body
            raise GMIError(f"image-gen job {status}: {err}")
        time.sleep(POLL_INTERVAL)
    raise GMIError(f"image-gen job timed out after {POLL_TIMEOUT}s (request_id={request_id})")


def _image_gen_iter(composition_prompt: str, reference_urls: list[str],
                    image_size: str, aspect: str, api_key: str, base_pct: int = 55):
    """Generator: submits an image-gen job and yields status tuples until done.
    Yields ('status', percent, message) until success, then ('done', image, rid, url)."""
    yield ('status', base_pct, "SUBMITTING IMAGE-GEN JOB…")
    rid = submit_image_job(
        composition_prompt=composition_prompt,
        reference_urls=reference_urls,
        image_size=image_size,
        aspect_ratio=aspect,
        api_key=api_key,
    )
    yield ('status', base_pct + 3, f"JOB SUBMITTED · {rid[:8]} · POLLING…")

    final = None
    span = 100 - base_pct - 10  # room from base_pct+5 .. ~90
    for upd in poll_until_done_iter(rid, api_key):
        if 'final' in upd:
            final = upd['final']
            break
        s = upd['status']
        bumps = {"queued": 0.2, "running": 0.55, "processing": 0.55}
        pct = base_pct + 5 + int(span * bumps.get(s, 0.75))
        yield ('status', pct, f"IMAGE-GEN STATUS: {s.upper()}…")

    media = ((final.get("outcome") or {}).get("media_urls") or [])
    if not media:
        raise GMIError(f"Job succeeded but returned no media: {final}")
    result_url = media[0].get("url") or ""
    if not result_url:
        raise GMIError(f"Empty media URL in response: {final}")

    yield ('status', 95, "DOWNLOADING RESULT…")
    result_img = download_image(result_url)
    yield ('done', result_img, rid, result_url)


def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))


# ---------------------------------------------------------------------------
# Orchestration (generator pipelines + helpers)
# ---------------------------------------------------------------------------

# Appended to the cached prompt on a RECOMPOSE so the image model explores
# something different rather than reproducing the same composite.
VARIATION_DIRECTIVE = (
    "\n\nALTERNATIVE TAKE: this is a second attempt — explore a different but "
    "equally natural placement, orientation, or arrangement for the added "
    "object(s). All preservation rules above remain in effect (do not modify "
    "any pixel of the destination outside the inserted objects, keep lighting "
    "subtle and natural, and include EVERY source object)."
)


MAX_SOURCES = 13  # Nano Banana 2 supports 14 reference images; 1 is the destination.


def _load_sources(files) -> list[Image.Image]:
    """Convert Gradio gr.File output (list of paths or NamedString) into PIL images.
    Raises a plain Exception on failure; the UI layer wraps it for display."""
    if not files:
        return []
    imgs: list[Image.Image] = []
    for entry in files:
        path = entry if isinstance(entry, str) else getattr(entry, "name", None) or str(entry)
        try:
            imgs.append(Image.open(path))
        except Exception as e:
            raise RuntimeError(f"could not open {path}: {e}") from e
    return imgs


def _format_object_analyses(analyses: list[str]) -> str:
    """Render the per-object analyses for the SCENE/OBJECT/PROMPT Textbox."""
    if not analyses:
        return "(no objects analyzed)"
    if len(analyses) == 1:
        return analyses[0]
    return "\n\n".join(
        f"━━ OBJECT #{i} ━━\n{a}" for i, a in enumerate(analyses, 1)
    )


import html as _html
import traceback as _tb


def _progress_html(pct: int, msg: str) -> str:
    pct = max(0, min(100, int(pct)))
    # Inline styles are belt-and-suspenders against Gradio's own CSS that
    # sometimes paints inner spans dark before our external rules can win.
    return (
        f'<div class="gmi-progress" style="color:#ffffff !important;">'
        f'  <div class="gmi-progress-label" style="color:#ffffff !important;">'
        f'    <span style="color:#ffffff !important;font-weight:700;">{_html.escape(msg)}</span>'
        f'    <span class="pct" style="color:#C5F542 !important;font-weight:700;">{pct}%</span>'
        f'  </div>'
        f'  <div class="gmi-progress-track">'
        f'    <div class="gmi-progress-fill" style="width: {pct}%"></div>'
        f'    <div class="gmi-progress-indeterminate"></div>'
        f'  </div>'
        f'</div>'
    )


# ----- output-tuple helpers ------------------------------------------------
# Output order in the UI binding is:
#   0 result_out (Image)
#   1 scene_out (Textbox)
#   2 object_out (Textbox)
#   3 prompt_out (Textbox)
#   4 debug_out (Textbox)
#   5 error_out (HTML)
#   6 status_out (HTML)   <- the in-output-column progress bar
#   7 cache_state (State)
#
# Every yield from a generator must be an 8-tuple in this order.

def _status_outputs(pct: int, msg: str, prev_cache=None):
    """Intermediate yield: show progress bar, leave existing outputs alone."""
    return (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(visible=False, value=""),                   # clear any prior error
        gr.update(visible=True, value=_progress_html(pct, msg)),  # show status bar
        prev_cache,
    )


def _ok_outputs(image, scene, obj, prompt, debug, cache):
    """Final success yield: populate outputs, hide error AND status."""
    return (
        image, scene, obj, prompt, debug,
        gr.update(visible=False, value=""),
        gr.update(visible=False, value=""),
        cache,
    )


def _err_output(title: str, msg: str, tb: str | None = None, prev_cache=None):
    """Failure yield: keep prior outputs visible, hide status bar, show error panel."""
    safe_msg = _html.escape(msg)
    tb_block = ""
    if tb:
        tb_block = (
            "<details><summary>FULL TRACEBACK</summary>"
            f"<pre>{_html.escape(tb)}</pre></details>"
        )
    html_block = (
        f'<div class="gmi-error">'
        f'<span class="title">▮ {_html.escape(title)}</span>'
        f'{safe_msg}{tb_block}'
        f'</div>'
    )
    return (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(visible=True, value=html_block),
        gr.update(visible=False, value=""),
        prev_cache,
    )


# ----- generator pipelines ------------------------------------------------

def run_pipeline(api_key, destination, source_files, instruction, image_size,
                 aspect_ratio_choice, prev_cache=None):
    """Full pipeline as a generator. Yields intermediate status updates so the
    output column's progress bar reflects what's happening in real time."""
    yield _status_outputs(2, "VALIDATING INPUTS…", prev_cache=prev_cache)

    if destination is None:
        yield _err_output("MISSING DESTINATION",
                          "Upload a destination image to start.",
                          prev_cache=prev_cache)
        return

    try:
        sources = _load_sources(source_files)
    except Exception as e:
        yield _err_output("COULD NOT READ SOURCE IMAGES", str(e),
                          _tb.format_exc(), prev_cache=prev_cache)
        return

    if len(sources) > MAX_SOURCES:
        yield _err_output(
            "TOO MANY SOURCE IMAGES",
            f"Up to {MAX_SOURCES} source images are supported "
            f"(you uploaded {len(sources)}). The image model accepts 14 reference "
            "images total — one is reserved for the destination.",
            prev_cache=prev_cache,
        )
        return

    if not sources and not instruction.strip():
        yield _err_output(
            "NOTHING TO ADD",
            "Either upload one or more source images OR describe what to add. "
            "The instruction is optional only when you also provide source image(s).",
            prev_cache=prev_cache,
        )
        return

    # Resolve the API key ONCE — passed explicitly through every GMI call below.
    # (Earlier we used contextvars here, but gradio runs each generator yield in
    # a different asyncio context, which made set()/reset() raise ValueError.)
    try:
        key = _resolve_api_key(api_key)
    except GMIError as e:
        yield _err_output("NO API KEY", str(e), prev_cache=prev_cache)
        return

    try:
        yield _status_outputs(8, "ANALYZING SCENE WITH GEMINI 3.1 PRO…", prev_cache=prev_cache)
        analysis = analyze_and_prompt(destination, sources, instruction, key)

        # As soon as we have analyses, surface them in the UI so the user can
        # read while uploads + image-gen proceed.
        obj_display = _format_object_analyses(analysis.object_analyses)
        yield (
            gr.update(),                                # image still empty
            analysis.scene_analysis,
            obj_display,
            analysis.composition_prompt,
            gr.update(),
            gr.update(visible=False, value=""),
            gr.update(visible=True, value=_progress_html(25, "UPLOADING DESTINATION IMAGE…")),
            prev_cache,
        )

        dest_url = upload_image(destination, key)
        ref_urls = [dest_url]

        for i, src in enumerate(sources, 1):
            pct = 30 + int(20 * i / max(len(sources), 1))
            yield _status_outputs(pct,
                                  f"UPLOADING SOURCE IMAGE {i}/{len(sources)}…",
                                  prev_cache=prev_cache)
            ref_urls.append(upload_image(src, key))

        aspect = (_choose_aspect_ratio(destination)
                  if aspect_ratio_choice == "auto" else aspect_ratio_choice)

        result_img = None
        rid = None
        result_url = None
        for upd in _image_gen_iter(analysis.composition_prompt, ref_urls,
                                   image_size, aspect, key, base_pct=55):
            if upd[0] == 'status':
                _, pct, msg = upd
                yield _status_outputs(pct, msg, prev_cache=prev_cache)
            else:
                _, result_img, rid, result_url = upd

        debug = (
            f"request_id: {rid}\n"
            f"result_url: {result_url}\n"
            f"references: {len(ref_urls)} (1 destination + {len(sources)} source)\n"
            f"objects analyzed: {len(analysis.object_analyses)}\n"
            f"aspect: {aspect}, size: {image_size}\n"
            f"variation #: 1 (initial)"
        )
        cache = {
            "composition_prompt": analysis.composition_prompt,
            "scene_analysis": analysis.scene_analysis,
            "object_analyses": analysis.object_analyses,
            "ref_urls": ref_urls,
            "aspect": aspect,
            "image_size": image_size,
            "variation_count": 1,
        }
        yield _ok_outputs(result_img, analysis.scene_analysis, obj_display,
                          analysis.composition_prompt, debug, cache=cache)
    except GMIError as e:
        yield _err_output("GMI PIPELINE ERROR", str(e), prev_cache=prev_cache)
    except requests.exceptions.RequestException as e:
        yield _err_output("NETWORK ERROR", f"Could not reach GMI Cloud: {e}",
                          _tb.format_exc(), prev_cache=prev_cache)
    except Exception as e:
        yield _err_output("UNEXPECTED ERROR", f"{type(e).__name__}: {e}",
                          _tb.format_exc(), prev_cache=prev_cache)


def recompose_pipeline(api_key, image_size, aspect_ratio_choice, prev_cache=None):
    """Cached pipeline as a generator. Skips analyze + upload, regenerates with
    a variation directive."""
    yield _status_outputs(5, "PREPARING ALTERNATIVE TAKE…", prev_cache=prev_cache)

    if not prev_cache or not prev_cache.get("composition_prompt"):
        yield _err_output(
            "NO PRIOR COMPOSE",
            "Run COMPOSE first. RECOMPOSE reuses the previous analysis and "
            "uploaded images to produce an alternative take — there's nothing "
            "to vary yet.",
            prev_cache=prev_cache,
        )
        return

    try:
        key = _resolve_api_key(api_key)
    except GMIError as e:
        yield _err_output("NO API KEY", str(e), prev_cache=prev_cache)
        return

    try:
        composition_prompt = prev_cache["composition_prompt"] + VARIATION_DIRECTIVE
        aspect = (prev_cache["aspect"]
                  if aspect_ratio_choice == "auto" else aspect_ratio_choice)

        result_img = None
        rid = None
        result_url = None
        for upd in _image_gen_iter(composition_prompt, prev_cache["ref_urls"],
                                   image_size, aspect, key, base_pct=20):
            if upd[0] == 'status':
                _, pct, msg = upd
                yield _status_outputs(pct, msg, prev_cache=prev_cache)
            else:
                _, result_img, rid, result_url = upd

        new_count = int(prev_cache.get("variation_count", 1)) + 1
        obj_display = _format_object_analyses(prev_cache.get("object_analyses", []))
        debug = (
            f"request_id: {rid}\n"
            f"result_url: {result_url}\n"
            f"references: {len(prev_cache['ref_urls'])} (cached — no re-upload, no re-analysis)\n"
            f"aspect: {aspect}, size: {image_size}\n"
            f"variation #: {new_count}"
        )
        updated_cache = {**prev_cache, "aspect": aspect,
                         "image_size": image_size, "variation_count": new_count}
        yield _ok_outputs(result_img, prev_cache["scene_analysis"], obj_display,
                          composition_prompt, debug, cache=updated_cache)
    except GMIError as e:
        yield _err_output("GMI PIPELINE ERROR", str(e), prev_cache=prev_cache)
    except requests.exceptions.RequestException as e:
        yield _err_output("NETWORK ERROR", f"Could not reach GMI Cloud: {e}",
                          _tb.format_exc(), prev_cache=prev_cache)
    except Exception as e:
        yield _err_output("UNEXPECTED ERROR", f"{type(e).__name__}: {e}",
                          _tb.format_exc(), prev_cache=prev_cache)


GMI_LIME = "#C5F542"
GMI_BLACK = "#000000"

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono:wght@400;700&display=swap');

/* ---------- root ---------- */
body, .gradio-container, gradio-app {
    background: #0a0a0a !important;
    color: #e8e8e8 !important;
    font-family: 'Space Mono', ui-monospace, SFMono-Regular, monospace !important;
}
.gradio-container { max-width: 1280px !important; padding: 24px !important; }
* { border-radius: 0 !important; }

/* ---------- strip default surfaces but allow our own bordered sections ---------- */
.block, .form, .gr-box, .gr-panel, .gr-form, .gr-group,
.gr-block, .gradio-row, .gradio-column {
    background: transparent !important;
    box-shadow: none !important;
}

/* ---------- hero ---------- */
.gmi-hero {
    padding: 12px 0 24px 0 !important;
    margin-bottom: 16px;
}
.gmi-badge {
    display: inline-block;
    border: 1px solid #4a4a4a;
    padding: 6px 10px;
    font-size: 11px;
    letter-spacing: 0.18em;
    color: #d8d8d8;
    text-transform: uppercase;
    margin-bottom: 22px;
    font-family: 'Space Mono', monospace;
}
.gmi-badge .dot { color: #C5F542; margin-right: 6px; }
.gmi-hero h1 {
    font-family: 'VT323', 'Space Mono', monospace !important;
    font-size: 84px !important;
    line-height: 0.95 !important;
    letter-spacing: 0.01em;
    color: #fff !important;
    margin: 0 0 12px 0 !important;
    font-weight: 400 !important;
}
.gmi-hero h1 .accent { color: #C5F542; }
.gmi-hero .subtitle {
    color: #c8c8c8;
    font-size: 14px;
    line-height: 1.55;
    max-width: 720px;
    margin: 12px 0 0 0;
}

/* ---------- section labels ---------- */
.gmi-section-label {
    color: #C5F542 !important;
    font-size: 11px !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    margin: 24px 0 8px 0 !important;
    padding: 0 !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
}
.gmi-section-label::before {
    content: "▸ ";
    color: #C5F542;
    margin-right: 4px;
}

/* ---------- inputs (dark with clearly visible borders) ---------- */
input, textarea, select,
.gr-input input, .gr-textbox textarea, .gr-dropdown select,
input[type="text"], input[type="password"] {
    background: #141414 !important;
    color: #e8e8e8 !important;
    border: 1px solid #3a3a3a !important;
    font-family: 'Space Mono', monospace !important;
    padding: 12px !important;
    font-size: 13px !important;
}
input::placeholder, textarea::placeholder { color: #666 !important; }
input:focus, textarea:focus, select:focus {
    border-color: #C5F542 !important;
    outline: none !important;
    background: #181818 !important;
}
textarea[readonly], .gr-textbox textarea[readonly] {
    background: #141414 !important;
    color: #d8d8d8 !important;
    border: 1px solid #3a3a3a !important;
    cursor: text;
}

/* ---------- dropdowns: every layer dark ---------- */
.gr-dropdown, .gr-dropdown *,
[data-testid="dropdown"], [data-testid="dropdown"] *,
.wrap-inner, .wrap, .secondary-wrap,
.gr-dropdown .container, .gr-dropdown .wrap {
    background: #141414 !important;
    color: #e8e8e8 !important;
}
.gr-dropdown, [data-testid="dropdown"] {
    border: 1px solid #3a3a3a !important;
}
/* The popup options menu */
ul.options, ul[role="listbox"], .options {
    background: #141414 !important;
    color: #e8e8e8 !important;
    border: 1px solid #3a3a3a !important;
}
ul.options li, ul[role="listbox"] li, .options .item, .options li {
    background: #141414 !important;
    color: #e8e8e8 !important;
    padding: 8px 12px !important;
}
ul.options li:hover, ul[role="listbox"] li:hover,
.options .item:hover, .options li:hover,
ul.options li[aria-selected="true"], ul[role="listbox"] li[aria-selected="true"],
.options .item.selected {
    background: #C5F542 !important;
    color: #000 !important;
}

/* ---------- image & file dropzones — visible dashed border ---------- */
.gr-image, [data-testid="image"], .image-container,
.file-preview, .gr-file, [data-testid="file"] {
    background: #141414 !important;
    border: 1px dashed #3a3a3a !important;
}

/* ---------- source-thumbnail gallery ---------- */
.gmi-source-gallery, .gr-gallery, [data-testid="gallery"], .gallery {
    background: #141414 !important;
    border: 1px solid #3a3a3a !important;
}
.gmi-source-gallery .thumbnail-item, .gr-gallery .thumbnail-item,
.gallery-item, .thumbnail-item {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
}
.gmi-source-gallery .thumbnail-item:hover {
    border-color: #C5F542 !important;
}
.gmi-source-gallery img, .gr-gallery img { background: #0a0a0a !important; }

/* gr.File post-upload list (filename rows, size, download/remove icons) */
.file-preview, .file-preview *,
.gr-file, .gr-file *,
[data-testid="file"], [data-testid="file"] *,
.file-list, .file-list *,
.upload-list, .upload-list *,
.file-row, .file-item, .upload-item,
.file-name, .filename, .file-size, .size, .filesize,
table.file-preview, table.file-preview td, table.file-preview tr {
    background-color: #141414 !important;
    color: #e8e8e8 !important;
}
.file-preview tr, .gr-file tr, [data-testid="file"] tr {
    border-bottom: 1px solid #2a2a2a !important;
}
/* Icons inside file rows: lime accent */
.file-preview svg, .gr-file svg, [data-testid="file"] svg,
.file-preview button, .gr-file button[class*="action"],
.file-preview a, .gr-file a {
    color: #C5F542 !important;
    fill: #C5F542 !important;
}
/* Hover state on file rows */
.file-preview tr:hover, .gr-file tr:hover, [data-testid="file"] tr:hover {
    background-color: #1a1a1a !important;
}

/* ---------- loading / status / placeholder — force dark + readable ---------- */
.empty, .loading, .loading-status, .progress, .pending,
.upload-container, .download, .holder, .placeholder,
.image-frame, .image-button, .image-button-row,
.status-tracker, .progress-level,
.loading-wrap, .status-wait, .progress-bar-wrap,
[class*="loading"], [class*="empty"], [class*="placeholder"],
[class*="skeleton"] {
    background: #141414 !important;
    color: #ffffff !important;
    border-color: #3a3a3a !important;
}

/* Built-in Gradio progress text — LIGHT, never dark (text sits on dark bg). */
.progress-text, .progress-level-text, .progress-value,
.progress .label, .progress-bar .label,
.gr-progress .progress-text, [class*="progress"] .progress-text,
[class*="progress"] [class*="text"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    background: transparent !important;
}
.progress-bar, [class*="progress"] > div.bar,
[class*="progress"] [class*="bar"] {
    background: #C5F542 !important;
}

/* ---------- custom output progress bar (.gmi-progress) ---------- */
.gmi-progress {
    background: #141414 !important;
    border: 1px solid #3a3a3a !important;
    padding: 20px !important;
    margin: 12px 0 20px 0 !important;
    font-family: 'Space Mono', monospace !important;
    color: #ffffff !important;
}
.gmi-progress * { color: #ffffff !important; }
.gmi-progress .pct { color: #C5F542 !important; }
.gmi-progress-label {
    color: #ffffff !important;
    font-size: 12px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    margin-bottom: 14px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: baseline !important;
    font-weight: 700 !important;
}
.gmi-progress-label > span:first-child { color: #ffffff !important; }
.gmi-progress-label .pct {
    color: #C5F542 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}
.gmi-progress-track {
    background: #0a0a0a !important;
    height: 10px !important;
    overflow: hidden !important;
    border: 1px solid #2a2a2a !important;
    position: relative !important;
}
.gmi-progress-fill {
    background: #C5F542 !important;
    height: 100% !important;
    transition: width 350ms ease !important;
    box-shadow: 0 0 12px rgba(197, 245, 66, 0.45) !important;
}
.gmi-progress-indeterminate {
    position: absolute !important; inset: 0 !important;
    background: linear-gradient(90deg, transparent, rgba(197, 245, 66, 0.18), transparent) !important;
    animation: gmi-slide 1.4s linear infinite !important;
}
@keyframes gmi-slide {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* ---------- labels (RESOLUTION, ASPECT, SCENE ANALYSIS, etc.) ---------- */
label > span, .label > span, .gr-form > label, .gr-input-label,
.gr-textbox label, .gr-dropdown label, .gr-image label, .gr-file label {
    color: #c8c8c8 !important;
    font-size: 11px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
}

/* ---------- buttons ---------- */
button.lg, button.primary, .primary > button, button[variant="primary"] {
    background: #C5F542 !important;
    color: #000 !important;
    border: 1px solid #C5F542 !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    padding: 18px 24px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    transition: background 120ms, color 120ms !important;
}
button.lg:hover, button.primary:hover, .primary > button:hover {
    background: #ffffff !important;
    border-color: #ffffff !important;
}
button.secondary, .secondary > button, button[variant="secondary"] {
    background: transparent !important;
    color: #C5F542 !important;
    border: 1px solid #C5F542 !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    padding: 18px 24px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    transition: background 120ms, color 120ms !important;
}
button.secondary:hover, .secondary > button:hover, button[variant="secondary"]:hover {
    background: #C5F542 !important;
    color: #000 !important;
}
button:disabled, button[disabled] {
    opacity: 0.4 !important;
    cursor: not-allowed !important;
}

/* ---------- accordion (MODEL REASONING) ---------- */
.gr-accordion, details {
    background: transparent !important;
    border: none !important;
    border-top: 1px solid #3a3a3a !important;
    margin-top: 24px !important;
    padding-top: 4px !important;
}
/* Covers BOTH <summary> (legacy) and <button class="label-wrap"> (Gradio 4.x) */
.gr-accordion summary, details summary,
.gr-accordion .label-wrap, .gr-accordion button.label-wrap,
.gr-accordion > button, .gr-accordion > .label-wrap,
button.label-wrap, .label-wrap {
    color: #ffffff !important;
    background: transparent !important;
    text-transform: uppercase !important;
    letter-spacing: 0.22em !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 16px 4px !important;
    cursor: pointer !important;
    list-style: none !important;
    border: none !important;
    transition: color 120ms;
    text-align: left !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}
/* Make sure the inner <span> inherits the white color */
.gr-accordion summary *, details summary *,
.gr-accordion .label-wrap *, .gr-accordion button.label-wrap *,
.label-wrap *, button.label-wrap * {
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 0.22em !important;
}
.gr-accordion summary::marker, details summary::marker,
.gr-accordion summary::-webkit-details-marker, details summary::-webkit-details-marker {
    display: none !important;
}
.gr-accordion summary::before, details summary::before,
.gr-accordion .label-wrap::before, .gr-accordion button.label-wrap::before {
    content: "▸";
    color: #C5F542 !important;
    margin-right: 10px;
    display: inline-block;
    transition: transform 120ms;
    font-weight: 700;
}
.gr-accordion[open] summary::before, details[open] summary::before,
.gr-accordion.open .label-wrap::before, .gr-accordion[data-open="true"] .label-wrap::before {
    transform: rotate(90deg);
}
.gr-accordion summary:hover, details summary:hover,
.gr-accordion .label-wrap:hover, .gr-accordion button.label-wrap:hover,
.gr-accordion summary:hover *, .gr-accordion .label-wrap:hover * {
    color: #C5F542 !important;
}

/* ---------- error panel ---------- */
.gmi-error {
    background: #1a0a0a !important;
    border-left: 3px solid #ff4444 !important;
    padding: 14px 18px !important;
    color: #ffbcbc !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.55 !important;
    margin: 16px 0 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}
.gmi-error .title {
    color: #ff5555 !important;
    font-size: 11px !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    margin-bottom: 8px !important;
    display: block;
}
.gmi-error details { border: none !important; margin-top: 10px !important; padding: 0 !important; }
.gmi-error details summary {
    color: #ff8888 !important;
    font-size: 10px !important;
    letter-spacing: 0.15em !important;
    padding: 4px 0 !important;
}
.gmi-error pre {
    margin: 8px 0 0 0 !important;
    padding: 10px !important;
    background: #0d0505 !important;
    color: #d88 !important;
    font-size: 11px !important;
    overflow-x: auto;
}

/* ---------- footer ---------- */
.gmi-footer {
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid #3a3a3a;
    color: #777;
    font-size: 11px;
    letter-spacing: 0.12em;
    font-family: 'Space Mono', monospace;
}
.gmi-footer .k { color: #aaa; }
.gmi-footer .v { color: #C5F542; }
.gmi-footer .sep { color: #3a3a3a; padding: 0 12px; }

footer { display: none !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.lime,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_none,
    font=[gr.themes.GoogleFont("Space Mono"), "ui-monospace", "monospace"],
)


with gr.Blocks(title="PICTURE_COMPOSE · GMI", theme=THEME, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_classes=["gmi-hero"]):
        gr.HTML(
            '<div class="gmi-badge"><span class="dot">▮</span>POWERED BY GMI CLOUD</div>'
            '<h1>PICTURE<br/><span class="accent">COMPOSE</span></h1>'
            '<p class="subtitle">'
            'Place one image into another, end-to-end on GMI Cloud. '
            'The destination scene is preserved precisely — only the inserted object '
            'is added, with lighting, shadows, perspective and color grade matched '
            'to make it indistinguishable from an in-camera photograph.'
            '</p>'
        )

    gr.HTML('<div class="gmi-section-label">00 · API KEY</div>')
    api_key_in = gr.Textbox(
        placeholder="Paste your GMI API key (or set the GMI_API_KEY env var). Get one at console.gmicloud.ai → API Keys.",
        type="password",
        lines=1,
        show_label=False,
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="gmi-section-label">01 · DESTINATION (REQUIRED)</div>')
            destination_in = gr.Image(
                label="The scene to edit",
                type="pil", height=300, show_label=False,
            )

            gr.HTML(f'<div class="gmi-section-label">02 · SOURCE OBJECTS (OPTIONAL · UP TO {MAX_SOURCES})</div>')
            source_in = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                label=f"Drop up to {MAX_SOURCES} object images",
                show_label=False,
                height=180,
            )
            # Thumbnail strip of uploaded source objects — populated reactively
            # from source_in via .change() below.
            source_gallery = gr.Gallery(
                show_label=False,
                columns=4,
                rows=1,
                height=140,
                object_fit="contain",
                preview=False,
                visible=False,
                elem_classes=["gmi-source-gallery"],
            )

            gr.HTML('<div class="gmi-section-label">03 · INSTRUCTION (OPTIONAL)</div>')
            instruction_in = gr.Textbox(
                placeholder=(
                    "Optional. If you provided source image(s), say WHERE to put them "
                    "(e.g. 'left corner of the balcony, facing the view'). "
                    "If you did not, say WHAT to add "
                    "(e.g. 'a tan wicker chair with a linen cushion')."
                ),
                lines=3, show_label=False,
            )

            with gr.Row():
                image_size_in = gr.Dropdown(
                    choices=["512", "1K", "2K", "4K"], value="1K", label="RESOLUTION",
                )
                aspect_in = gr.Dropdown(
                    choices=["auto"] + ALLOWED_ASPECTS, value="auto", label="ASPECT",
                )

            with gr.Row():
                run_btn = gr.Button("COMPOSE  →", variant="primary", size="lg")
                recompose_btn = gr.Button("RECOMPOSE  ⟳", variant="secondary", size="lg")

        with gr.Column(scale=1):
            gr.HTML('<div class="gmi-section-label">OUTPUT</div>')
            # Custom in-column progress bar — hidden when idle, populated by
            # the pipeline generators on each yield.
            status_out = gr.HTML(visible=False)
            result_out = gr.Image(
                label="Composite", type="pil", height=420, show_label=False,
            )
            with gr.Accordion("MODEL REASONING", open=False):
                scene_out = gr.Textbox(label="SCENE ANALYSIS", lines=5)
                object_out = gr.Textbox(label="OBJECT ANALYSIS (PER SOURCE)", lines=8)
                prompt_out = gr.Textbox(label="COMPOSITION PROMPT SENT TO IMAGE MODEL", lines=8)
                debug_out = gr.Textbox(label="JOB INFO", lines=4)

    # Inline error display — hidden until something fails.
    error_out = gr.HTML(visible=False)

    # Cache for RECOMPOSE — holds the prior analysis result + uploaded ref URLs so
    # the second/third/Nth click skips the LLM analysis and the image uploads.
    cache_state = gr.State(value=None)

    run_btn.click(
        run_pipeline,
        inputs=[api_key_in, destination_in, source_in, instruction_in,
                image_size_in, aspect_in, cache_state],
        outputs=[result_out, scene_out, object_out, prompt_out, debug_out,
                 error_out, status_out, cache_state],
    )
    recompose_btn.click(
        recompose_pipeline,
        inputs=[api_key_in, image_size_in, aspect_in, cache_state],
        outputs=[result_out, scene_out, object_out, prompt_out, debug_out,
                 error_out, status_out, cache_state],
    )

    # Live thumbnail strip: when files are added/removed, refresh the gallery.
    def _refresh_source_gallery(files):
        if not files:
            return gr.update(visible=False, value=[])
        paths = []
        for entry in files:
            p = entry if isinstance(entry, str) else getattr(entry, "name", None) or str(entry)
            paths.append(p)
        return gr.update(visible=True, value=paths)

    source_in.change(_refresh_source_gallery, inputs=[source_in], outputs=[source_gallery])

    gr.HTML(
        '<div class="gmi-footer">'
        '<span class="k">VISION_LLM:</span> <span class="v">google/gemini-3.1-pro-preview</span>'
        '<span class="sep">|</span>'
        '<span class="k">IMAGE_MODEL:</span> <span class="v">gemini-3.1-flash-image-preview</span>'
        '<span class="sep">|</span>'
        '<span class="k">PROVIDER:</span> <span class="v">gmicloud.ai</span>'
        '</div>'
    )


if __name__ == "__main__":
    # When running inside an AgentBox container, GMI routes external traffic
    # to whatever port the container listens on. Bind to 0.0.0.0 (not the
    # default 127.0.0.1) so the listener accepts connections from outside the
    # container, and respect the PORT env var if AgentBox sets one.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
