import os
import time
import uuid
import json
import logging
from typing import Optional, Dict, Any, List

import requests
from requests.exceptions import (
    ReadTimeout,
    ConnectTimeout,
    Timeout,
    HTTPError,
    RequestException,
)
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gateway")

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
LLM1_URL = os.getenv("LLM1_URL", "http://llm1:11434/api/generate")
LLM2_URL = os.getenv("LLM2_URL", "http://llm2:11434/api/generate")
NEWS_URL = os.getenv("NEWS_URL", "http://news:8080/news")

LLM1_MODEL = os.getenv("LLM1_MODEL", "llama3.2:3b")
LLM2_MODEL = os.getenv("LLM2_MODEL", "llama3.1:8b")

REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "300"))
NEWS_TIMEOUT_SEC = int(os.getenv("NEWS_TIMEOUT_SEC", "15"))

# Keep generous; if you want effectively no clamp, set to e.g. 200000
MAX_NEWS_CHARS = int(os.getenv("MAX_NEWS_CHARS", "20000"))
MAX_PROFILE_CHARS = int(os.getenv("MAX_PROFILE_CHARS", "4000"))

RETRY_COUNT = int(os.getenv("RETRY_COUNT", "2"))
RETRY_BACKOFF_SEC = float(os.getenv("RETRY_BACKOFF_SEC", "0.8"))

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="Gateway", version="1.0.0")

# -------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------
class ProcessRequest(BaseModel):
    user_description: str = Field(..., min_length=1)

    # kept for backward compatibility; gateway ignores it and always uses NEWS_URL
    news_blob: Optional[str] = None

    # forwarded to Ollama
    options: Optional[Dict[str, Any]] = None


class ProcessResponse(BaseModel):
    profile: str
    personalized_news: str
    meta: Dict[str, Any]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_rid(req: Request) -> str:
    return req.headers.get("x-request-id") or str(uuid.uuid4())[:8]


def clamp_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    s = s.strip()
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


# -------------------------------------------------------------------
# HTTP: retries + robust JSON parsing
# -------------------------------------------------------------------
def _classify_requests_exc(exc: Exception) -> str:
    if isinstance(exc, (ReadTimeout, Timeout)):
        return "read_timeout"
    if isinstance(exc, ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return f"http_error_{status if status is not None else 'unknown'}"
    if isinstance(exc, RequestException):
        return "request_exception"
    return exc.__class__.__name__


def safe_json(resp: requests.Response, *, rid: str, url: str) -> dict:
    try:
        return resp.json()
    except Exception as exc:
        body_preview = clamp_text(resp.text or "", 800)
        logger.error(
            "[%s] Non-JSON response from upstream url=%s status=%s body_preview=%s",
            rid,
            url,
            resp.status_code,
            body_preview,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Upstream returned invalid JSON",
                "url": url,
                "status_code": resp.status_code,
                "error_type": exc.__class__.__name__,
                "body_preview": body_preview,
            },
        )


def post_json_with_retry(url: str, payload: dict, *, timeout: int, rid: str) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            t0 = time.time()
            resp = requests.post(url, json=payload, timeout=timeout)
            dt = time.time() - t0
            logger.info("[%s] POST %s -> %s (%.2fs)", rid, url, resp.status_code, dt)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[%s] POST failed attempt=%s url=%s kind=%s err=%s",
                rid,
                attempt + 1,
                url,
                _classify_requests_exc(exc),
                repr(exc),
            )
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))

    raise HTTPException(
        status_code=502,
        detail={
            "message": f"Upstream POST failed: {url}",
            "error_type": _classify_requests_exc(last_exc) if last_exc else "unknown",
            "repr": repr(last_exc),
        },
    )


def get_json_with_retry(url: str, *, timeout: int, rid: str) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            t0 = time.time()
            resp = requests.get(url, timeout=timeout)
            dt = time.time() - t0
            logger.info("[%s] GET %s -> %s (%.2fs)", rid, url, resp.status_code, dt)
            resp.raise_for_status()
            return safe_json(resp, rid=rid, url=url)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[%s] GET failed attempt=%s url=%s kind=%s err=%s",
                rid,
                attempt + 1,
                url,
                _classify_requests_exc(exc),
                repr(exc),
            )
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))

    raise HTTPException(
        status_code=502,
        detail={
            "message": f"Upstream GET failed: {url}",
            "error_type": _classify_requests_exc(last_exc) if last_exc else "unknown",
            "repr": repr(last_exc),
        },
    )


# -------------------------------------------------------------------
# Ollama call
# -------------------------------------------------------------------
def call_ollama_generate(
    *,
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    options: Optional[Dict[str, Any]],
    timeout: int,
    rid: str,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    resp = post_json_with_retry(url, payload, timeout=timeout, rid=rid)
    data = safe_json(resp, rid=rid, url=url)
    out = data.get("response", "") or ""
    if not out:
        logger.warning("[%s] Ollama returned empty response model=%s url=%s", rid, model, url)
    return out


# -------------------------------------------------------------------
# LLM1: extract preferences as strict JSON (stable)
# -------------------------------------------------------------------
LLM1_EXTRACT_SYSTEM = """
Extract user preferences from the user description.

Return ONLY valid JSON with exactly these keys:
{
  "likes": ["..."],
  "avoids": ["..."],
  "tone": "neutral|friendly|formal|casual",
  "detail_level": "low|medium|high"
}

Rules:
- Use English.
- likes/avoids must be arrays of short lowercase phrases.
- If unknown, use "unspecified" for tone/detail_level and empty arrays for likes/avoids.
- Output JSON only. No markdown. No explanation.
""".strip()


def _parse_prefs_json(raw: str, rid: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    try:
        prefs = json.loads(raw)
    except Exception as exc:
        logger.warning("[%s] LLM1 JSON parse failed err=%s raw_preview=%s", rid, repr(exc), clamp_text(raw, 400))
        prefs = {}

    likes = prefs.get("likes") or []
    avoids = prefs.get("avoids") or []
    tone = prefs.get("tone") or "unspecified"
    detail = prefs.get("detail_level") or "unspecified"

    if not isinstance(likes, list):
        likes = []
    if not isinstance(avoids, list):
        avoids = []
    if not isinstance(tone, str):
        tone = "unspecified"
    if not isinstance(detail, str):
        detail = "unspecified"

    likes = [str(x).strip().lower() for x in likes if str(x).strip()]
    avoids = [str(x).strip().lower() for x in avoids if str(x).strip()]

    return {
        "likes": likes,
        "avoids": avoids,
        "tone": tone.strip(),
        "detail_level": detail.strip(),
        "raw": raw,
    }


def extract_preferences(user_desc: str, *, rid: str, llm1_options: Dict[str, Any]) -> Dict[str, Any]:
    raw = call_ollama_generate(
        url=LLM1_URL,
        model=LLM1_MODEL,
        system_prompt=LLM1_EXTRACT_SYSTEM,
        user_prompt=user_desc,
        options=llm1_options,
        timeout=REQUEST_TIMEOUT_SEC,
        rid=rid,
    )
    raw = clamp_text(raw, MAX_PROFILE_CHARS)
    return _parse_prefs_json(raw, rid=rid)


# -------------------------------------------------------------------
# LLM2: deterministic system prompt for long TEXT (no bullets)
# -------------------------------------------------------------------
def build_llm2_system_prompt(prefs: Dict[str, Any]) -> str:
    likes = prefs.get("likes") or []
    avoids = prefs.get("avoids") or []
    tone = prefs.get("tone") or "unspecified"
    detail = prefs.get("detail_level") or "unspecified"

    likes_str = ", ".join(likes) if likes else "unspecified"
    avoids_str = ", ".join(avoids) if avoids else "unspecified"

    return f"""
You are a news personalization engine.

Input: a plain-text list of news items from an internal feed. Each item may include Category, Title, Source, URL, Summary, and Content.

Goal: Select only the items that match the user's interests and write long-form narrative coverage for each selected item, as continuous text (no bullet lists).

User preferences:
- Likes: {likes_str}
- Avoids: {avoids_str}
- Tone: {tone}
- Detail level: {detail}

Hard rules:
- Use ONLY the provided news input. Do not browse. Do not add facts. Do not speculate.
- Select items strictly by semantic relevance to Likes.
- Exclude any item related to Avoids, including synonyms and closely related terms.
- Do NOT mention Avoids or excluded topics in the output.
- Do NOT copy/paste the input text verbatim; you may restate facts, but do not reproduce large chunks literally.
- Do NOT include raw field labels like "Category:", "Source:", "Summary:", "Content:" in your output.
- Do NOT use bullet lists or numbered lists.
- Do NOT use ellipses "..." anywhere.
- If no items qualify, output exactly: No relevant news is available for your preferences at this time.

Output format (MANDATORY):
- Write 1–2 sections, each corresponding to one selected article.
- For EACH selected article:
  - Start with a single line containing the title only.
  - Then write 4–8 short paragraphs of continuous prose describing the article, its main points, and its implications, in a neutral tone.
  - If a URL exists, mention it once at the end of the last paragraph in the form: URL: <url>.
- Keep the text compact and informative: avoid repetition and filler sentences.
""".strip()


# -------------------------------------------------------------------
# News blob builder (service -> LLM input)
# -------------------------------------------------------------------
def build_news_blob_from_payload(news_payload: dict) -> str:
    items = news_payload.get("items") or []
    blocks: List[str] = []

    for it in items:
        cat = (it.get("category") or "").upper()
        title = it.get("title") or ""
        source = it.get("source") or ""
        url = it.get("url") or ""
        description = it.get("description") or ""
        content = it.get("content") or ""
        err = it.get("error")

        blocks.append(f"[{cat}] {title}".strip())
        if source:
            blocks.append(f"Source: {source}")
        if url:
            blocks.append(f"URL: {url}")
        if err:
            blocks.append(f"Error: {err}")

        if description:
            blocks.append(f"Summary: {description}")

        if content:
            blocks.append("Content:")
            blocks.append(content)

        blocks.append("")

    blob = "\n".join(blocks).strip()
    return clamp_text(blob, MAX_NEWS_CHARS)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    status = {"ok": True, "deps": {}}
    for name, url in [
        ("news", NEWS_URL.replace("/news", "/health")),
        ("llm1", LLM1_URL.replace("/api/generate", "/api/tags")),
        ("llm2", LLM2_URL.replace("/api/generate", "/api/tags")),
    ]:
        try:
            r = requests.get(url, timeout=5)
            status["deps"][name] = {"ok": r.status_code < 400, "status": r.status_code}
        except Exception as e:
            status["deps"][name] = {"ok": False, "error": repr(e)}
            status["ok"] = False
    return status


@app.get("/debug/news")
def debug_news(request: Request):
    rid = get_rid(request)
    payload = get_json_with_retry(NEWS_URL, timeout=NEWS_TIMEOUT_SEC, rid=rid)
    return payload


@app.get("/debug/llm2")
def debug_llm2(request: Request):
    rid = get_rid(request)
    payload = {
        "model": LLM2_MODEL,
        "system": "You are a health-check endpoint. Reply with: LLM2 OK",
        "prompt": "Say: LLM2 OK",
        "stream": False,
        "options": {"num_predict": 32, "temperature": 0.0, "top_p": 0.9},
    }
    resp = post_json_with_retry(LLM2_URL, payload, timeout=min(REQUEST_TIMEOUT_SEC, 60), rid=rid)
    data = safe_json(resp, rid=rid, url=LLM2_URL)
    return {
        "ok": True,
        "rid": rid,
        "model": LLM2_MODEL,
        "response_preview": clamp_text(data.get("response", ""), 200),
    }


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest, request: Request) -> ProcessResponse:
    rid = get_rid(request)

    user_desc = req.user_description.strip()
    if not user_desc:
        raise HTTPException(status_code=400, detail="user_description is empty")

    logger.info("[%s] === /process called ===", rid)
    logger.info("[%s] user_description=%s", rid, clamp_text(user_desc, 500))

    base_options = req.options or {"temperature": 0.2, "top_p": 0.9, "num_predict": 4500}

    # LLM1: deterministic extraction
    llm1_options = dict(base_options)
    llm1_options["temperature"] = 0.0
    llm1_options["num_predict"] = min(220, int(llm1_options.get("num_predict", 220)))

    # LLM2: very large output budget to support 40–70 lines per item
    llm2_options = dict(base_options)
    llm2_options["num_predict"] = max(int(llm2_options.get("num_predict", 4500)), 4500)

    # Optional but recommended for large input + large output (remove if your model errors)
    llm2_options.setdefault("num_ctx", 8192)

    # 1) LLM1 -> JSON preferences
    prefs = extract_preferences(user_desc, rid=rid, llm1_options=llm1_options)
    llm2_system = build_llm2_system_prompt(prefs)
    profile_debug = clamp_text(prefs.get("raw", ""), MAX_PROFILE_CHARS)

    # 2) ALWAYS fetch from news service
    news_source = "service"
    news_payload = get_json_with_retry(NEWS_URL, timeout=NEWS_TIMEOUT_SEC, rid=rid)
    news_blob = build_news_blob_from_payload(news_payload)

    logger.info("[%s] news_source=%s news_blob_len=%s", rid, news_source, len(news_blob))

    # 3) LLM2 generate
    llm2_out = call_ollama_generate(
        url=LLM2_URL,
        model=LLM2_MODEL,
        system_prompt=llm2_system,
        user_prompt=news_blob,
        options=llm2_options,
        timeout=REQUEST_TIMEOUT_SEC,
        rid=rid,
    )

    logger.info("[%s] === LLM2 FINAL RESPONSE (len=%s) ===", rid, len(llm2_out))

    return ProcessResponse(
        profile=profile_debug,
        personalized_news=llm2_out,
        meta={
            "rid": rid,
            "news_source": news_source,
            "llm1_model": LLM1_MODEL,
            "llm2_model": LLM2_MODEL,
            "preferences": {
                "likes": prefs.get("likes"),
                "avoids": prefs.get("avoids"),
                "tone": prefs.get("tone"),
                "detail_level": prefs.get("detail_level"),
            },
            "options_llm1": llm1_options,
            "options_llm2": llm2_options,
        },
    )
