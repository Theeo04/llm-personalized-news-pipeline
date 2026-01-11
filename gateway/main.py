import os
import json
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple

import requests
from requests import ReadTimeout, ConnectTimeout, HTTPError
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
LLM2_MODEL = os.getenv("LLM2_MODEL", "llama3.2:3b")

# Increase default request timeout to 5 minutes
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "300"))
NEWS_TIMEOUT_SEC = int(os.getenv("NEWS_TIMEOUT_SEC", "15"))

MAX_NEWS_CHARS = int(os.getenv("MAX_NEWS_CHARS", "6000"))  # protect LLM2 prompt
MAX_LLM2_NEWS_CHARS = int(os.getenv("MAX_LLM2_NEWS_CHARS", "3500"))
MAX_PROFILE_CHARS = int(os.getenv("MAX_PROFILE_CHARS", "2000"))

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
    # user's profile description (typed in UI)
    user_description: str = Field(..., min_length=1)

    # optional: UI can pass news directly; if None -> gateway pulls from NEWS service
    news_blob: Optional[str] = None

    # optional knobs, forwarded to Ollama
    # example: {"temperature": 0.2, "top_p": 0.9, "num_predict": 250}
    options: Optional[Dict[str, Any]] = None


class ProcessResponse(BaseModel):
    profile: str
    personalized_news: str
    meta: Dict[str, Any]


# -------------------------------------------------------------------
# Helpers (RID / logging)
# -------------------------------------------------------------------
def get_rid(req: Request) -> str:
    return req.headers.get("x-request-id") or str(uuid.uuid4())[:8]


def clamp_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    s = s.strip()
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


# -------------------------------------------------------------------
# HTTP helpers with retry
# -------------------------------------------------------------------
def _classify_requests_exc(exc: Exception) -> str:
    if isinstance(exc, ReadTimeout):
        return "read_timeout"
    if isinstance(exc, ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, HTTPError):
        return f"http_error_{exc.response.status_code if exc.response is not None else 'unknown'}"
    return exc.__class__.__name__


def post_json_with_retry(url: str, payload: dict, timeout: int, rid: str) -> requests.Response:
    last_exc = None
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
            kind = _classify_requests_exc(exc)
            logger.warning(
                "[%s] POST failed attempt=%s url=%s kind=%s err=%s",
                rid,
                attempt + 1,
                url,
                kind,
                repr(exc),
            )
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
    # Propagate a detailed error to caller
    raise HTTPException(
        status_code=502,
        detail={
            "message": f"Upstream POST failed: {url}",
            "error_type": _classify_requests_exc(last_exc) if last_exc else "unknown",
            "repr": repr(last_exc),
        },
    )


def get_json_with_retry(url: str, timeout: int, rid: str) -> dict:
    last_exc = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            t0 = time.time()
            resp = requests.get(url, timeout=timeout)
            dt = time.time() - t0
            logger.info("[%s] GET %s -> %s (%.2fs)", rid, url, resp.status_code, dt)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            logger.warning("[%s] GET failed attempt=%s url=%s err=%s", rid, attempt + 1, url, repr(exc))
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
    raise HTTPException(status_code=502, detail=f"Upstream GET failed: {url}. Last error: {last_exc}")


# -------------------------------------------------------------------
# Ollama calls
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
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options

    resp = post_json_with_retry(url, payload, timeout=timeout, rid=rid)
    data = resp.json()
    out = data.get("response", "")
    if not out:
        logger.warning("[%s] Ollama returned empty response model=%s url=%s", rid, model, url)
    return out


# -------------------------------------------------------------------
# Prompt engineering
# -------------------------------------------------------------------
def llm1_profile_system_instruction() -> str:
    """
    LLM1 generates a SYSTEM PROMPT for LLM2.

    Goal:
    - Extract from the user's description:
      - what topics they LIKE (include),
      - what topics they want to AVOID (exclude),
      - optional tone and detail level.
    - Build a compact, operational system prompt that LLM2 will use
      to rewrite news items coming from the news service.

    Hard requirements:
    - LLM2 MUST ONLY use information present in the news text it receives
      (no external knowledge, no web search, no invented facts).
    - LLM2 MUST filter news strictly based on likes / avoids.
    """
    return (
        "You are configuring another LLM (LLM2).\n"
        "Input: a free-text description from the user.\n"
        "Output: ONLY the final SYSTEM PROMPT text for LLM2, no explanations.\n"
        "\n"
        "GENERAL RULES:\n"
        "- Use English.\n"
        "- Be short, clear, and operational.\n"
        "- Follow EXACTLY the structure and labels below.\n"
        "- Every list item MUST start with '- ' (dash + space).\n"
        "- If a field is missing, use 'unspecified'.\n"
        "\n"
        "STRUCTURE:\n"
        "USER_PROFILE:\n"
        "- likes: comma-separated topics the user is interested in (derived from the description; if none, 'unspecified').\n"
        "- avoids: comma-separated topics or themes the user does NOT want (if none, 'unspecified').\n"
        "- tone: short description like 'neutral', 'enthusiastic', 'formal', etc. (or 'unspecified').\n"
        "- detail_level: short description like 'high-level', 'medium', 'detailed' (or 'unspecified').\n"
        "\n"
        "OUTPUT_FORMAT:\n"
        "- text\n"
        "\n"
        "CONSTRAINTS:\n"
        "- Include ONLY news items whose topics clearly match likes; if an item does not match, omit it completely.\n"
        "- Completely exclude any news item that touches avoided topics.\n"
        "- Never mention avoided topics explicitly or implicitly.\n"
        "- Use ONLY the information present in the provided news text; do NOT add external facts, background, or speculation.\n"
        "- If some detail is missing from the news text, do NOT invent it.\n"
        "- The final response MUST be at least 1000 characters long (including spaces).\n"
        "- If no items remain after filtering, output EXACTLY: \"No relevant news is available for your preferences at this time.\" and nothing else.\n"
    )


def build_news_blob_from_payload(news_payload: dict) -> str:
    """
    Convert JSON payload to a compact, LLM-friendly string.
    Avoid full HTML, keep only fields that matter.
    """
    items = news_payload.get("items") or []
    blocks: List[str] = []
    for it in items:
        cat = (it.get("category") or "").upper()
        title = it.get("title") or ""
        source = it.get("source") or ""
        url = it.get("url") or ""
        desc = it.get("description") or it.get("content") or ""
        err = it.get("error")

        blocks.append(f"[{cat}] {title}".strip())
        if source:
            blocks.append(f"Source: {source}")
        if url:
            blocks.append(f"URL: {url}")
        if err:
            blocks.append(f"Error: {err}")
        if desc:
            blocks.append(f"Summary: {desc}")
        blocks.append("")  # spacing

    blob = "\n".join(blocks).strip()
    return clamp_text(blob, MAX_NEWS_CHARS)


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    # Quick reachability check (best-effort).
    # We do not hard-fail if one dependency is down; return status map.
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
    """
    Quick sanity check for LLM2: send a tiny prompt to verify model health.
    Does not involve news or LLM1.
    """
    rid = get_rid(request)
    logger.info("[%s] /debug/llm2 called", rid)
    try:
        payload = {
            "model": LLM2_MODEL,
            "system": "You are a health-check endpoint. Reply with a short OK message.",
            "prompt": "Say: LLM2 OK",
            "stream": False,
            "options": {"num_predict": 32, "temperature": 0.0, "top_p": 0.9},
        }
        # Cap health-check timeout to max 60s even though main timeout is 300s
        resp = post_json_with_retry(LLM2_URL, payload, timeout=min(REQUEST_TIMEOUT_SEC, 60), rid=rid)
        data = resp.json()
        return {
            "ok": True,
            "rid": rid,
            "model": LLM2_MODEL,
            "response_preview": clamp_text(data.get("response", ""), 200),
        }
    except HTTPException as http_exc:
        # propagate our structured error
        raise http_exc
    except Exception as exc:
        logger.exception("[%s] /debug/llm2 failed: %s", rid, repr(exc))
        raise HTTPException(
            status_code=502,
            detail={"message": "LLM2 debug call failed", "error_type": exc.__class__.__name__, "repr": repr(exc)},
        )


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest, request: Request) -> ProcessResponse:
    rid = get_rid(request)

    user_desc = req.user_description.strip()
    if not user_desc:
        raise HTTPException(status_code=400, detail="user_description is empty")

    logger.info("[%s] === /process called ===", rid)
    logger.info("[%s] user_description=%s", rid, clamp_text(user_desc, 500))

    # Base options from request or defaults
    base_options = req.options or {
        "temperature": 0.8,
        "top_p": 0.9,
        "num_predict": 300,
    }

    # Derive options for each LLM separately (mainly to keep LLM2 cheaper/faster)
    llm1_options = dict(base_options)
    llm2_options = dict(base_options)
    # If user didn't explicitly request num_predict, shrink for LLM2
    if "num_predict" not in (req.options or {}):
        llm2_options["num_predict"] = min(220, llm2_options.get("num_predict", 300))

    # ----------------------------------------------------------------
    # 1) LLM1: generate profile SYS_PROMPT
    # ----------------------------------------------------------------
    llm1_sys = llm1_profile_system_instruction()
    llm1_out = call_ollama_generate(
        url=LLM1_URL,
        model=LLM1_MODEL,
        system_prompt=llm1_sys,
        user_prompt=user_desc,
        options=llm1_options,
        timeout=REQUEST_TIMEOUT_SEC,
        rid=rid,
    )
    llm1_out = clamp_text(llm1_out, MAX_PROFILE_CHARS)

    logger.info("[%s] === LLM1 SYS_PROMPT (clamped) ===\n%s", rid, llm1_out)

    # ----------------------------------------------------------------
    # 2) News: use req.news_blob or fetch cached from news service
    # ----------------------------------------------------------------
    news_source = "request"
    if req.news_blob and req.news_blob.strip():
        raw_news_blob = clamp_text(req.news_blob.strip(), MAX_NEWS_CHARS)
    else:
        news_source = "service"
        news_payload = get_json_with_retry(NEWS_URL, timeout=NEWS_TIMEOUT_SEC, rid=rid)
        raw_news_blob = build_news_blob_from_payload(news_payload)

    # Extra clamp for LLM2 specifically
    news_blob = clamp_text(raw_news_blob, MAX_LLM2_NEWS_CHARS)

    logger.info(
        "[%s] news_source=%s news_blob_len=%s preview=%s",
        rid,
        news_source,
        len(news_blob),
        clamp_text(news_blob, 600),
    )

    # ----------------------------------------------------------------
    # 3) LLM2: personalize on the news
    # ----------------------------------------------------------------
    try:
        llm2_out = call_ollama_generate(
            url=LLM2_URL,
            model=LLM2_MODEL,
            system_prompt=llm1_out,   # profile as SYS_PROMPT
            user_prompt=news_blob,    # only the news_blob as prompt
            options=llm2_options,
            timeout=REQUEST_TIMEOUT_SEC,
            rid=rid,
        )
    except HTTPException as http_exc:
        # Surface a clear, structured failure to the UI
        raise HTTPException(
            status_code=http_exc.status_code,
            detail={
                "message": "LLM2 generation failed",
                "cause": http_exc.detail,
                "rid": rid,
            },
        )

    logger.info("[%s] === LLM2 FINAL RESPONSE ===\n%s", rid, llm2_out)

    return ProcessResponse(
        profile=llm1_out,
        personalized_news=llm2_out,
        meta={
            "rid": rid,
            "news_source": news_source,
            "llm1_model": LLM1_MODEL,
            "llm2_model": LLM2_MODEL,
            "options_llm1": llm1_options,
            "options_llm2": llm2_options,
        },
    )
