import os
import json
import time
import uuid
import logging
from typing import Optional, Dict, Any, List, Tuple

import requests
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

REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "120"))
NEWS_TIMEOUT_SEC = int(os.getenv("NEWS_TIMEOUT_SEC", "15"))

MAX_NEWS_CHARS = int(os.getenv("MAX_NEWS_CHARS", "6000"))  # protect LLM2 prompt
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
            logger.warning("[%s] POST failed attempt=%s url=%s err=%s", rid, attempt + 1, url, repr(exc))
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_BACKOFF_SEC * (2 ** attempt))
    raise HTTPException(status_code=502, detail=f"Upstream POST failed: {url}. Last error: {last_exc}")


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
    # Make it deterministic and “operational”, not fluffy.
    return (
        "Task: Generate a SYSTEM PROMPT for another LLM, based on the user's description.\n"
        "Rules:\n"
        "- Output ONLY the system prompt content. No titles, no explanations, no quotes.\n"
        "- Use Romanian.\n"
        "- Keep it concise and operational.\n"
        "- Include sections exactly in this format:\n"
        "USER_PROFILE:\n"
        "- likes: ...\n"
        "- avoids: ...\n"
        "- tone: ...\n"
        "- detail_level: ...\n"
        "OUTPUT_FORMAT:\n"
        "- ...\n"
        "CONSTRAINTS:\n"
        "- ...\n"
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


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest, request: Request) -> ProcessResponse:
    rid = get_rid(request)

    user_desc = req.user_description.strip()
    if not user_desc:
        raise HTTPException(status_code=400, detail="user_description is empty")

    logger.info("[%s] === /process called ===", rid)
    logger.info("[%s] user_description=%s", rid, clamp_text(user_desc, 500))

    # options for ollama (safe defaults)
    ollama_options = req.options or {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_predict": 300,
    }

    # ----------------------------------------------------------------
    # 1) LLM1: generate profile SYS_PROMPT
    # ----------------------------------------------------------------
    llm1_sys = llm1_profile_system_instruction()
    llm1_out = call_ollama_generate(
        url=LLM1_URL,
        model=LLM1_MODEL,
        system_prompt=llm1_sys,
        user_prompt=user_desc,
        options=ollama_options,
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
        news_blob = clamp_text(req.news_blob.strip(), MAX_NEWS_CHARS)
    else:
        news_source = "service"
        news_payload = get_json_with_retry(NEWS_URL, timeout=NEWS_TIMEOUT_SEC, rid=rid)
        news_blob = build_news_blob_from_payload(news_payload)

    logger.info("[%s] news_source=%s news_blob_len=%s preview=%s", rid, news_source, len(news_blob), clamp_text(news_blob, 600))

    # ----------------------------------------------------------------
    # 3) LLM2: personalize on the news
    # ----------------------------------------------------------------
    llm2_out = call_ollama_generate(
        url=LLM2_URL,
        model=LLM2_MODEL,
        system_prompt=llm1_out,   # profile as SYS_PROMPT
        user_prompt=news_blob,    # only the news_blob as prompt
        options=ollama_options,
        timeout=REQUEST_TIMEOUT_SEC,
        rid=rid,
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
            "options": ollama_options,
        },
    )
