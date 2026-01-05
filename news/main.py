import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("news")

DATA_DIR = Path("/data").resolve()

PREDEFINED_ITEMS = [
    {
        "category": "it",
        "title": "Quantization makes small open-source LLMs faster on consumer GPUs",
        "source": "Tech Digest",
        "url": "https://example.com/it-quantization",
        "description": "New quantization recipes reduce VRAM usage while keeping quality competitive.",
        "content": "Quantization techniques continue to improve inference efficiency..."
    },
    {
        "category": "business",
        "title": "Startups shift from growth-at-all-costs to efficient execution",
        "source": "Business Weekly",
        "url": "https://example.com/business-efficiency",
        "description": "Investors reward profitability metrics and strong unit economics in 2026.",
        "content": "Founders focus on burn multiple, CAC/LTV, and sustainable go-to-market..."
    },
    {
        "category": "sports",
        "title": "Wearables and analytics reshape training plans in elite football",
        "source": "Sports Insights",
        "url": "https://example.com/sports-analytics",
        "description": "Clubs use load management and recovery signals to reduce injuries.",
        "content": "Performance teams increasingly combine GPS, HRV, and match data..."
    },
]

class NewsItem(BaseModel):
    category: str
    title: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None

class NewsPayload(BaseModel):
    timestamp: str
    items: List[NewsItem]
    meta: Dict[str, Any] = {}

app = FastAPI(title="News Service", version="1.0.0")

def _latest_path() -> Path:
    return DATA_DIR / "latest.json"

def _write_latest(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _latest_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Updated %s", _latest_path())

def _read_latest() -> Optional[dict]:
    p = _latest_path()
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def _build_payload() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    payload = NewsPayload(
        timestamp=now,
        items=[NewsItem(**x) for x in PREDEFINED_ITEMS],
        meta={"source": "predefined", "count": len(PREDEFINED_ITEMS)},
    ).model_dump()
    return payload

@app.on_event("startup")
def startup():
    logger.info("News service started. data_dir=%s", DATA_DIR)
    if _read_latest() is None:
        _write_latest(_build_payload())

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.post("/refresh", response_model=NewsPayload)
def refresh():
    payload = _build_payload()
    _write_latest(payload)
    return payload

@app.get("/news", response_model=NewsPayload)
def get_news():
    payload = _read_latest()
    if payload is None:
        raise HTTPException(status_code=404, detail="No cached news. Call /refresh.")
    logger.info("Serving cached news: ts=%s", payload.get("timestamp"))
    return payload
