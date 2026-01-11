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
        "description": (
            "Recent advances in quantization techniques are allowing smaller open-source "
            "large language models to achieve significantly better performance on consumer-grade GPUs."
        ),
        "content": (
            "Quantization techniques continue to play a crucial role in improving the inference efficiency "
            "of large language models, particularly in resource-constrained environments. By reducing the "
            "precision of model weights from 16-bit or 32-bit floating point representations to 8-bit, 4-bit, "
            "or mixed-precision formats, developers are able to dramatically lower VRAM consumption without "
            "a proportional loss in model accuracy.\n\n"
            "These improvements have enabled a new class of open-source LLMs to run effectively on consumer "
            "GPUs such as NVIDIA RTX-series cards, as well as on high-end laptops and small on-premise servers. "
            "For independent developers and startups, this reduces reliance on expensive cloud infrastructure "
            "and lowers the barrier to experimentation and deployment.\n\n"
            "Recent benchmarks show that quantized models can deliver near-parity performance on common NLP "
            "tasks such as summarization, question answering, and code generation, while achieving latency "
            "reductions of up to 40%. Tooling ecosystems around frameworks like GGUF and ONNX are also maturing, "
            "making it easier to integrate quantized models into production systems.\n\n"
            "As the open-source AI ecosystem continues to evolve, quantization is expected to remain a key "
            "optimization strategy, particularly for edge deployments, private inference, and cost-sensitive "
            "applications."
        ),
    },
    {
        "category": "business",
        "title": "Startups shift from aggressive growth to efficient execution",
        "source": "Business Weekly",
        "url": "https://example.com/business-efficiency",
        "description": (
            "In 2026, investors increasingly favor startups that demonstrate disciplined execution, "
            "capital efficiency, and a clear path to sustainable profitability."
        ),
        "content": (
            "After years of prioritizing rapid user growth and market share expansion, many startups are "
            "recalibrating their strategies toward operational efficiency and financial sustainability. "
            "This shift reflects changing investor sentiment, higher interest rates, and a more cautious "
            "funding environment across global venture capital markets.\n\n"
            "Founders are paying closer attention to metrics such as burn multiple, gross margin stability, "
            "customer acquisition cost (CAC), and lifetime value (LTV). Rather than pursuing growth at all "
            "costs, leadership teams are focusing on repeatable go-to-market strategies, pricing discipline, "
            "and tighter alignment between product development and customer needs.\n\n"
            "Operationally, this trend has led to leaner teams, more deliberate hiring, and increased adoption "
            "of automation tools to improve productivity. Startups are also extending their cash runways by "
            "renegotiating vendor contracts, optimizing cloud spend, and prioritizing core revenue-generating "
            "products over experimental initiatives.\n\n"
            "Industry analysts note that while this environment may slow headline growth rates, it is producing "
            "more resilient companies with stronger fundamentals. For many founders, efficient execution is no "
            "longer a defensive move, but a competitive advantage in an increasingly selective market."
        ),
    },
    {
        "category": "sports",
        "title": "The evolution of tactics and training in elite football",
        "source": "Sports Insights",
        "url": "https://example.com/football-analysis",
        "description": (
            "Elite football clubs are increasingly leveraging data analysis, sports science, and tactical "
            "innovation to gain marginal advantages at the highest levels of competition."
        ),
        "content": (
            "Modern elite football has undergone a significant transformation over the past decade, driven "
            "by advances in sports science, tactical analytics, and training methodologies. Coaching staffs "
            "now rely heavily on match data, GPS tracking, and video analysis to optimize player performance "
            "and tactical decision-making.\n\n"
            "Training sessions are increasingly individualized, with workloads tailored to each player’s "
            "physical condition, injury history, and positional demands. Strength and conditioning programs "
            "are closely coordinated with tactical objectives, ensuring that players can execute complex "
            "pressing systems and positional rotations throughout a full match.\n\n"
            "Tactically, teams are experimenting with flexible formations that shift dynamically between "
            "defensive and attacking phases. Emphasis on build-up play from the back, compact defensive blocks, "
            "and coordinated pressing has reshaped how matches are approached at the elite level.\n\n"
            "As competition intensifies, marginal gains in preparation, recovery, and tactical execution "
            "can make the difference between success and failure. Clubs that effectively integrate analytics "
            "with traditional coaching expertise are increasingly setting the standard in top-tier football."
        ),
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
