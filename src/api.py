"""
FastAPI application – ViewZen RAG Chatbot API
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.chatbot import ask
from src.config import settings
from src.vector_store import build_vector_store, get_index_stats

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure vector store is ready on startup."""
    logger.info("Starting up – connecting to Pinecone …")
    try:
        stats = get_index_stats()
        if stats["total_vector_count"] == 0:
            logger.warning(
                "Vector store is empty. Run the pipeline first:\n"
                "  python -m src.pipeline"
            )
        else:
            logger.info("Pinecone index ready with %d vectors", stats["total_vector_count"])
    except Exception as exc:
        logger.error("Pinecone init error: %s", exc)
    yield
    logger.info("Shutting down …")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ViewZen RAG Chatbot API",
    description=(
        "AI-powered assistant that answers questions about ViewZen products "
        "using official documentation as the knowledge source."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of chunks to retrieve")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the standard roles in ViewZen?",
                    "top_k": 5,
                }
            ]
        }
    }


class SourceReference(BaseModel):
    url: str
    title: str
    section: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceReference]


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"message": "ViewZen RAG Chatbot API is running. Visit /docs for API documentation."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check service health and vector store status."""
    try:
        stats = get_index_stats()
        count = stats["total_vector_count"]
    except Exception:
        count = 0
    return HealthResponse(
        status="healthy",
        vector_store_count=count,
        model=settings.llm_model,
    )


@app.post("/ask", response_model=QueryResponse, tags=["Chat"])
async def ask_endpoint(request: QueryRequest):
    """
    Ask a question about ViewZen products.

    The system retrieves relevant documentation context and generates
    an accurate, source-backed answer using an LLM.
    """
    try:
        result = ask(query=request.query, top_k=request.top_k)
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            sources=[SourceReference(**s) for s in result["sources"]],
        )
    except Exception as exc:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
