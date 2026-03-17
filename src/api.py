from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chatbot import ask

app = FastAPI(title="ViewZen RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.get("/health")
def health():
    """Simple health endpoint used by UI checks."""
    return {"status": "healthy"}


@app.post("/ask")
def ask_endpoint(req: QueryRequest):
    """Forward user question to RAG chatbot and return structured response."""
    try:
        return ask(query=req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))