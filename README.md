# ViewZen RAG Chatbot (Simple Version)

A simple RAG chatbot for ViewZen docs.

Tech stack:
- Scraping: Playwright
- Vector DB: Pinecone
- Backend: FastAPI
- UI: Streamlit
- LLM: Google Gemini

## Project flow
1. Scrape docs from sitemap
2. Clean and split into chunks
3. Create embeddings and store in Pinecone
4. Retrieve best chunks for a query
5. Generate answer with OpenAI

## Files
- `src/scraper.py`: scrape docs
- `src/processor.py`: clean + chunk
- `src/vector_store.py`: embed + query Pinecone
- `src/chatbot.py`: RAG answer logic
- `src/api.py`: FastAPI endpoints
- `src/pipeline.py`: run all steps
- `streamlit_app.py`: chat UI

## Setup
1. Create `.env` from template:
   - `cp .env.example .env`
2. Fill required values:
   - `GOOGLE_API_KEY`
   - `PINECONE_API_KEY`
3. Install deps:
   - `pip install -r requirements.txt`
4. Install browser:
   - `playwright install chromium`

## Run pipeline
```bash
python -m src.pipeline
```

## Run app locally
One command (recommended):
```bash
bash run.sh
```

Or run manually:

Terminal 1:
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:
```bash
python -m streamlit run streamlit_app.py --server.port 8501
```

Open:
- API docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

## Docker
```bash
docker compose up --build
```

Open:
- API docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`
