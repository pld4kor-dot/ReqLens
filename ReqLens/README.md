# ReqLens

**An Evidence-Grounded Multi-Agent System for Graph-Based Requirements Engineering.**

> *LLMs propose. Evidence gates. Graph validates. Humans approve.*

---

## Setup

### 1. Create and activate the virtual environment

```bash
cd /path/to/ReqLens

conda create -p ./myenv python=3.11 -y
conda activate ./myenv
```

### 2. Install the package

```bash
pip install -e ".[dev]"
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```env
# Required — Azure OpenAI credentials
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_BASE_URL=https://<your-resource>.openai.azure.com/openai/v1/

# Model deployments — set to your actual deployment names
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5
AZURE_OPENAI_REASONING_DEPLOYMENT=gpt-5-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Database (PostgreSQL with pgvector)
DATABASE_URL=postgresql+psycopg://reqlens:reqlens@localhost:5432/reqlens

# Vector and graph backends
VECTOR_BACKEND=pgvector
GRAPH_BACKEND=networkx

# Neo4j (only needed if running with --profile full)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-neo4j-password>

# Runtime
ENVIRONMENT=dev
LOG_LEVEL=INFO
ENABLE_LLM_CACHE=true
ENABLE_HUMAN_REVIEW_GATE=true
```

### 4. Start infrastructure services

```bash
# Start PostgreSQL and Redis (minimum required)
docker compose up -d

# Start PostgreSQL, Redis, and Neo4j
docker compose --profile full up -d
```

### 5. Run database migrations

```bash
alembic upgrade head
```

---

## Running the Application

### Terminal 1 — API server

```bash
conda activate ./myenv
python -m uvicorn reqlens.main:app --reload --port 8081
```

### Terminal 2 — Streamlit UI

```bash
conda activate ./myenv
python -m streamlit run src/reqlens/ui/streamlit_app.py
```

The UI is available at `http://localhost:8501`.

> **Remote server:** If running on a remote machine, open an SSH tunnel on your local machine:
> ```bash
> ssh -L 8501:localhost:8501 -L 8081:localhost:8081 <user>@<server-ip>
> ```
> Then browse to `http://localhost:8501` locally.

> **Port conflict:** If port 8001 is already in use, start on a different port and set the env var so the UI can find the API:
> ```bash
> # Terminal 1
> python -m uvicorn reqlens.main:app --reload --port 8081
> # Terminal 2
> python -m streamlit run src/reqlens/ui/streamlit_app.py
> ```

---

## Pipeline

```
Upload source documents
        ↓
Split into evidence source spans
        ↓
Extract candidate requirements
        ↓
Verify each candidate against source evidence
        ↓
Classify FR / NFR / NFR subtype
        ↓
Detect ambiguity and vague phrasing
        ↓
Build requirement dependency graph
        ↓
Detect duplicates, conflicts, and missing traces
        ↓
Generate elicitation questions for unresolved gaps
        ↓
Compose final SRS from accepted graph nodes only
```

---

## Agents

| Agent | Responsibility |
|-------|----------------|
| Extraction | Extract atomic candidate requirements from source spans |
| Evidence | Gate each candidate — accept, reject, or flag hallucinations |
| Classification | Label each requirement as FR / NFR / constraint / assumption |
| Ambiguity | Detect vague or unverifiable phrasing |
| Dependency | Map requires / conflicts / duplicates / refines edges |
| Consistency | Detect unresolved contradictions across requirements |
| Traceability | Build requirement-to-source trace matrix |
| Elicitation | Generate stakeholder clarification questions |
| Composer | Generate final SRS document from accepted graph nodes only |
| Impact | Analyse change impact across the dependency graph |

---

## Demo Workflow

After the API is running, use the scripts to ingest a demo project and run the pipeline:

```bash
# Ingest the demo project documents
python scripts/ingest_demo_project.py --base-url http://localhost:8081

# Run the full pipeline on the ingested project
python scripts/run_pipeline.py --project-id <PROJECT_ID> --base-url http://localhost:8081
```

---

## Testing

```bash
pytest                    # unit tests
pytest -m integration     # integration tests (requires running DB)
pytest -m llm             # LLM tests (requires Azure OpenAI credentials)
```

---

## License

MIT
