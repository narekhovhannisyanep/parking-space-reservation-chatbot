# Parking Space Reservation Chatbot

An intelligent chatbot for parking space reservation built with Python, LangChain, and LangGraph. The system uses a RAG architecture to answer user queries, collects reservation details interactively, routes requests to a human administrator for approval, and persists confirmed reservations via an MCP server — all orchestrated as a single LangGraph pipeline.

---

## Architecture Overview

```
User
 │
 ▼
[Agent 1 — RAG Chatbot]
 │  · Answers questions about the parking facility (hours, prices, availability, location)
 │  · Collects reservation details (name, surname, car number, period)
 │  · Applies guardrails to prevent sensitive data exposure
 │
 ▼
[Agent 2 — Admin Approval Agent]  ◄──► Administrator (email / messenger / REST API)
 │  · Sends reservation request to admin
 │  · Waits for confirm / refuse response
 │
 ▼
[MCP Server — Reservation Writer]
 │  · Writes confirmed reservations to file
 │  · Format: Name | Car Number | Reservation Period | Approval Time
 │
 └── LangGraph orchestrates all nodes and state transitions
```

**Data storage (optional enhancement):**
- Static data (general info, parking details, location, booking process) → Vector database
- Dynamic data (availability, prices, working hours) → SQL database

---

## Stages

### Stage 1 — RAG System and Chatbot
- RAG pipeline over parking facility documents stored in a vector database (Milvus / Pinecone / Weaviate)
- Interactive conversation: answers user queries and collects reservation inputs (name, surname, car number, reservation period)
- Guardrails: NLP-based filtering to prevent exposure of sensitive data from the vector store
- RAG evaluation: Recall@K, Precision, request latency

### Stage 2 — Human-in-the-Loop Admin Agent
- Second LangChain agent responsible for contacting the administrator
- Sends reservation requests and receives confirm/refuse responses (via email, messenger, or REST API)
- Integrated with Agent 1 via LangGraph: reservation escalation triggers automatically after all user details are collected

### Stage 3 — MCP Server for Reservation Processing
- MCP server (open-source or custom FastAPI-based) handles confirmed reservations
- On admin approval, writes a record to a text file: `Name | Car Number | Reservation Period | Approval Time`
- Secured against unauthorized access

### Stage 4 — LangGraph Orchestration
- Full pipeline orchestrated as a LangGraph state graph:
  - **Node 1**: User interaction (RAG context + chatbot)
  - **Node 2**: Administrator approval (human-in-the-loop)
  - **Node 3**: Data recording (MCP server call)
- End-to-end integration testing and load testing across all components

---

## Project Structure

```
parking-space-reservation-chatbot/
├── agents/
│   ├── rag_agent.py          # Agent 1: RAG chatbot
│   └── admin_agent.py        # Agent 2: admin approval agent
├── graph/
│   └── pipeline.py           # LangGraph state graph (orchestration)
├── mcp_server/
│   └── server.py             # MCP server for writing confirmed reservations
├── data/
│   ├── static/               # Documents for vector store ingestion
│   └── dynamic/              # Seed data for SQL database (optional)
├── vector_store/
│   └── ingestion.py          # Embedding and indexing pipeline
├── guardrails/
│   └── filter.py             # Sensitive data filtering
├── evaluation/
│   └── metrics.py            # Recall@K, Precision, latency measurements
├── tests/                    # pytest test suite (≥2 tests per module)
├── reservations.txt          # Output file for confirmed reservations
├── requirements.txt
└── main.py                   # Entry point
```

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd parking-space-reservation-chatbot

# Create virtualenv and install dependencies
uv sync --extra dev
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Copy and fill in environment variables
cp .env.example .env
```

Required environment variables (`.env`):
```
ANTHROPIC_API_KEY=
VECTOR_DB_URL=
VECTOR_DB_API_KEY=
# Optional: SQL DB, email/messenger credentials for admin notifications
```

---

## Usage

```bash
# Run the full chatbot pipeline
python main.py

# Run the MCP server separately (if standalone)
python mcp_server/server.py

# Run tests
pytest

# Run a single test module
pytest tests/test_rag_agent.py -v
```

---

## Evaluation

The evaluation report covers:
- **Retrieval accuracy**: Recall@K and Precision against a labeled question set
- **Response latency**: end-to-end request timing per pipeline node
- **Integration tests**: full reservation flow from user input to file write

---

