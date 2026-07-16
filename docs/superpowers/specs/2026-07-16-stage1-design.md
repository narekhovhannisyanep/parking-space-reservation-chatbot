# Stage 1 Design: RAG System and Chatbot

**Date:** 2026-07-16  
**Scope:** Stage 1 of the parking space reservation chatbot

---

## Overview

A LangGraph-based RAG chatbot that answers questions about a parking facility and interactively collects reservation details from users via a CLI. All parking knowledge lives in Pinecone. Two local Ollama models handle different responsibilities: `gemma4:31b-cloud` for natural language generation and `qwen2.5:3b` for structured field extraction.

---

## Architecture

```
CLI loop
  └── LangGraph graph (ChatState)
        ├── Node: route        — classifies intent: info query vs reservation
        ├── Node: retrieve     — Pinecone top-3 similarity search
        ├── Node: generate     — gemma4:31b-cloud generates answer from context
        ├── Node: collect      — qwen2.5:3b extracts reservation fields into structured output
        └── Node: confirm      — summarises collected fields, asks user to confirm
```

**LLM assignment:**
- `gemma4:31b-cloud` — all natural language generation (answering, routing, confirmation messages)
- `qwen2.5:3b` — structured extraction only (parsing name, surname, car number, dates from free text)

---

## LangGraph State

```python
class ReservationData(TypedDict):
    name: str | None
    surname: str | None
    car_number: str | None
    period_start: str | None
    period_end: str | None

class ChatState(TypedDict):
    messages: list[BaseMessage]
    query: str
    context: list[str]
    mode: Literal["info", "reservation"]
    reservation: ReservationData
    awaiting_confirmation: bool
```

**Flow:**
- Every user message enters the graph at `route`
- `info` path: `route` → `retrieve` → `generate` → exit
- `reservation` path: `route` → `collect` (loops until all fields filled) → `confirm` → exit
- On confirmation, Stage 2 will add an admin escalation node after `confirm`

---

## Mock Data

Single file `data/parking_info.md` with sections:
- General Information (facility name, capacity, space types: standard, disabled, EV)
- Location (address, nearby landmarks)
- Working Hours (weekday/weekend schedule)
- Prices (hourly, daily, monthly rates per space type)
- Availability (approximate availability per zone)
- Booking Process (reservation steps, required info, cancellation policy)

---

## Pinecone Ingestion

- **Index name:** `parking-chatbot`
- **Embedding model:** `nomic-embed-text` via Ollama (requires `ollama pull nomic-embed-text` — not in the current model list)
- **Chunking:** markdown section-aware splitting, ~500 tokens, 50-token overlap
- **Metadata per chunk:** `category` (section name), `source` (filename)
- **Retrieval:** top-3 chunks by cosine similarity

Script: `vector_store/ingestion.py` — run once to populate the index.

---

## Guardrails

Minimal: a thin post-generation check in the `generate` node that ensures the response does not echo raw reservation records (simple pattern match against known PII formats like plate number patterns). No heavy NLP model dependency.

---

## Evaluation

Script: `evaluation/evaluate.py`  
Dataset: `evaluation/test_dataset.json` — 15 Q&A pairs with `expected_chunks` (category tags) and `expected_answer_keywords`.

**Metrics:**
- **Recall@3** — were relevant chunks in top-3 results?
- **Precision@3** — fraction of top-3 results that were relevant?
- **Mean response latency** — wall-clock time per query, averaged across test set

Relevance determined by chunk `category` metadata matching expected categories (no LLM judge needed).

Output: printed report + `evaluation/results.json`.

---

## Project Structure

```
parking-space-reservation-chatbot/
├── data/
│   └── parking_info.md
├── vector_store/
│   ├── ingestion.py
│   └── retriever.py
├── graph/
│   ├── state.py
│   ├── nodes.py
│   └── pipeline.py
├── evaluation/
│   ├── test_dataset.json
│   └── evaluate.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_retriever.py
│   ├── test_nodes.py
│   └── test_evaluation.py
├── main.py
├── .env.example
└── requirements.txt
```

---

## Out of Scope for Stage 1

- Admin approval flow (Stage 2)
- MCP server integration (Stage 3)
- Full LangGraph multi-agent orchestration (Stage 4)
- CI/CD, Terraform, PowerPoint presentation
