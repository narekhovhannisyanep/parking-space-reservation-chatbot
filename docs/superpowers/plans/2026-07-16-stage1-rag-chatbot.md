# Stage 1: RAG Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-based RAG chatbot that answers parking queries and collects reservation details via CLI, backed by Pinecone and local Ollama models.

**Architecture:** Five-node LangGraph graph (route → retrieve/collect → generate/confirm) with MemorySaver for multi-turn state. `nomic-embed-text` handles embeddings; `gemma4:31b-cloud` handles generation; `qwen2.5:3b` handles structured field extraction. Pinecone stores document chunks split from a mock markdown data file.

**Tech Stack:** Python 3.11+, langchain 0.3+, langgraph 0.2+, langchain-ollama, langchain-pinecone, langchain-text-splitters, pinecone 5+, pydantic 2+, pytest

---

## File Map

| File | Responsibility |
|------|----------------|
| `requirements.txt` | All project dependencies |
| `.env.example` | Environment variable template |
| `data/parking_info.md` | Mock parking facility content (6 `##` sections) |
| `vector_store/__init__.py` | Package marker |
| `vector_store/ingestion.py` | Load markdown, split by `##` header, embed, push to Pinecone |
| `vector_store/retriever.py` | Wrap Pinecone index as a LangChain retriever (top-3) |
| `graph/__init__.py` | Package marker |
| `graph/state.py` | `ChatState` and `ReservationData` TypedDicts |
| `graph/nodes.py` | `route`, `retrieve`, `generate`, `collect`, `confirm` node functions |
| `graph/pipeline.py` | Assemble and compile LangGraph StateGraph with MemorySaver |
| `evaluation/__init__.py` | Package marker |
| `evaluation/test_dataset.json` | 15 Q&A pairs with `expected_categories` |
| `evaluation/evaluate.py` | Recall@3, Precision@3, mean retrieval + generation latency |
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | `make_state()` helper + shared fixtures |
| `tests/test_ingestion.py` | Tests for document splitting and metadata |
| `tests/test_retriever.py` | Tests for retriever wrapper |
| `tests/test_nodes.py` | Tests for each LangGraph node function |
| `tests/test_evaluation.py` | Tests for Recall@K and Precision@K functions |
| `main.py` | CLI loop: read input → invoke graph → print response |

---

## Task 0: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `vector_store/__init__.py`, `graph/__init__.py`, `evaluation/__init__.py`, `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
langchain>=0.3.0
langchain-community>=0.3.0
langchain-ollama>=0.2.0
langchain-pinecone>=0.2.0
langchain-text-splitters>=0.3.0
langgraph>=0.2.0
pinecone>=5.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install without errors.

- [ ] **Step 3: Create `.env.example`**

```
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX=parking-chatbot
```

- [ ] **Step 4: Copy `.env.example` to `.env` and fill in your Pinecone API key**

Run: `cp .env.example .env`
Then open `.env` and replace `your-pinecone-api-key-here` with your actual key from https://app.pinecone.io.

- [ ] **Step 5: Create empty package `__init__.py` files**

Create four empty files:
- `vector_store/__init__.py`
- `graph/__init__.py`
- `evaluation/__init__.py`
- `tests/__init__.py`

- [ ] **Step 6: Create `tests/conftest.py`**

```python
import pytest
from graph.state import ChatState, ReservationData


def make_state(**overrides) -> ChatState:
    base: ChatState = {
        "messages": [],
        "query": "test query",
        "context": [],
        "mode": "info",
        "reservation": {
            "name": None,
            "surname": None,
            "car_number": None,
            "period_start": None,
            "period_end": None,
        },
        "awaiting_confirmation": False,
    }
    base.update(overrides)
    return base


@pytest.fixture
def sample_state() -> ChatState:
    return make_state()


@pytest.fixture
def full_reservation() -> dict:
    return {
        "name": "John",
        "surname": "Doe",
        "car_number": "AB123CD",
        "period_start": "2026-07-20",
        "period_end": "2026-07-25",
    }
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example vector_store/__init__.py graph/__init__.py evaluation/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: project setup — dependencies, env template, package structure"
```

---

## Task 1: Mock Parking Data

**Files:**
- Create: `data/parking_info.md`

- [ ] **Step 1: Create `data/parking_info.md`**

```markdown
## General Information
Park & Go is a modern 5-level parking facility in the heart of the city centre. It offers 225 spaces across three categories: standard, accessible (disabled), and EV charging. The facility is staffed 7 days a week and features 24/7 CCTV surveillance, barrier-controlled entry, and a mobile app for advance booking.

## Location
Address: 47 Central Avenue, City Centre, CC1 2AB
Nearest landmarks: Central Train Station (200 m north), City Hall (350 m east), Grand Shopping Mall (100 m west)
GPS coordinates: 51.5074 N, 0.1278 W
Public transport: Bus lines 12, 34, and 56 stop directly outside. Nearest metro: Central Park Station (Line 2), 3 min walk.

## Working Hours
Monday to Friday: 06:00 to 23:00
Saturday: 07:00 to 22:00
Sunday: 08:00 to 21:00
Public Holidays: 09:00 to 20:00
The facility is closed on Christmas Day (25 December) and New Year's Day (1 January).

## Prices
Standard space:
  Hourly: $2.50 per hour
  Daily (up to 24 hours): $18.00
  Monthly pass: $120.00

Accessible (disabled) space:
  Hourly: $1.25 per hour (50 percent discount)
  Daily: $9.00
  Monthly pass: $65.00

EV Charging space (charging included):
  Hourly: $4.00 per hour
  Daily: $28.00
  Monthly pass: $180.00

First 15 minutes are free for all space types.

## Availability
Level 1 Ground floor - Standard: 45 of 80 spaces available
Level 2 - EV Charging: 12 of 20 spaces available
Level 3 - Accessible: 8 of 15 spaces available
Level 4 - Standard: 30 of 60 spaces available
Level 5 Top floor - Standard: 22 of 50 spaces available
Total: 117 of 225 spaces currently available

## Booking Process
1. Start a reservation conversation with the chatbot.
2. Provide your full name, car registration number, and desired reservation period (start date and end date).
3. Your request is forwarded to an administrator for confirmation.
4. You will receive a confirmation or refusal within 15 minutes.
5. On arrival, present your confirmation reference at the entrance barrier.
Cancellations must be made at least 2 hours before the reservation start time.
No-show after 30 minutes from the reserved start time results in automatic cancellation.
```

- [ ] **Step 2: Commit**

```bash
git add data/parking_info.md
git commit -m "feat: add mock parking facility data"
```

---

## Task 2: Vector Store Ingestion

**Files:**
- Create: `vector_store/ingestion.py`
- Test: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing tests in `tests/test_ingestion.py`**

```python
def test_split_produces_six_chunks():
    from vector_store.ingestion import split_documents
    chunks = split_documents()
    assert len(chunks) == 6


def test_chunks_have_category_metadata():
    from vector_store.ingestion import split_documents
    chunks = split_documents()
    categories = {c.metadata["category"] for c in chunks}
    assert categories == {
        "General Information",
        "Location",
        "Working Hours",
        "Prices",
        "Availability",
        "Booking Process",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ingestion.py -v`
Expected: `ImportError` — `vector_store.ingestion` does not exist yet.

- [ ] **Step 3: Implement `vector_store/ingestion.py`**

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "parking_info.md"
INDEX_NAME = os.environ.get("PINECONE_INDEX", "parking-chatbot")
EMBEDDING_DIM = 768  # nomic-embed-text output dimension


def split_documents() -> list[Document]:
    text = DATA_PATH.read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "category")]
    )
    return splitter.split_text(text)


def _ensure_index(pc: Pinecone) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def ingest() -> None:
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    _ensure_index(pc)
    docs = split_documents()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print(f"Ingested {len(docs)} chunks into '{INDEX_NAME}'")


if __name__ == "__main__":
    ingest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingestion.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Run ingestion to populate Pinecone**

Run: `python -m vector_store.ingestion`
Expected: `Ingested 6 chunks into 'parking-chatbot'`
Note: Pinecone may take 10-20 seconds to initialise a new index on first run.

- [ ] **Step 6: Commit**

```bash
git add vector_store/ingestion.py tests/test_ingestion.py
git commit -m "feat: add Pinecone ingestion pipeline with MarkdownHeader splitting"
```

---

## Task 3: Vector Store Retriever

**Files:**
- Create: `vector_store/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests in `tests/test_retriever.py`**

```python
from unittest.mock import MagicMock, patch


def test_get_retriever_calls_as_retriever_with_k():
    with patch("vector_store.retriever.PineconeVectorStore") as MockStore:
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        MockStore.return_value = mock_store

        from vector_store.retriever import get_retriever
        result = get_retriever(k=3)

        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        assert result is mock_retriever


def test_get_retriever_default_k_is_three():
    with patch("vector_store.retriever.PineconeVectorStore") as MockStore:
        mock_store = MagicMock()
        MockStore.return_value = mock_store

        from vector_store.retriever import get_retriever
        get_retriever()

        mock_store.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: `ImportError` — `vector_store.retriever` does not exist yet.

- [ ] **Step 3: Implement `vector_store/retriever.py`**

```python
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.environ.get("PINECONE_INDEX", "parking-chatbot")


def get_retriever(k: int = 3):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return store.as_retriever(search_kwargs={"k": k})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vector_store/retriever.py tests/test_retriever.py
git commit -m "feat: add Pinecone retriever wrapper"
```

---

## Task 4: LangGraph State

**Files:**
- Create: `graph/state.py`

- [ ] **Step 1: Create `graph/state.py`**

```python
from typing import Annotated, Literal, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ReservationData(TypedDict):
    name: Optional[str]
    surname: Optional[str]
    car_number: Optional[str]
    period_start: Optional[str]
    period_end: Optional[str]


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    context: list[str]
    mode: Literal["info", "reservation"]
    reservation: ReservationData
    awaiting_confirmation: bool
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "from graph.state import ChatState, ReservationData; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add graph/state.py
git commit -m "feat: add LangGraph ChatState and ReservationData TypedDicts"
```

---

## Task 5: LangGraph Nodes

**Files:**
- Create: `graph/nodes.py`
- Test: `tests/test_nodes.py`

- [ ] **Step 1: Write failing tests in `tests/test_nodes.py`**

```python
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from tests.conftest import make_state


# ── route ──────────────────────────────────────────────────────────────────

def test_route_returns_info_mode():
    with patch("graph.nodes.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value = MagicMock(content="info")
        from graph.nodes import route
        result = route(make_state(query="What are the working hours?"))
    assert result["mode"] == "info"


def test_route_returns_reservation_mode():
    with patch("graph.nodes.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value = MagicMock(content="reservation")
        from graph.nodes import route
        result = route(make_state(query="I want to reserve a parking space"))
    assert result["mode"] == "reservation"


# ── retrieve ────────────────────────────────────────────────────────────────

def test_retrieve_fills_context():
    fake_doc = MagicMock()
    fake_doc.page_content = "Monday to Friday: 06:00 to 23:00"
    with patch("graph.nodes.get_retriever") as mock_get:
        mock_get.return_value.invoke.return_value = [fake_doc, fake_doc, fake_doc]
        from graph.nodes import retrieve
        result = retrieve(make_state(query="working hours"))
    assert len(result["context"]) == 3
    assert result["context"][0] == "Monday to Friday: 06:00 to 23:00"


def test_retrieve_returns_empty_context_when_no_docs():
    with patch("graph.nodes.get_retriever") as mock_get:
        mock_get.return_value.invoke.return_value = []
        from graph.nodes import retrieve
        result = retrieve(make_state(query="anything"))
    assert result["context"] == []


# ── generate ────────────────────────────────────────────────────────────────

def test_generate_returns_ai_message():
    with patch("graph.nodes.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value = MagicMock(content="We open at 6am on weekdays.")
        from graph.nodes import generate
        result = generate(make_state(query="hours?", context=["06:00-23:00"]))
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)


def test_generate_guardrail_redacts_plate_pattern():
    with patch("graph.nodes.ChatOllama") as MockLLM:
        MockLLM.return_value.invoke.return_value = MagicMock(
            content="The car AB-1234-CD has been registered."
        )
        from graph.nodes import generate
        result = generate(make_state(query="info", context=[]))
    assert "AB-1234-CD" not in result["messages"][0].content


# ── collect ─────────────────────────────────────────────────────────────────

def test_collect_partial_fields_asks_for_missing():
    mock_fields = MagicMock()
    mock_fields.name = "Alice"
    mock_fields.surname = None
    mock_fields.car_number = None
    mock_fields.period_start = None
    mock_fields.period_end = None

    with patch("graph.nodes.ChatOllama") as MockLLM:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_fields
        MockLLM.return_value.with_structured_output.return_value = mock_chain
        from graph.nodes import collect
        result = collect(make_state(query="My name is Alice"))

    assert result["reservation"]["name"] == "Alice"
    assert result["awaiting_confirmation"] is False
    assert len(result["messages"]) == 1


def test_collect_all_fields_sets_awaiting_confirmation():
    mock_fields = MagicMock()
    mock_fields.name = "Alice"
    mock_fields.surname = "Smith"
    mock_fields.car_number = "AB123CD"
    mock_fields.period_start = "2026-07-20"
    mock_fields.period_end = "2026-07-25"

    with patch("graph.nodes.ChatOllama") as MockLLM:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_fields
        MockLLM.return_value.with_structured_output.return_value = mock_chain
        from graph.nodes import collect
        result = collect(make_state(query="all details provided"))

    assert result["awaiting_confirmation"] is True
    assert result.get("messages", []) == []


# ── confirm ──────────────────────────────────────────────────────────────────

def test_confirm_includes_all_reservation_fields(full_reservation):
    from graph.nodes import confirm
    result = confirm(make_state(reservation=full_reservation))
    text = result["messages"][0].content
    assert "John" in text
    assert "Doe" in text
    assert "AB123CD" in text
    assert "2026-07-20" in text


def test_confirm_returns_single_ai_message(full_reservation):
    from graph.nodes import confirm
    result = confirm(make_state(reservation=full_reservation))
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -v`
Expected: `ImportError` — `graph.nodes` does not exist yet.

- [ ] **Step 3: Implement `graph/nodes.py`**

```python
import re
from typing import Optional
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from graph.state import ChatState, ReservationData
from vector_store.retriever import get_retriever


class ReservationFields(BaseModel):
    name: Optional[str] = Field(default=None, description="First name")
    surname: Optional[str] = Field(default=None, description="Last name")
    car_number: Optional[str] = Field(default=None, description="Car registration plate")
    period_start: Optional[str] = Field(default=None, description="Reservation start date")
    period_end: Optional[str] = Field(default=None, description="Reservation end date")


_FIELD_LABELS = {
    "name": "first name",
    "surname": "last name",
    "car_number": "car registration number",
    "period_start": "reservation start date",
    "period_end": "reservation end date",
}

_PLATE_PATTERN = re.compile(r"\b[A-Z]{2,3}[-\s]?\d{2,4}[-\s]?[A-Z]{0,3}\b")


def route(state: ChatState) -> dict:
    llm = ChatOllama(model="gemma4:31b-cloud")
    prompt = (
        "Classify the user's intent as exactly one word: 'reservation' or 'info'.\n"
        f"User message: {state['query']}"
    )
    response = llm.invoke(prompt)
    mode = "reservation" if "reservation" in response.content.lower() else "info"
    return {"mode": mode}


def retrieve(state: ChatState) -> dict:
    docs = get_retriever(k=3).invoke(state["query"])
    return {"context": [doc.page_content for doc in docs]}


def generate(state: ChatState) -> dict:
    llm = ChatOllama(model="gemma4:31b-cloud")
    context_text = "\n\n".join(state["context"]) if state["context"] else "No context available."
    prompt = (
        "You are a helpful parking facility assistant. "
        "Use only the context below to answer the question. "
        "If the answer is not in the context, say you don't have that information.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {state['query']}"
    )
    response = llm.invoke(prompt)
    answer = _PLATE_PATTERN.sub("[REDACTED]", response.content)
    return {"messages": [AIMessage(content=answer)]}


def collect(state: ChatState) -> dict:
    llm = ChatOllama(model="qwen2.5:3b")
    extractor = llm.with_structured_output(ReservationFields)
    existing = state["reservation"]
    prompt = (
        "Extract reservation details explicitly mentioned in the message. "
        "Return null for any field not mentioned.\n"
        f"Already known fields: {dict(existing)}\n"
        f"User message: {state['query']}"
    )
    extracted: ReservationFields = extractor.invoke(prompt)
    updated: ReservationData = {
        "name": extracted.name or existing.get("name"),
        "surname": extracted.surname or existing.get("surname"),
        "car_number": extracted.car_number or existing.get("car_number"),
        "period_start": extracted.period_start or existing.get("period_start"),
        "period_end": extracted.period_end or existing.get("period_end"),
    }
    all_filled = all(v is not None for v in updated.values())
    if not all_filled:
        first_missing = next(k for k, v in updated.items() if v is None)
        ask = f"To complete your reservation, could you please provide your {_FIELD_LABELS[first_missing]}?"
        return {"reservation": updated, "awaiting_confirmation": False, "messages": [AIMessage(content=ask)]}
    return {"reservation": updated, "awaiting_confirmation": True}


def confirm(state: ChatState) -> dict:
    r = state["reservation"]
    summary = (
        "Here is a summary of your reservation request:\n"
        f"  Name: {r['name']} {r['surname']}\n"
        f"  Car number: {r['car_number']}\n"
        f"  Period: {r['period_start']} to {r['period_end']}\n\n"
        "Type 'yes' to submit for administrator approval, or 'no' to cancel."
    )
    return {"messages": [AIMessage(content=summary)]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nodes.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add graph/nodes.py tests/test_nodes.py
git commit -m "feat: implement LangGraph nodes (route, retrieve, generate, collect, confirm)"
```

---

## Task 6: LangGraph Pipeline + CLI

**Files:**
- Create: `graph/pipeline.py`
- Create: `main.py`

- [ ] **Step 1: Create `graph/pipeline.py`**

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from graph.state import ChatState
from graph.nodes import route, retrieve, generate, collect, confirm


def _after_route(state: ChatState) -> str:
    return state["mode"]


def _after_collect(state: ChatState) -> str:
    return "confirm" if state["awaiting_confirmation"] else END


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("collect", collect)
    g.add_node("confirm", confirm)

    g.set_entry_point("route")
    g.add_conditional_edges("route", _after_route, {
        "info": "retrieve",
        "reservation": "collect",
    })
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    g.add_conditional_edges("collect", _after_collect, {
        "confirm": "confirm",
        END: END,
    })
    g.add_edge("confirm", END)

    return g.compile(checkpointer=MemorySaver())


app = build_graph()
```

- [ ] **Step 2: Verify the graph compiles**

Run: `python -c "from graph.pipeline import app; print('Graph compiled OK')"`
Expected: `Graph compiled OK`

- [ ] **Step 3: Create `main.py`**

```python
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph.pipeline import app
from graph.state import ReservationData

load_dotenv()

CONFIG = {"configurable": {"thread_id": "session"}}


def main() -> None:
    print("Parking Space Reservation Chatbot")
    print("Type 'quit' to exit.\n")
    first_turn = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        if first_turn:
            input_state = {
                "messages": [HumanMessage(content=user_input)],
                "query": user_input,
                "context": [],
                "mode": "info",
                "reservation": ReservationData(
                    name=None, surname=None, car_number=None,
                    period_start=None, period_end=None,
                ),
                "awaiting_confirmation": False,
            }
            first_turn = False
        else:
            input_state = {
                "messages": [HumanMessage(content=user_input)],
                "query": user_input,
            }

        result = app.invoke(input_state, CONFIG)
        print(f"\nBot: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke test the CLI**

Run: `python main.py`
Type: `What are the working hours on weekdays?`
Expected: Bot responds with hours from the parking data.
Type: `I want to reserve a space`
Expected: Bot asks for your first name (or another first missing field).
Type: `quit`

- [ ] **Step 5: Commit**

```bash
git add graph/pipeline.py main.py
git commit -m "feat: assemble LangGraph pipeline and CLI entry point"
```

---

## Task 7: Evaluation

**Files:**
- Create: `evaluation/test_dataset.json`
- Create: `evaluation/evaluate.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Create `evaluation/test_dataset.json`**

```json
[
  {"id": 1,  "question": "What are the working hours on weekdays?",       "expected_categories": ["Working Hours"]},
  {"id": 2,  "question": "What time does the parking close on Sunday?",   "expected_categories": ["Working Hours"]},
  {"id": 3,  "question": "Is the parking open on public holidays?",       "expected_categories": ["Working Hours"]},
  {"id": 4,  "question": "What is the hourly rate for a standard space?", "expected_categories": ["Prices"]},
  {"id": 5,  "question": "Do you offer monthly parking passes?",          "expected_categories": ["Prices"]},
  {"id": 6,  "question": "How much does an EV charging space cost daily?","expected_categories": ["Prices"]},
  {"id": 7,  "question": "Is there a discount for disabled drivers?",     "expected_categories": ["Prices"]},
  {"id": 8,  "question": "Where is the parking facility located?",        "expected_categories": ["Location"]},
  {"id": 9,  "question": "How do I get there by public transport?",       "expected_categories": ["Location"]},
  {"id": 10, "question": "How many EV charging spaces are available?",    "expected_categories": ["Availability"]},
  {"id": 11, "question": "How many total spaces does the facility have?", "expected_categories": ["Availability", "General Information"]},
  {"id": 12, "question": "What types of parking spaces are offered?",     "expected_categories": ["General Information"]},
  {"id": 13, "question": "What information do I need to make a reservation?", "expected_categories": ["Booking Process"]},
  {"id": 14, "question": "How long does it take to get a confirmation?",  "expected_categories": ["Booking Process"]},
  {"id": 15, "question": "What happens if I do not show up on time?",     "expected_categories": ["Booking Process"]}
]
```

- [ ] **Step 2: Write failing tests in `tests/test_evaluation.py`**

```python
import pytest
from unittest.mock import MagicMock


def _make_doc(category: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"category": category}
    return doc


def test_recall_at_k_hit():
    from evaluation.evaluate import recall_at_k
    docs = [_make_doc("Working Hours")] * 3
    assert recall_at_k(docs, ["Working Hours"]) is True


def test_recall_at_k_miss():
    from evaluation.evaluate import recall_at_k
    docs = [_make_doc("Prices")] * 3
    assert recall_at_k(docs, ["Working Hours"]) is False


def test_precision_at_k_all_relevant():
    from evaluation.evaluate import precision_at_k
    docs = [_make_doc("Prices")] * 3
    assert precision_at_k(docs, ["Prices"]) == pytest.approx(1.0)


def test_precision_at_k_partial():
    from evaluation.evaluate import precision_at_k
    docs = [_make_doc("Prices"), _make_doc("Location"), _make_doc("Location")]
    assert precision_at_k(docs, ["Prices"]) == pytest.approx(1 / 3)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_evaluation.py -v`
Expected: `ImportError` — `evaluation.evaluate` does not exist yet.

- [ ] **Step 4: Implement `evaluation/evaluate.py`**

```python
import json
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from vector_store.retriever import get_retriever
from graph.pipeline import app
from graph.state import ReservationData

load_dotenv()

DATASET_PATH = Path(__file__).parent / "test_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
K = 3


def recall_at_k(docs: list[Document], expected_categories: list[str]) -> bool:
    retrieved_cats = {d.metadata.get("category", "") for d in docs}
    return any(cat in retrieved_cats for cat in expected_categories)


def precision_at_k(docs: list[Document], expected_categories: list[str]) -> float:
    if not docs:
        return 0.0
    hits = sum(1 for d in docs if d.metadata.get("category", "") in expected_categories)
    return hits / len(docs)


def run_evaluation() -> dict[str, Any]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    retriever = get_retriever(k=K)
    recalls, precisions, retrieval_times, generation_times = [], [], [], []

    for item in dataset:
        question = item["question"]
        expected = item["expected_categories"]

        t0 = time.perf_counter()
        docs = retriever.invoke(question)
        retrieval_times.append(time.perf_counter() - t0)
        recalls.append(recall_at_k(docs, expected))
        precisions.append(precision_at_k(docs, expected))

        config = {"configurable": {"thread_id": f"eval-{item['id']}"}}
        t0 = time.perf_counter()
        app.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "query": question,
                "context": [],
                "mode": "info",
                "reservation": ReservationData(
                    name=None, surname=None, car_number=None,
                    period_start=None, period_end=None,
                ),
                "awaiting_confirmation": False,
            },
            config,
        )
        generation_times.append(time.perf_counter() - t0)

    results = {
        f"Recall@{K}": round(sum(recalls) / len(recalls), 4),
        f"Precision@{K}": round(sum(precisions) / len(precisions), 4),
        "mean_retrieval_latency_s": round(sum(retrieval_times) / len(retrieval_times), 4),
        "mean_generation_latency_s": round(sum(generation_times) / len(generation_times), 4),
        "total_questions": len(dataset),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for key, val in results.items():
        print(f"{key:38s}: {val}")
    return results


if __name__ == "__main__":
    run_evaluation()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Run evaluation against live Pinecone index**

Run: `python -m evaluation.evaluate`
Expected: Printed metrics table + `evaluation/results.json` written.

- [ ] **Step 7: Commit**

```bash
git add evaluation/test_dataset.json evaluation/evaluate.py tests/test_evaluation.py
git commit -m "feat: add RAG evaluation script (Recall@K, Precision@K, latency)"
```

---

## Task 8: Full Test Suite

- [ ] **Step 1: Run the complete test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS. Confirm at least 2 tests per module.

- [ ] **Step 2: Final commit**

```bash
git add .
git commit -m "chore: verify full test suite passes — Stage 1 complete"
```
