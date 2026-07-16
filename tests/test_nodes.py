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
