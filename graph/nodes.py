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
