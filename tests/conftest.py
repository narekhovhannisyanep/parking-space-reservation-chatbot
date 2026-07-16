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
