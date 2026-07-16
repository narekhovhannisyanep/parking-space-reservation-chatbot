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
