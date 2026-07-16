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
