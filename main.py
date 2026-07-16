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
