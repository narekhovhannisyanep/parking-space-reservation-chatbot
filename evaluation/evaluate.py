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
