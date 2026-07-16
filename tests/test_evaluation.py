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
