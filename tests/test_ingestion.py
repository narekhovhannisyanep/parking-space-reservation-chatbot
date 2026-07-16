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
