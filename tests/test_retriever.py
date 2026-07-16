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
