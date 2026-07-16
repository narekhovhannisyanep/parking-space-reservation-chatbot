import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "parking_info.md"
INDEX_NAME = os.environ.get("PINECONE_INDEX", "parking-chatbot")
EMBEDDING_DIM = 768  # nomic-embed-text output dimension


def split_documents() -> list[Document]:
    text = DATA_PATH.read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "category")]
    )
    return splitter.split_text(text)


def _ensure_index(pc: Pinecone) -> None:
    indexes = {idx.name: idx for idx in pc.list_indexes()}
    if INDEX_NAME in indexes:
        if indexes[INDEX_NAME].dimension != EMBEDDING_DIM:
            pc.delete_index(INDEX_NAME)
        else:
            return
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


def ingest() -> None:
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    _ensure_index(pc)
    docs = split_documents()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print(f"Ingested {len(docs)} chunks into '{INDEX_NAME}'")


if __name__ == "__main__":
    ingest()
