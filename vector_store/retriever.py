import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = os.environ.get("PINECONE_INDEX", "parking-chatbot")


def get_retriever(k: int = 3):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return store.as_retriever(search_kwargs={"k": k})
