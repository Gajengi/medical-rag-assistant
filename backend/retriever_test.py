import requests
import chromadb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "medical_knowledge"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(name=COLLECTION_NAME)

def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": text},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]

question = "What are the symptoms of diabetes?"

query_embedding = get_embedding(question)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

for i, doc in enumerate(results["documents"][0], start=1):
    print(f"\n--- Result {i} ---")
    print(doc[:700])
    print("\nSource:")
    print(results["metadatas"][0][i - 1])