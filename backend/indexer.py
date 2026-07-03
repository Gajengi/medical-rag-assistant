import uuid
import requests
import chromadb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "medical_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "medical_knowledge"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

print("Indexer started")
print("Reading documents from:", DATA_DIR)
print("Saving Chroma DB to:", CHROMA_DIR)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

try:
    chroma_client.delete_collection(name=COLLECTION_NAME)
    print("Old collection deleted")
except Exception:
    print("No old collection found")

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def simple_chunk_text(text, chunk_size=1100, overlap=210):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]

def extract_source_url(text):
    for line in text.splitlines():
        if line.startswith("Source URL:"):
            return line.replace("Source URL:", "").strip()
    return "Unknown"

def read_documents():
    documents = []

    for file_path in DATA_DIR.rglob("*.txt"):
        topic = file_path.parent.name

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        source_url = extract_source_url(text)
        
        documents.append({
            "text": text,
            "topic": topic,
            "filename": file_path.name,
            "file_path": str(file_path),
            "source_url": source_url
        })

    return documents


def build_index():
    docs = read_documents()
    print(f"Found {len(docs)} documents")

    total_chunks = 0

    for doc in docs:
        chunks = simple_chunk_text(doc["text"])

        for chunk_number, chunk in enumerate(chunks, start=1):
            print(f"Embedding {doc['filename']} chunk {chunk_number}/{len(chunks)}")

            embedding = get_embedding(chunk)

            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "topic": doc["topic"],
                    "filename": doc["filename"],
                    "source_url": doc["source_url"],
                    "chunk_number": chunk_number
                }]
            )

            total_chunks += 1

        print(f"Indexed {doc['filename']} | chunks: {len(chunks)}")

    print("Indexing completed")
    print("Total chunks:", total_chunks)


if __name__ == "__main__":
    build_index()