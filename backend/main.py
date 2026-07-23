import requests
import chromadb
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "medical_knowledge"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:0.5b"

app = FastAPI(title="Medical Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(name=COLLECTION_NAME)


class QuestionRequest(BaseModel):
    question: str


def get_embedding(text: str):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]


def retrieve_chunks(question: str, top_k: int = 3):
    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    chunks = []

    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": doc,
            "metadata": metadata
        })

    return chunks


def generate_answer(question: str, chunks):
    context = "\n\n".join(
        [f"Source {i+1}:\n{chunk['text']}" for i, chunk in enumerate(chunks)]
    )

    prompt = f"""
You are a Medical Knowledge Assistant.

Answer in plain text.

Rules:
- Do NOT use Markdown.
- Do NOT use ** or __.
- Do NOT use # headings.
- Use numbered headings only.
- Use simple paragraphs and bullet points.

Use ONLY the provided context to answer the user's question.

Your response should be detailed, clear, and educational.

Important rules:
1. Do not diagnose the user.
2. Do not prescribe medicine, dosage, or personalized treatment.
3. Do not claim to cure or reverse a disease.
4. If the user asks for a personal plan, explain that you cannot personalize medical advice.
5. Provide general medical education if supported by the context.
6. If the context does not contain enough information, clearly say so.
7. Recommend consulting a qualified healthcare professional for personal medical decisions.

Answer format:

1. Direct answer
- Start with a clear answer to the user's question.

2. Explanation
- Explain the concept in simple language.
- Use examples if helpful.

3. Causes or risk factors
- Include only if relevant to the question and available in the context.

4. General management or prevention
- Mention lifestyle, monitoring, or prevention steps only if supported by the context.

5. When to seek medical care
- Mention when the user should consult a healthcare professional.

6. Sources used
- List the source numbers used from the context.

Context:
{context}

User question:
{question}

Now provide a complete answer.
"""
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 400,
                "temperature": 0.2
            }
        },
        timeout=240
    )

    response.raise_for_status()
    return response.json()["response"]

#made changes in return
@app.get("/")
def home():
    return {"message": "Medical Knowledge Asssistant API is runnning"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    chunks = retrieve_chunks(request.question)
    answer = generate_answer(request.question, chunks)

    return {
        "question": request.question,
        "answer": answer,
        "sources": [chunk["metadata"] for chunk in chunks],
    }
