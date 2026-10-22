"""rag_app.py: a local Retrieval-Augmented Generation (RAG) application.

Everything runs on this machine and costs nothing: Sentence Transformers embeds
the text, FAISS searches it, a cross-encoder reranks the candidates, and Qwen 3
4B answers through Ollama's local API. No OpenAI SDK, API key, or .env file.

Pipeline for each question:
  1. embed the question and pull the top_k closest chunks from FAISS
  2. rerank those candidates with the cross-encoder and keep the best top_m
  3. send the kept chunks to Qwen with an instruction to answer only from them

Run:  python rag_app.py                interactive questions (exit or quit to stop)
      python rag_app.py --experiment   chunk_size / chunk_overlap comparison (Step 6)
"""

# ---------------------------------------------------------------------------
# Adjustable parameters. Change these, save, and restart: chunks and
# embeddings are rebuilt at startup. To try another model, run
# `ollama pull <model>` and change llm_model.
# ---------------------------------------------------------------------------
chunk_size = 500          # characters per chunk
chunk_overlap = 50        # characters shared by neighboring chunks
embedding_model_name = "sentence-transformers/all-distilroberta-v1"   # 768-number vectors
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"           # scores (question, chunk) pairs
top_k = 20                # candidates FAISS returns per question
top_m = 8                 # chunks kept after reranking and sent to the model
llm_model = "qwen3:4b"
ollama_url = "http://localhost:11434/api/chat"

SOURCE_FILE = "Selected_Document.txt"
# The first wording, "say that you don't know", made qwen3:4b-instruct reply
# with the literal sentence "You don't know." Quoting the exact reply fixed it.
SYSTEM_PROMPT = (
    "You are a careful assistant. Answer the question using only the context provided. "
    'If the context does not contain the answer, reply exactly: "I don\'t know." '
    "Do not add facts that are not in the context."
)

# ---------------------------------------------------------------------------
# Quiet the libraries so only the program's own output shows.
# ---------------------------------------------------------------------------
import logging
import warnings
from transformers import logging as hf_logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

import sys

import faiss
import numpy as np
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer


def load_and_split_document(path=SOURCE_FILE, size=chunk_size, overlap=chunk_overlap):
    """Read the source document and split it into overlapping chunks.

    The splitter tries paragraph breaks first, then line breaks, then spaces,
    and only cuts inside a word as a last resort.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        sys.exit(f"Error: {path} was not found. Run the text extractor first.")
    if not text.strip():
        sys.exit(f"Error: {path} is empty.")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def build_faiss_index(chunks, embedder):
    """Embed every chunk and store the vectors in an exact (flat) L2 FAISS index."""
    vectors = np.array(embedder.encode(chunks, show_progress_bar=False), dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def dedupe_preserve_order(chunks):
    """Drop repeated chunks (ignoring whitespace differences) and keep the first of each."""
    seen, unique = set(), []
    for chunk in chunks:
        key = " ".join(chunk.split())
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def retrieve_and_rerank(question, chunks, index, embedder, reranker, k=top_k, m=top_m):
    """Retrieve up to k candidates with FAISS, rerank them, and return the best m."""
    q_vec = np.array(embedder.encode([question], show_progress_bar=False), dtype="float32")
    _, ids = index.search(q_vec, min(k, len(chunks)))
    candidates = [chunks[i] for i in ids[0] if i != -1]
    scores = reranker.predict([(question, c) for c in candidates])   # higher = more relevant
    ranked = [c for c, _ in sorted(zip(candidates, scores), key=lambda p: p[1], reverse=True)]
    return dedupe_preserve_order(ranked[:m])


def generate_answer(question, context_chunks):
    """Ask Qwen through Ollama's chat API, giving it only the retrieved context."""
    context = "\n\n".join(context_chunks)
    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        "stream": False,
        "think": False,                       # as the assignment specifies; see the note below for qwen3:4b
        "options": {"num_predict": 500},      # cap on generated tokens
    }
    try:
        response = requests.post(ollama_url, json=payload, timeout=300)
        response.raise_for_status()
    except requests.ConnectionError:
        return "Error: could not reach Ollama. Start it with `ollama serve` and try again."
    except requests.HTTPError as e:
        return f"Error: Ollama returned HTTP {e.response.status_code}: {e.response.text[:200]}"
    except requests.RequestException as e:
        return f"Error: request to Ollama failed: {e}"
    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()

    # The model Ollama installs as "qwen3:4b" is the reasoning-only Qwen 3 4B
    # build (same image as qwen3:4b-thinking-2507). It reasons even with
    # "think": False, and Ollama then returns the reasoning inside the content,
    # closed by a </think> tag. Keep only what comes after the tag.
    if "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()
    elif data.get("done_reason") == "length" and model_reasons():
        return ("No answer: Qwen used its whole num_predict budget reasoning before it reached "
                "an answer. Raise num_predict, or set llm_model to a non-thinking model such as "
                "qwen3:4b-instruct.")
    return answer or "Error: the model returned an empty response."


_capabilities = {}


def model_reasons():
    """Ask Ollama once whether llm_model is a thinking model (it lists 'thinking' in its capabilities)."""
    if llm_model not in _capabilities:
        try:
            show = requests.post(ollama_url.replace("/api/chat", "/api/show"),
                                 json={"model": llm_model}, timeout=30).json()
            _capabilities[llm_model] = "thinking" in show.get("capabilities", [])
        except requests.RequestException:
            _capabilities[llm_model] = False
    return _capabilities[llm_model]


def answer_question(question, chunks, index, embedder, reranker):
    """Full pipeline for one question."""
    return generate_answer(question, retrieve_and_rerank(question, chunks, index, embedder, reranker))


def run_experiments(embedder, reranker, questions, settings):
    """Step 6: rebuild chunks and index at several settings and answer the same questions."""
    for size, overlap in settings:
        chunks = load_and_split_document(size=size, overlap=overlap)
        index = build_faiss_index(chunks, embedder)
        print(f"\n=== chunk_size={size}, chunk_overlap={overlap}: {len(chunks)} chunks ===")
        for q in questions:
            print(f"\nQ: {q}\nA: {answer_question(q, chunks, index, embedder, reranker)}")


def main():
    embedder = SentenceTransformer(embedding_model_name)
    reranker = CrossEncoder(reranker_model_name)

    if "--experiment" in sys.argv:
        with open("questions.txt", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        run_experiments(embedder, reranker, questions,
                        [(200, 20), (500, 50), (1000, 100), (500, 0), (500, 150)])
        return

    chunks = load_and_split_document()
    index = build_faiss_index(chunks, embedder)
    print(f"Loaded {len(chunks)} chunks. Ask a question, or type exit or quit to stop.")
    while True:
        question = input("\nYour question: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        print("Answer:", answer_question(question, chunks, index, embedder, reranker))


if __name__ == "__main__":
    main()
