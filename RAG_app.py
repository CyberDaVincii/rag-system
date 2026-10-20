"""RAG_app.py: a Retrieval-Augmented Generation system over one document.

Pipeline, in the order the steps were built:
  3.1  quiet the library logs
  3.2  load the OpenAI key from .env
  3.3  set the parameters
  3.4  read the pre-scraped document (Selected_Document.txt)
  3.5  split it into overlapping chunks
  3.6  embed every chunk locally and store the vectors in a FAISS index
  3.7  retrieve the top_k chunks closest to a question
  3.8  re-rank those chunks with a cross-encoder and keep the best top_m
  3.9  send the kept chunks plus the question to the ChatGPT API for the answer
  3.10 ask questions in a loop until "exit" or "quit"

Everything except step 3.9 runs locally and free: the embedding model, the
cross-encoder, and FAISS all run on this machine's CPU.

Run:  python RAG_app.py                interactive questions
      python RAG_app.py --experiment   chunk-size / overlap comparison (Step 4.1)
"""

# ---------------------------------------------------------------------------
# 3.1 Suppress noisy logs
# ---------------------------------------------------------------------------
import logging
import warnings
from transformers import logging as hf_logging

logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()          # the transformers library's own logger
warnings.filterwarnings("ignore")         # hide deprecation and future warnings

# ---------------------------------------------------------------------------
# 3.2 ChatGPT API credentials
# ---------------------------------------------------------------------------
import os
import openai
from dotenv import load_dotenv

load_dotenv()                                     # reads .env from the working folder
openai.api_key = os.getenv("OPENAI_API_KEY")      # never printed, never committed (.env is in .gitignore)

# ---------------------------------------------------------------------------
# 3.3 Parameters
# ---------------------------------------------------------------------------
chunk_size = 500        # characters per chunk
chunk_overlap = 50      # characters shared by neighboring chunks, so a sentence cut at a border survives in one of them
model_name = "sentence-transformers/all-distilroberta-v1"   # bi-encoder: turns text into a 768-number vector
top_k = 20              # how many chunks FAISS returns for a question

# Re-ranking parameters
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # reads question and chunk together and scores relevance
top_m = 8               # how many re-ranked chunks go to the language model

# ---------------------------------------------------------------------------
# 3.4 Read the pre-scraped document
# ---------------------------------------------------------------------------
with open("Selected_Document.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------------------------------------------------------------------------
# 3.5 Split into appropriately sized chunks
# ---------------------------------------------------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The splitter tries the separators in order: paragraph breaks first, then
# line breaks, then spaces, and only cuts mid-word ("") as a last resort.
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = text_splitter.split_text(text)

# ---------------------------------------------------------------------------
# 3.6 Embed the chunks and build the FAISS index
# ---------------------------------------------------------------------------
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks, show_progress_bar=False)   # one vector per chunk
embedding_array = np.array(embeddings, dtype="float32")        # FAISS requires float32

dimension = embedding_array.shape[1]           # 768 for all-distilroberta-v1
faiss_index = faiss.IndexFlatL2(dimension)     # exact search by Euclidean (L2) distance, no approximation
faiss_index.add(embedding_array)


# ---------------------------------------------------------------------------
# 3.7 Retrieval function
# ---------------------------------------------------------------------------
def retrieve_chunks(question, k=top_k):
    """Return the k chunks whose embeddings are closest to the question's embedding."""
    q_vec = embedder.encode([question], show_progress_bar=False)   # embed the question the same way as the chunks
    q_arr = np.array(q_vec, dtype="float32")
    D, I = faiss_index.search(q_arr, k)                             # D = distances, I = chunk positions
    return [chunks[i] for i in I[0] if i != -1]                     # -1 means "no result" when k > number of chunks


# ---------------------------------------------------------------------------
# 3.8 Cross-encoder re-ranker
# ---------------------------------------------------------------------------
from sentence_transformers import CrossEncoder

reranker = CrossEncoder(cross_encoder_name)


def dedupe_preserve_order(items):
    """Drop duplicate chunks, keeping the first one, after normalizing whitespace."""
    seen = set()
    unique = []
    for item in items:
        key = " ".join(item.split())          # collapse runs of spaces and newlines
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    """Score each (question, chunk) pair with the cross-encoder and keep the best m.

    The bi-encoder in retrieve_chunks embeds the question and each chunk
    separately, which is fast but coarse. The cross-encoder reads the question
    and a chunk together, which is slower but judges relevance much better, so
    it only runs on the top_k candidates. No bi-encoder call happens here.
    """
    pairs = [(question, chunk) for chunk in candidate_chunks]
    scores = reranker.predict(pairs)                                  # higher = more relevant
    ranked = sorted(zip(candidate_chunks, scores), key=lambda pair: pair[1], reverse=True)
    best = [chunk for chunk, _ in ranked[:m]]
    return dedupe_preserve_order(best)


# ---------------------------------------------------------------------------
# 3.9 Q&A with ChatGPT
# ---------------------------------------------------------------------------
from openai import OpenAI

client = OpenAI(api_key=openai.api_key)


def build_prompts(question, relevant_chunks):
    """Helper: join the kept chunks into one context block and build both prompts."""
    context = "\n\n".join(relevant_chunks)
    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don’t know."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return system_prompt, user_prompt


def ask_chatgpt(system_prompt, user_prompt):
    """Call the Chat Completions API exactly as the assignment specifies.

    GPT-5 models may reject temperature=0.0 and max_tokens (newer models take
    max_completion_tokens and only the default temperature). If the API answers
    with that kind of parameter error, retry once with the supported settings.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model="gpt-5", messages=messages, temperature=0.0, max_tokens=500
        )
    except openai.BadRequestError as e:
        if "max_tokens" not in str(e) and "temperature" not in str(e):
            raise
        resp = client.chat.completions.create(
            model="gpt-5", messages=messages, max_completion_tokens=500
        )
    return resp.choices[0].message.content.strip()


# Free local fallback. The course-provided key returned "You have no credits
# remaining" (HTTP 429, insufficient_quota) on Oct 19, 2026, so when the API
# call fails the same prompts go to a small open model running on this CPU.
# This keeps the whole pipeline free and locally runnable, the assignment's
# stated objective.
local_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
_local = {}


def ask_local_model(system_prompt, user_prompt):
    """Answer with a local instruction-tuned model, greedy decoding (the equivalent of temperature 0)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if not _local:
        _local["tokenizer"] = AutoTokenizer.from_pretrained(local_model_name)
        _local["model"] = AutoModelForCausalLM.from_pretrained(local_model_name, torch_dtype=torch.float32)
    tokenizer, model = _local["tokenizer"], _local["model"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]          # drop the echoed prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


answer_source = {"last": None}     # which generator produced the most recent answer


def answer_question(question):
    """Retrieve, re-rank, and ask the model to answer only from the retrieved context."""
    candidates = retrieve_chunks(question)                           # step 1: top_k = 20 by vector distance
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)   # step 2: best top_m = 8 by cross-encoder
    system_prompt, user_prompt = build_prompts(question, relevant_chunks)
    try:                                                             # step 3: generate the grounded answer
        answer_source["last"] = "gpt-5"
        return ask_chatgpt(system_prompt, user_prompt)
    except openai.APIError as e:
        if answer_source.get("warned") is None:
            print(f"[ChatGPT API unavailable: {type(e).__name__}. Using local model {local_model_name}.]")
            answer_source["warned"] = True
        answer_source["last"] = local_model_name
        return ask_local_model(system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Step 4.1 experiments: rebuild the index at several chunk settings
# ---------------------------------------------------------------------------
def run_experiments(questions, settings):
    """For each (chunk_size, chunk_overlap), rebuild chunks and index, then answer every question."""
    global chunks, faiss_index
    for size, overlap in settings:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], chunk_size=size, chunk_overlap=overlap
        )
        chunks = splitter.split_text(text)
        vectors = np.array(embedder.encode(chunks, show_progress_bar=False), dtype="float32")
        faiss_index = faiss.IndexFlatL2(vectors.shape[1])
        faiss_index.add(vectors)
        print(f"\n=== chunk_size={size}, chunk_overlap={overlap}: {len(chunks)} chunks ===")
        for q in questions:
            print(f"\nQ: {q}\nA: {answer_question(q)}")


# ---------------------------------------------------------------------------
# 3.10 Interactive loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if "--experiment" in sys.argv:
        EXPERIMENT_QUESTIONS = [line.strip() for line in open("questions.txt", encoding="utf-8") if line.strip()]
        run_experiments(EXPERIMENT_QUESTIONS, [(200, 20), (500, 50), (1000, 100), (500, 0), (500, 150)])
    else:
        print("Enter 'exit' or 'quit' to end.")
        while True:
            question = input("Your question: ")
            if question.lower() in ("exit", "quit"):
                break
            print("Answer:", answer_question(question))
