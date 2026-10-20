# Prompts used

Each section of the code was generated from the assignment's prompts, in this order, and pasted at the bottom of the program as it grew. Where I changed or added to the generated code, the change is listed under the prompt.

## text_extractor.py

**2.2 Web scraper.** Write a Python function that fetches the page at a specific URL (hardcoded in the script). The script should use the requests library to retrieve the page, parse it with BeautifulSoup, extract all `<p>` tags located within a `<div class="mw-parser-output">`, join their text with blank lines, write the result to a file called Selected_Document.txt encoded in UTF-8, print a success or failure message depending on the HTTP status code, and return the extracted text. Include a main() function that calls the scraping function using the hardcoded URL, and an `if __name__ == '__main__':` block that runs the script when executed.

- Added: a User-Agent header (Wikipedia returned HTTP 403 without one).
- Changed: paragraph text is read with `get_text()` and whitespace-collapsed, not `get_text(strip=True)`, which glued linked words together.
- Added: footnote markers are removed before extraction.

## RAG_app.py

**3.1 Suppress noisy logs.** Import logging, transformers.logging (as hf_logging), and warnings; set the log level of langchain.text_splitter and transformers to ERROR, and filter Python warnings.

**3.2 ChatGPT API credentials.** Use python-dotenv to load environment variables from a .env file and read OPENAI_API_KEY into openai.api_key.

**3.3 Parameters.** chunk_size = 500, chunk_overlap = 50, model_name = "sentence-transformers/all-distilroberta-v1", top_k = 20; re-ranking: cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_m = 8.

**3.4 Read the pre-scraped document.** Open Selected_Document.txt in UTF-8 mode and read its contents into a variable text.

**3.5 Split into chunks.** Import and use RecursiveCharacterTextSplitter with separators `["\n\n", "\n", " ", ""]` and the chunk_size and chunk_overlap above to split text into a list chunks.

- Changed: imported from `langchain_text_splitters`, the package the requirements install; the older `langchain.text_splitter` path is deprecated.

**3.6 Embed and build the FAISS index.** Load SentenceTransformer(model_name), encode chunks with the progress bar hidden, convert to a NumPy float32 array, create a FAISS IndexFlatL2 with the right dimension, and add the array.

**3.7 Retrieval function.** retrieve_chunks(question, k=top_k): encode the question with the embedder, convert to float32, search faiss_index for the k nearest neighbors, and return the matching chunks. Variable names q_vec, q_arr, I.

- Added: skip index -1, which FAISS returns when k is larger than the number of chunks. That never happens with this document, but it would with one under about 20 chunks, like the RAG article I tried first.

**3.8 Cross-encoder re-ranker.** Initialize CrossEncoder(cross_encoder_name); write dedupe_preserve_order(items), which removes duplicates while keeping order and normalizing whitespace; write rerank_chunks(question, candidate_chunks, m=top_m), which scores (question, chunk) pairs with reranker.predict, sorts by score, keeps the top m, and dedupes. No bi-encoder calls inside.

**3.9 Q&A with ChatGPT.** answer_question(question): retrieve_chunks, then rerank_chunks, join the kept chunks with blank lines into a context, use the given system prompt, build the user prompt "Context: ... Question: ... Answer:", and call Chat Completions with model="gpt-5", temperature=0.0, max_tokens=500.

- Added: if the API rejects temperature or max_tokens for this model, retry once with max_completion_tokens.
- Added: if the API call fails entirely, the same prompts go to a local model, Qwen2.5-1.5B-Instruct, with greedy decoding. The course key returned "You have no credits remaining" (HTTP 429), so every answer in this repository came from the local model. See the reflection.

**3.10 Interactive loop.** Under `if __name__ == "__main__":`, print the exit instructions and loop on input until the user types exit or quit, printing answer_question for each question.

- Added: `--experiment` flag that runs Step 4.1, rebuilding the chunks and index at five settings and answering the questions in questions.txt at each one.

## Reflection

**4.2 Deep-dive questions.** "Generate five important questions to understand this RAG system, about embedding dimensionality, FAISS search behavior, chunk overlap, and prompt design, and answer each from this code and document." Every number in the answers was measured by running the code.
