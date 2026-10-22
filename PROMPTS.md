# Prompts used

The code was generated from the assignment's two prompts, then checked by running it. Changes made after checking are listed under each prompt.

## text_extractor_web.py

> Write a complete Python script named text_extractor_web.py. Use requests to fetch a hardcoded page URL and Beautiful Soup to parse the HTML. Extract the main readable article text: use the `<article>` element when present; if it isn't present, try the page's main-content container (for Wikipedia, `#mw-content-text .mw-parser-output`). Extract headings, paragraphs, and list items, joining them with blank lines. Save the text to Selected_Document.txt using UTF-8 encoding. Handle request errors, unsuccessful HTTP status codes, and missing or empty article content with clear messages. Return the extracted text on success. Include a main() function and an `if __name__ == "__main__":` block.

Changes after checking the output:

- Added a User-Agent header. Wikipedia refused the request with HTTP 403 without one.
- Removed page furniture before extracting: footnote markers like `[1]`, the `[edit]` links beside headings, the reference list, navigation boxes and sidebars. Extraction stops at the "See also" or "References" heading, because everything after it is links and citations, not article text.
- Each element is read with `get_text(" ")` and its whitespace collapsed, then stray spaces around brackets and punctuation are removed. Reading with `get_text(strip=True)` instead glues linked words to their neighbors.
- Nested list items are skipped, because their text is already included in the parent item.

## rag_app.py

> Write a complete Python script named rag_app.py for a local Retrieval-Augmented Generation (RAG) application.
>
> Put all adjustable parameters together near the beginning of the script, before the functions:
>
> ```
> chunk_size = 500
> chunk_overlap = 50
> embedding_model_name = "sentence-transformers/all-distilroberta-v1"
> reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
> top_k = 20
> top_m = 8
> llm_model = "qwen3:4b"
> ollama_url = "http://localhost:11434/api/chat"
> ```
>
> Use Selected_Document.txt as the source document. Split its text with RecursiveCharacterTextSplitter, using chunk_size and chunk_overlap. Use the Sentence Transformers embedding model to embed the document chunks and user questions. Store the chunk embeddings in a local FAISS IndexFlatL2 index.
>
> When a user asks a question, retrieve up to top_k candidate chunks from FAISS. Use the cross-encoder to rerank the candidates and keep the top_m highest-scoring chunks. Remove duplicate chunks while preserving their order.
>
> Generate answers by sending the question and reranked chunks as context to the Ollama chat API at ollama_url, using the model named by llm_model. Set "stream": False, "think": False, and "options": {"num_predict": 500}. Instruct Qwen to answer only from the supplied context and to say it does not know if the context does not contain the answer. Do not use the OpenAI SDK, an API key, or a .env file.
>
> Include clear error handling for a missing or empty source document, Ollama connection or HTTP errors, and an empty model response. Suppress noisy library logs using logging, transformers.logging, and warnings.
>
> Organize the script into functions for loading and splitting the document, building the FAISS index, retrieving and reranking chunks, and generating answers. Add an interactive loop that accepts questions until the user enters exit or quit, ignoring letter case. Skip blank input.
>
> Include the correct Python main guard: `if __name__ == "__main__":`
>
> Return the complete script in one Python code block, with no partial snippets.

Changes after checking:

- Added `--experiment` mode for Step 6. It rebuilds the chunks and the index at five settings and answers the questions in questions.txt at each one, so the comparison is reproducible from one command.
- Retrieval asks FAISS for `min(top_k, number of chunks)` results, so a document shorter than 20 chunks does not return empty slots.

## Reflection

> Generate five conceptual questions about this RAG system (embedding dimensions, FAISS distance search, chunk overlap, reranking, and prompt design) and answer them from this code and this document.

Every number in the answers was measured by running the code, not taken from the AI's description of it.
