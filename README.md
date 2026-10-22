# Local Retrieval-Augmented Generation over "Large language model"

Angela Reinhold · The Artificial Intelligence Ecosystem, Week 4 · project 4.2

A question-answering system that runs entirely on my own machine. It extracts the text of one article, splits it into chunks, embeds them, searches them with FAISS, reranks the best candidates with a cross-encoder, and has Qwen 3 4B answer through Ollama using only the retrieved text. No API key, no cloud model, no cost.

## Files

| File | What it is |
|---|---|
| `Selected_Document.txt` | The cleaned article text the system answers from |
| `text_extractor_web.py` | Extracts the article and writes `Selected_Document.txt` |
| `rag_app.py` | Chunking, embeddings, FAISS index, retrieval, reranking, generation through Ollama, the question loop, and the experiment mode |
| `questions.txt` | The three questions used in the chunk-size experiments |
| `qa_output.txt` | The four questions, answered by both models |
| `experiment_output_qwen3-4b.txt` | Experiment run with `qwen3:4b`, as specified |
| `experiment_output_qwen3-4b-instruct.txt` | Experiment run with `qwen3:4b-instruct` |
| `PROMPTS.md` | The prompts used to generate the code, and what changed after checking it |
| `requirements.txt` | Python dependencies |

## How to run

In WSL Ubuntu, with Ollama installed and the model downloaded (`ollama pull qwen3:4b`):

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python text_extractor_web.py          # writes Selected_Document.txt
python rag_app.py                     # ask questions; exit or quit to stop
python rag_app.py --experiment        # chunk_size and chunk_overlap comparison
```

To try a different model, run `ollama pull <model>` and change `llm_model` at the top of `rag_app.py`.

## The document

I chose the English Wikipedia article [Large language model](https://en.wikipedia.org/wiki/Large_language_model). It covers how LLMs are built and trained, how attention and context windows work, what they can do, and where they fail, including hallucination and bias, so it gives the system questions with clear answers in specific places. `Selected_Document.txt` is that article's text, used under Wikipedia's [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

The extracted document has 190 blocks (headings, paragraphs and list items) and 49,748 characters, which makes 144 chunks at the default settings.

I first tried the Wikipedia article on retrieval-augmented generation itself. With this extractor it came out to 10,556 characters and 32 chunks. The program retrieves 20 chunks per question, so FAISS would have handed over almost two thirds of that article every time, and the retrieval step would barely have narrowed anything. A document this system can actually search needs to be much larger than `top_k` chunks.

**How the extractor cleans the text.** I checked the extracted text against the page and built the extractor to keep only readable article text:

1. **Spaces around links are kept.** Each element is read with `get_text(" ")` and its whitespace collapsed. A common shortcut, `get_text(strip=True)`, glues linked words to their neighbors; on this article it turns the first sentence into "Alarge language model(LLM) is amachine learning model(typically aneural network)". That matters for retrieval: the embedding model's tokenizer reads "large language" as `large`, `language`, but "largelanguage" as the fragments `larg`, `el`, `anguage`. A final pass removes spaces left just inside brackets and before punctuation.
2. **Page furniture is removed.** Footnote markers like `[1]`, the `[edit]` links beside every heading, the reference list, and navigation boxes are dropped before extraction, and extraction stops at the "See also" heading, since everything after it is links and citations. The saved file contains no footnote markers, no edit links, and no fused words.
3. **The request identifies itself.** Wikipedia asks scripts to send a descriptive User-Agent header and returns HTTP 403 without one, so the extractor sends one.

## The pipeline

1. **Chunking.** `RecursiveCharacterTextSplitter` cuts at paragraph breaks first, then line breaks, then spaces, and only inside a word as a last resort. Default: 500 characters with 50 characters of overlap.
2. **Embeddings.** `all-distilroberta-v1` turns each chunk into a 768-number vector, locally.
3. **Index.** The vectors go into a FAISS `IndexFlatL2`, an exact nearest-neighbor index.
4. **Retrieval.** The question is embedded the same way, and FAISS returns the 20 closest chunks.
5. **Reranking.** The cross-encoder `ms-marco-MiniLM-L-6-v2` scores each of the 20 against the question, reading both together, and the best 8 are kept after removing duplicates.
6. **Generation.** The 8 chunks go to Qwen 3 4B through Ollama's local chat API, with an instruction to answer only from that context and to say it does not know otherwise. Thinking is turned off (`"think": False`) and answers are capped at 500 tokens.

## Questions and answers

These answers come from `qwen3:4b-instruct` at the default settings (500-character chunks, 50 overlap). Why this model and not the default `qwen3:4b` is explained in the next section. Every output, for both models, is in `qa_output.txt` and the two experiment files.

**1. What is a context window?**

> A context window is the number of tokens that an LLM considers when processing input. It defines the segment of text that is used as context for a given token, and within this window, the attention mechanism calculates "soft" weights for each token to determine relevance. For example, in a model with a context window of 1k tokens, the model only considers the most recent 1,000 tokens when processing a new token. This window limits the amount of historical information the model can access and use to generate responses.

Mostly grounded. The attention passage and the 1k-token example (GPT-2) are in the article. "Only considers the most recent 1,000 tokens" is the model's own generalization; the article describes the context window but does not phrase it that way.

**2. Why do large language models hallucinate?**

> Large language models hallucinate because they generate syntactically fluent text that appears factually sound but is internally inconsistent with training data or factually incorrect. This occurs partly through memorization of training data combined with extrapolation beyond factual boundaries. Evaluations have shown that models can output verbatim passages from training data when subjected to specific prompting sequences.

Fully grounded. It follows the article's own explanation almost word for word.

**3. When was the transformer architecture introduced, and in which paper?**

> The transformer architecture was introduced in 2017 in the paper "Attention Is All You Need."

Correct and grounded.

**4. Who won the 2022 FIFA World Cup?** (a question the article cannot answer)

> I don't know.

Correct behavior: the answer is not in the document, so the system says so instead of using what the model might know.

## Switching models

The assignment specifies `llm_model = "qwen3:4b"` with `"think": False`. Today that tag installs the reasoning-only build of Qwen 3 4B (details in deep-dive question 5): it reasons before every answer even with thinking turned off, and the reasoning uses up the 500-token cap. The assignment also asks us to practice switching models in Ollama, so I ran the same experiment on `qwen3:4b-instruct`, the same size of Qwen 3 4B without the reasoning step, by changing only `llm_model`.

| Model | Answers completed (of 15) | Time for the 15 |
|---|---|---|
| `qwen3:4b` (as specified) | 5 | 118 s |
| `qwen3:4b-instruct` | 15 | 41 s |

Both ran on the same GPU with the same settings, chunks, and reranked context. With `qwen3:4b`, the transformer question was answered every time, because it needs little reasoning. In the final run, the other two questions ran out of budget at every setting. The result also changed between runs: an earlier run, before a one-line prompt change, completed 8 of 15. `rag_app.py` keeps `qwen3:4b` as the assignment specifies and reports clearly when the budget runs out; to reproduce the answers above, change `llm_model` to `qwen3:4b-instruct`.

## Chunk size and overlap experiments

I ran the same three questions at five settings with `qwen3:4b-instruct`. Reading the answers alone could not tell me whether a difference came from retrieval or from the model, so I also checked what the 8 reranked chunks contained: whether the passage that actually answers the question was there, and at what rank.

| chunk_size / overlap | Chunks | Context sent (chars, Q2) | Q1 evidence | Q2 key sentence | Q3 evidence |
|---|---|---|---|---|---|
| 200 / 20 | 351 | 1,443 | rank 2 | **not retrieved** | rank 1 |
| 500 / 50 | 144 | 3,067 | rank 1 | rank 6 | rank 1 |
| 1000 / 100 | 69 | 5,940 | rank 1 | rank 1 | rank 1 |
| 500 / 0 | 140 | 2,945 | rank 1 | rank 5 | rank 1 |
| 500 / 150 | 148 | 3,328 | rank 1 | rank 5 | rank 1 |

Q2's key sentence is "These hallucinations arise partly through memorization of training data combined with extrapolation beyond factual boundaries."

**What changed and why.**

- **A one-sentence fact does not care about chunk size.** "Attention Is All You Need" ranked first at every setting, and every answer was correct.
- **An explanation needs room, and the system was honest when it did not get it.** At 200 characters, the sentence that explains hallucination was never retrieved, and the model answered "I don't know." instead of guessing. From 500 characters up, the sentence was in the context and every answer named memorization and extrapolation.
- **Bigger chunks kept the explanation together.** At 1,000 characters, the key sentence moved from rank 5 or 6 up to rank 1, because the whole hallucination passage fit in one chunk. The cost is double the context sent to the model, about 5,900 characters instead of about 3,000, which means slower answers and more room for unrelated text.
- **Overlap barely mattered here.** Going from 0 to 150 characters of overlap changed the chunk count from 140 to 148 and moved the key sentence by one rank at most, because most of this article's blocks fit inside one chunk.
- **Small chunks pulled in the wrong neighbor.** At 500 characters with no overlap, the model defined a context window as "the number of tokens in the text corpus." That phrase comes from the article's perplexity formula, which sat in a retrieved chunk next to the context-window passage. The retrieval was right, and the model blended two nearby passages into one wrong sentence.

For this document, 500 to 1,000 characters with a small overlap worked best: big enough to keep an explanation together, small enough to keep unrelated passages out. 200 was too small for anything beyond single facts.

## Five deep-dive questions and AI answers

I asked Claude to generate five questions that would test whether I understand this system, and to answer each one from this code and this document, with every number measured rather than assumed.

### 1. What does each embedding look like, and why does FAISS use L2 distance when most RAG tutorials talk about cosine similarity?

**Answer:** `all-distilroberta-v1` turns each chunk into a vector of 768 numbers, and the FAISS index is built with that dimension. The model's last layer normalizes its output, so every vector has length 1 (two test sentences both measured exactly 1.0). For vectors of length 1, ranking by smallest L2 distance gives exactly the same order as ranking by highest cosine similarity, because squared L2 distance equals 2 minus 2 times the cosine. So `IndexFlatL2` is doing cosine search here without saying so.

### 2. How does FAISS search, and is any accuracy lost?

**Answer:** `IndexFlatL2` is a flat index: it compares the question vector against every stored vector and returns the true nearest neighbors. Nothing is approximated, so no accuracy is lost. The cost is a full scan, which is trivial for 144 chunks and only becomes a problem at millions of vectors, where approximate indexes such as IVF or HNSW trade a little accuracy for speed.

### 3. Why retrieve 20 chunks and then keep only 8?

**Answer:** The two models do different jobs. The bi-encoder embeds the question and each chunk separately, so it can search everything quickly but only judges rough similarity. The cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question and one chunk together and judges relevance far more precisely, but it has to run once per pair, so it is too slow to run on every chunk. Retrieving 20 by vector distance and then reranking those 20 gets most of the cross-encoder's precision at a fraction of its cost. Its scores are raw relevance scores, not probabilities: a sentence that answers "What is a context window?" scored 9.69, and an unrelated sentence about seahorses scored -11.22.

### 4. What does chunk overlap actually do in this document, and can a chunk ever be too long?

**Answer:** Overlap only applies when the splitter has to cut inside a block of text. In this document 161 of the 190 blocks are 500 characters or shorter (the median is 192), so most chunks end at a block boundary and share nothing with their neighbors. Raising the overlap from 0 to 150 characters only changed the chunk count from 140 to 148. On length: the embedding model reads at most 512 tokens and silently drops the rest. The longest chunk was 152 tokens at chunk_size 500 and 280 tokens at 1,000, so nothing here was cut off, but chunks of roughly 2,000 characters or more could start losing their endings.

### 5. How does the prompt design shape what Qwen says?

**Answer:** Three settings matter, and testing them showed each one doing something different from what I expected.

- **The instruction to stay in the context.** The system prompt tells Qwen to answer only from the retrieved text. It held both when retrieval found the answer and when it did not: at 200-character chunks, where the sentence explaining hallucination was not retrieved, the non-reasoning model said the context did not explain the cause instead of inventing one.
- **The exact wording of the fallback.** I tested two wordings. With "say that you don't know", the model replied with the literal sentence "You don't know." With the reply quoted, `reply exactly: "I don't know."`, it answered "I don't know." every time. Small models follow instructions very literally, so the prompt quotes the exact reply.
- **`"think": False` and the 500-token cap.** The model Ollama installs as `qwen3:4b` is the reasoning-only build of Qwen 3 4B; it has the same image ID as `qwen3:4b-thinking-2507` and lists thinking as always on. It reasons no matter what `think` is set to. With `"think": False`, Ollama returns that reasoning inside the answer, ending with a `</think>` tag; with `"think": True`, it returns the reasoning in a separate field. Either way the reasoning counts against the 500-token cap, and on harder questions the model ran out before it reached an answer. `rag_app.py` strips anything before `</think>` and reports clearly when the budget ran out.

## Working with the AI

I built this with Claude: the code came from the assignment's prompts, and I directed the experiments and checked every result by running it. The decisions that shaped the project:

1. **Building it fully local.** The project uses the updated assignment's setup: Qwen 3 4B through Ollama, with no API key and no cloud model. Everything, from extraction to answers, runs on my own machine and GPU.
2. **Comparing two Qwen 3 4B builds.** The model installed as `qwen3:4b` reasons before every answer, so I had us confirm what it was (`ollama show` lists thinking as always on, and its image ID matches `qwen3:4b-thinking-2507`) and then run the same experiment on `qwen3:4b-instruct`. Switching models in Ollama is part of the assignment, and the comparison is in the results above.
3. **Choosing a document big enough to search.** I measured the RAG article first: 32 chunks against 20 retrieved per question would have handed over almost two thirds of it every time. The "Large language model" article gives 144 chunks, so retrieval actually has to choose.
4. **Reading the extracted text, not just running the extractor.** The cleaning rules in this report came from comparing the saved text against the page and counting what was left, not from the extractor running without errors.
5. **Checking answers against the article.** The system prompt asks Qwen to answer only from the context, but the only way to know whether it did is to check. That is why the results section marks each answer as grounded, partly grounded, or wrong, and measures what each chunk setting actually retrieved.

What worked best was treating every output as something to verify. The chunk counts, evidence ranks, token lengths, and cleaning counts in this report all came from running the code, not from the AI's description of it.
