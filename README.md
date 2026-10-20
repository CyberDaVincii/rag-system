# Retrieval-Augmented Generation over "Large language model"

Angela Reinhold · The Artificial Intelligence Ecosystem, Week 4 · project 4.2

A question-answering system over one document: it scrapes the text, splits it into chunks, embeds them locally, searches them with FAISS, re-ranks the best candidates with a cross-encoder, and has a language model answer using only the retrieved text.

## Files

| File | What it is |
|---|---|
| `Selected_Document.txt` | The cleaned article text the system answers from |
| `text_extractor.py` | Scrapes the article and writes `Selected_Document.txt` |
| `RAG_app.py` | Chunking, embeddings, FAISS index, retrieval, re-ranking, generation, the question loop, and the experiment mode |
| `questions.txt` | The three questions used for the chunk-size experiments |
| `experiment_output.txt` | The full output of the experiment run |
| `PROMPTS.md` | The prompts used to generate each part of the code |
| `requirements.txt` | Python dependencies |

`.env` (the API key) and `venv/` are listed in `.gitignore` and are not in this repository.

## How to run

Inside WSL, in a Python 3.13 virtual environment:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python text_extractor.py            # writes Selected_Document.txt
python RAG_app.py                   # ask questions; type exit or quit to stop
python RAG_app.py --experiment      # chunk-size and overlap comparison
```

The CPU build of PyTorch is installed first because the default Linux build pulls several gigabytes of GPU libraries this project never uses.

## The document

I chose the English Wikipedia article [Large language model](https://en.wikipedia.org/wiki/Large_language_model). It covers how LLMs are built and trained, how attention and context windows work, what they can do, and where they fail, including hallucination and bias, so it gives the system questions with clear answers in specific places. `Selected_Document.txt` is that article's paragraph text, used under Wikipedia's [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

My first choice was the Wikipedia article on retrieval-augmented generation itself, but it was too short for this system. It came out to 8,871 characters, about 20 chunks at the default chunk size, and the program retrieves 20 chunks per question. FAISS would have returned the entire document every time, and the retrieval step would have done nothing. The LLM article is 46,030 characters in 110 paragraphs, which makes 130 chunks at the default settings, so retrieval has to choose.

**Cleaning.** `text_extractor.py` keeps only the paragraph text inside Wikipedia's main content area and joins the paragraphs with blank lines. It does two things the course example does not:

1. It reads each paragraph with `get_text()` and collapses the whitespace, instead of `get_text(strip=True)`. `strip=True` trims each piece of text inside a paragraph and glues the pieces together without spaces, so every linked word fuses with its neighbors. Run the course example's extraction on this article and the first sentence comes out as "Alarge language model(LLM) is anAI model(typically aneural network) trained on a vast amount of text fornatural language processingtasks". That damages retrieval for exactly the linked terms, which are usually the important ones: the embedding model's tokenizer reads "large language" as `large`, `language`, but "largelanguage" as the fragments `larg`, `el`, `anguage`.
2. It removes footnote markers such as `[1]` and `[citation needed]` before reading the text. The course example's extraction leaves 191 of them in this article; the saved file here has none.

It also sends a descriptive User-Agent header. Without one, Wikipedia refused the request with HTTP 403.

## The pipeline

1. **Chunking.** `RecursiveCharacterTextSplitter` cuts at paragraph breaks first, then line breaks, then spaces, and only mid-word as a last resort. Default: 500 characters with 50 characters of overlap.
2. **Embeddings.** `all-distilroberta-v1` turns each chunk into a 768-number vector, locally.
3. **Index.** The vectors go into a FAISS `IndexFlatL2`, an exact nearest-neighbor index.
4. **Retrieval.** A question is embedded the same way, and FAISS returns the 20 closest chunks.
5. **Re-ranking.** The cross-encoder `ms-marco-MiniLM-L-6-v2` scores each of those 20 against the question, reading both together, and the best 8 are kept after removing duplicates.
6. **Generation.** The 8 chunks become the context in a prompt that tells the model to answer only from the context and to say it does not know otherwise.

## Questions and answers

These are the system's answers at the default settings (500-character chunks, 50 overlap). The full output for every setting is in `experiment_output.txt`. Every answer was written by the local model, Qwen2.5-1.5B-Instruct, because the course API key had no credits (see "Working with the AI").

**1. What is a context window?**

> A context window refers to the range of tokens considered when determining relevance between tokens within the scope of the context window. In the context provided, this concept is discussed in relation to how attention mechanisms calculate "soft" weights for each token, more specifically for its embedding, using multiple attention heads. The context window size affects the effectiveness of these calculations, with smaller windows potentially leading to less accurate results due to limited contextual information.

Partly grounded. The article never defines a context window; it only mentions one while explaining attention, and the answer leans on that passage. The opening definition is circular.

**2. Why do large language models hallucinate?**

> Large language models hallucinate due to several factors: 1. Overfitting ... 2. Memorization and Extrapolation ... 3. Evaluation Metrics ... 4. Mechanisms of Language Learning ... 5. Lack of Context Understanding ...

Mixed. "Memorization and extrapolation" is the article's own explanation: hallucinations "arise partly through memorization of training data combined with extrapolation beyond factual boundaries." "Overfitting" and "evaluation metrics" are not. The article mentions overfitting only to explain why models are evaluated by perplexity, and the model stitched that neighboring passage into a list of causes. The other items are general knowledge the model brought in on its own, despite the instruction to answer only from the context.

**3. When was the transformer architecture introduced, and in which paper?**

> The transformer architecture was introduced in the 2017 NeurIPS conference, specifically in the paper titled "Attention Is All You Need." This paper, authored by Google researchers, aimed to improve upon previous sequential-to-sequence technologies like seq2seq and was heavily influenced by the attention mechanisms developed by Bahdanau et al. in 2014.

Correct and fully grounded. Every detail, including Bahdanau, is in the article.

## Chunk size and overlap experiments

I ran the same three questions at five settings. Reading the answers alone could not tell me whether a difference came from retrieval or from the language model, so I also checked what the 8 re-ranked chunks contained at each setting: whether the passage that actually answers the question was in the context, and where it ranked.

| chunk_size / overlap | Chunks | Context sent (chars, Q2) | Q1 evidence | Q2 key sentence | Q3 evidence |
|---|---|---|---|---|---|
| 200 / 20 | 294 | 1,481 | rank 1 | **not retrieved** | rank 1 |
| 500 / 50 | 130 | 3,006 | rank 1 | rank 4 | rank 1 |
| 1000 / 100 | 64 | 5,893 | rank 1 | rank 4 | rank 1 |
| 500 / 0 | 126 | 2,956 | rank 1 | rank 4 | rank 1 |
| 500 / 150 | 132 | 3,280 | rank 1 | rank 4 | rank 1 |

Q2's key sentence is "memorization of training data combined with extrapolation beyond factual boundaries."

**What changed and why.**

- **A one-sentence fact does not care about chunk size.** "Attention Is All You Need" ranked first at every setting, and all five answers were correct.
- **An explanation needs room.** At 200 characters, the sentence that explains hallucination was not among the 8 chunks the model saw, and the answer fell back on overfitting, synthetic data, and the model's own guesses. It also ran out of its 300-token answer budget mid-list. From 500 characters up, the key sentence was in the context and every answer named memorization and extrapolation.
- **Bigger is not better.** At 1,000 characters the context doubled to about 5,900 characters and pulled in a passage about token bias in multiple-choice tests. The model then listed "token bias" as a cause of hallucination, which the article never says.
- **Overlap barely mattered here.** Going from 0 to 150 characters of overlap changed the chunk count from 126 to 132 and moved nothing in the rankings, because most of this article's paragraphs fit inside one chunk. Overlap would matter more for text with long paragraphs.
- **Some differences were the generator, not retrieval.** For Q1, the sentence with GPT-2's "context window of only 1k tokens" was in the context at every setting, but only the 500/150 answer used it. Retrieval gave the model the same evidence five times, and a 1.5-billion-parameter model used it differently each time.

For this document, 500 characters with a small overlap was the best setting: big enough to keep an explanation together, small enough to keep unrelated passages out.

## Five deep-dive questions and AI answers

I asked Claude to generate five questions that would test whether I understand this system, and to answer each one from this code and this document, with every number measured rather than assumed.

### 1. What does each embedding look like, and why does FAISS use L2 distance when most RAG tutorials talk about cosine similarity?

**Answer:** `all-distilroberta-v1` turns each chunk into a vector of 768 numbers, and the FAISS index is built with that dimension (`faiss_index.d` printed 768). The model's last layer is a Normalize step, so every vector has length 1 (two test sentences both measured exactly 1.0). For unit-length vectors, ranking by smallest L2 distance gives exactly the same order as ranking by highest cosine similarity, because squared L2 distance equals 2 minus 2 times the cosine. So `IndexFlatL2` is doing cosine search here without saying so.

### 2. How does FAISS search, and is any accuracy lost?

**Answer:** `IndexFlatL2` is a flat index: it compares the question vector against every stored vector and returns the true nearest neighbors. No accuracy is lost, because nothing is approximated. The cost is a full scan, which is fine for 130 chunks and would only become a problem at millions of vectors, where approximate indexes such as IVF or HNSW trade a little accuracy for speed.

### 3. Why retrieve 20 chunks and then keep only 8?

**Answer:** The two models do different jobs. The bi-encoder embeds the question and each chunk separately, so it can search everything quickly but only judges rough similarity. The cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question and one chunk together and judges relevance much more precisely, but it has to run once per pair, so it is too slow to run on every chunk. Retrieving 20 by vector distance and then re-ranking those 20 gets most of the cross-encoder's precision at a fraction of its cost. Its scores are raw relevance logits, not probabilities: a matching sentence about context windows scored 9.69, and an unrelated sentence about seahorses scored -11.22.

### 4. Does a chunk ever get cut off by the embedding model?

**Answer:** No. The model reads at most 512 tokens and silently drops anything past that. I measured every chunk at every setting I tested: the longest chunk at chunk_size 1000 was 247 tokens, and at the default 500 it was 146 tokens. A chunk size around 2,000 characters or more could start to be truncated, and the end of those chunks would stop counting toward their embedding.

### 5. What does chunk overlap actually do in this document?

**Answer:** Less than I expected. Overlap only applies when the splitter has to cut inside a paragraph. In this article 81 of the 110 paragraphs are 500 characters or shorter (the median is 352), so most chunks end at a paragraph break and share nothing with their neighbors. Raising the overlap from 0 to 150 characters only changed the chunk count from 126 to 132. Overlap matters more for documents with long paragraphs, such as PDFs extracted without paragraph breaks.

## Working with the AI

I built this with Claude, following the assignment's prompts step by step, and checked each piece by running it. These are the moments that taught me the most.

1. **The API key had no credits.** The code ran cleanly up to the last step, and then the course key returned "You have no credits remaining" (HTTP 429, insufficient_quota). The assignment's objective says the system should use only free, locally runnable components, so instead of stopping there we added a fallback: the program still calls ChatGPT first, exactly as the prompt specifies, and if that call fails it sends the same system and user prompts to Qwen2.5-1.5B-Instruct running on my CPU. That kept the whole pipeline free and working. One thing I could not test because of the missing credits is whether `gpt-5` accepts the prompt's `temperature=0.0` and `max_tokens=500`; newer OpenAI models expect `max_completion_tokens` instead, so the code retries with that if the API rejects the original settings.
2. **My first document was too small.** The Wikipedia article on RAG made about 20 chunks, and the system retrieves 20, so retrieval would have returned the whole document for every question and the FAISS step would have done nothing. Claude caught it by comparing the chunk count to `top_k` before we ran a single question. I switched to the much longer "Large language model" article.
3. **The course example's scraper glued words together.** Its extraction call turned "a large language model" into "Alarge language model" everywhere a word was a link, and left 191 footnote markers in the text. We fixed both before embedding anything, and checked the fix by counting what was left.
4. **A grounded prompt is not a grounded answer.** The system prompt tells the model to answer only from the context. When I checked the answers against the article, the small model still blended in its own knowledge and stitched neighboring passages into claims the article never makes, like listing overfitting as a cause of hallucination. Retrieval did its job, and the weak link was generation. That is why I measured what each setting retrieved instead of judging the experiments by how good the answers sounded.

What worked best was the same thing that worked in the image classification project: treating every output as something to check. The chunk counts, the token lengths, the 191 footnotes, and the evidence ranks in the experiment table all came from running the code, not from the AI's description of it.
