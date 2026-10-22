# Local RAG over the "Large language model" article

Angela Reinhold · The Artificial Intelligence Ecosystem, Week 4 · project 4.2

This system answers questions about one Wikipedia article, and it runs entirely on my own laptop. It pulls the article's text, cuts it into chunks, embeds them, searches them with FAISS, reranks the best matches with a cross-encoder, and has Qwen 3 4B answer through Ollama using only what it retrieved. No API key, no cloud model.

## Files

| File | What it is |
|---|---|
| `Selected_Document.txt` | The cleaned article text |
| `text_extractor_web.py` | Pulls the article and writes `Selected_Document.txt` |
| `rag_app.py` | The RAG app: chunking, embeddings, FAISS, reranking, Ollama, the question loop, and an experiment mode |
| `questions.txt` | The three questions I used for the chunk experiments |
| `qa_output.txt` | My four questions, answered by both models |
| `experiment_output_qwen3-4b.txt` | Experiment run with `qwen3:4b` |
| `experiment_output_qwen3-4b-instruct.txt` | Experiment run with `qwen3:4b-instruct` |
| `PROMPTS.md` | The prompts I used and what I changed after testing |
| `requirements.txt` | Python packages |

## How to run

In WSL Ubuntu with Ollama installed:

```
ollama pull qwen3:4b
ollama pull qwen3:4b-instruct
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python text_extractor_web.py      # writes Selected_Document.txt
python rag_app.py                 # ask questions, type exit or quit to stop
python rag_app.py --experiment    # runs the chunk size and overlap comparison
```

To switch models, change `llm_model` at the top of `rag_app.py`.

## The document

I picked the Wikipedia article [Large language model](https://en.wikipedia.org/wiki/Large_language_model). It explains how LLMs are trained, how attention and context windows work, and where they go wrong, like hallucination, so there are real answers sitting in specific places. The text is used under Wikipedia's [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

After cleaning, it's 190 blocks of headings, paragraphs, and list items, 49,748 characters, and 144 chunks at the default settings.

I almost used the Wikipedia article on RAG itself, but I measured it first. It only makes 32 chunks, and the app pulls 20 per question, so it would hand over most of the article every time and the search would barely matter. The LLM article is big enough that retrieval actually has to choose.

**Cleaning the text.** I compared the saved text against the page and built the extractor to keep only the article:

- **Linked words keep their spaces.** The common shortcut `get_text(strip=True)` glues linked words together. On this article it turns the first sentence into "Alarge language model(LLM) is amachine learning model(typically aneural network)". That hurts search, because the embedding model reads "largelanguage" as the pieces `larg`, `el`, `anguage`. My extractor reads with `get_text(" ")` and cleans up the spacing after.
- **Page clutter comes out.** Footnote markers like `[1]`, the `[edit]` links, the reference list, and navigation boxes are removed, and it stops at "See also", because everything after that is links and citations.
- **It identifies itself.** Wikipedia blocks scripts that don't send a User-Agent (HTTP 403), so the extractor sends one.

## How it works

1. **Chunking.** `RecursiveCharacterTextSplitter` cuts at paragraph breaks first and only cuts inside a word as a last resort. Default is 500 characters with 50 overlapping.
2. **Embeddings.** `all-distilroberta-v1` turns every chunk into 768 numbers.
3. **Index.** The vectors go into a FAISS `IndexFlatL2`, which checks every chunk exactly.
4. **Retrieval.** The question gets embedded the same way, and FAISS returns the 20 closest chunks.
5. **Reranking.** The cross-encoder `ms-marco-MiniLM-L-6-v2` reads the question and each of those 20 together, and the best 8 stay.
6. **Answering.** The 8 chunks go to Qwen through Ollama with an instruction to answer only from them and say "I don't know." otherwise. The request sets `"think": False` and a 500-token limit, as the assignment specifies. On `qwen3:4b` that setting doesn't actually stop it from reasoning, which is why I compared two models below.

## My questions and the answers

These come from `qwen3:4b-instruct` at the default settings. Both models' answers are in `qa_output.txt`.

**1. What is a context window?**

> A context window is the number of tokens that an LLM considers when processing input. It defines the segment of text that is used as context for a given token, and within this window, the attention mechanism calculates "soft" weights for each token to determine relevance. For example, in a model with a context window of 1k tokens, the model only considers the most recent 1,000 tokens when processing a new token. This window limits the amount of historical information the model can access and use to generate responses.

Mostly right. The attention part and the 1k-token example are from the article. "Only considers the most recent 1,000 tokens" is the model adding its own wording.

**2. Why do large language models hallucinate?**

> Large language models hallucinate because they generate syntactically fluent text that appears factually sound but is internally inconsistent with training data or factually incorrect. This occurs partly through memorization of training data combined with extrapolation beyond factual boundaries. Evaluations have shown that models can output verbatim passages from training data when subjected to specific prompting sequences.

Fully grounded. It follows the article almost word for word.

**3. When was the transformer architecture introduced, and in which paper?**

> The transformer architecture was introduced in 2017 in the paper "Attention Is All You Need."

Correct.

**4. Who won the 2022 FIFA World Cup?** I asked this on purpose, because the article doesn't say.

> I don't know.

That's what I wanted. To make sure it was really holding back, I asked the same model with no document, three times. It said Argentina every time, which is right, but twice it put the final on November 18, 2022. The final was December 18. So the model knew the answer and still stuck to the document, and when it answered from memory it got a detail wrong. That test is in `qa_output.txt`.

## Comparing two models

The assignment uses `qwen3:4b`. The version Ollama installs under that name today reasons before every answer, even with thinking set to off, and the reasoning eats the 500-token limit before the answer comes out. The assignment also asks us to practice switching models, so I ran the same experiment on `qwen3:4b-instruct`, the same size of Qwen without the reasoning step.

| Model | Answers completed (of 15) | Time |
|---|---|---|
| `qwen3:4b` | 5 | 118 s |
| `qwen3:4b-instruct` | 15 | 41 s |

Same GPU, same chunks, same context. `qwen3:4b` always got the transformer question, which takes almost no reasoning, but it ran out of tokens on the other two. It also varies from run to run: an earlier run finished 8 of 15. The app keeps `qwen3:4b` as the assignment says and tells you plainly when it runs out of tokens.

## Chunk size and overlap

I ran my three questions at five settings with `qwen3:4b-instruct`. Reading the answers alone couldn't tell me whether a difference came from the search or from the model, so I also checked whether the passage that answers each question made it into the 8 chunks, and at what rank.

| chunk_size / overlap | Chunks | Context sent for Q2 (chars) | Q1 evidence | Q2 key sentence | Q3 evidence |
|---|---|---|---|---|---|
| 200 / 20 | 351 | 1,443 | rank 2 | **not retrieved** | rank 1 |
| 500 / 50 | 144 | 3,067 | rank 1 | rank 6 | rank 1 |
| 1000 / 100 | 69 | 5,940 | rank 1 | rank 1 | rank 1 |
| 500 / 0 | 140 | 2,945 | rank 1 | rank 5 | rank 1 |
| 500 / 150 | 148 | 3,328 | rank 1 | rank 5 | rank 1 |

The key sentence for Q2 is "These hallucinations arise partly through memorization of training data combined with extrapolation beyond factual boundaries."

What I learned from it:

- **A one-line fact doesn't care about chunk size.** "Attention Is All You Need" came up first every time, and every answer was right.
- **An explanation needs room.** At 200 characters, the sentence explaining hallucination never made it in, and the model said "I don't know." instead of making something up. That's the honest answer when the search comes up empty. From 500 up, the sentence was there and every answer used it.
- **Bigger chunks kept the explanation together.** At 1,000, the whole hallucination passage fit in one chunk and jumped to rank 1. The tradeoff is twice as much text going to the model, which is slower and leaves more room for unrelated passages.
- **Overlap barely mattered.** Going from 0 to 150 overlap only changed the chunk count from 140 to 148, because most of this article's paragraphs fit in one chunk anyway.
- **A neighbor can sneak in.** At 500 with no overlap, the model called a context window "the number of tokens in the text corpus." That phrase is from the perplexity formula, which was in one of the retrieved chunks. The search was fine, and the model blended two nearby passages into one wrong sentence.

For this article, 500 to 1,000 characters with a little overlap worked best. 200 was too small for anything more than single facts.

## Five deep-dive questions

I asked Claude for five questions that test whether I understand how this system works, and had it answer them from this code and this document, using measured numbers.

**1. What do the embeddings look like, and why does FAISS use L2 distance instead of cosine similarity?**

Each chunk becomes 768 numbers. The model normalizes its output, so every vector has a length of 1, and both test sentences I checked measured exactly 1.0. When every vector has length 1, sorting by smallest L2 distance gives the same order as sorting by highest cosine similarity, because squared L2 distance equals 2 minus 2 times the cosine. So `IndexFlatL2` is really doing cosine search.

**2. How does FAISS search, and is any accuracy lost?**

`IndexFlatL2` compares the question to every stored vector, so it always finds the true closest matches and nothing is lost. That full check is instant for 144 chunks. It only gets slow at millions of vectors, which is when people switch to approximate indexes like IVF or HNSW.

**3. Why retrieve 20 chunks and keep only 8?**

The two models do different jobs. The embedding model looks at the question and each chunk separately, which is fast but rough. The cross-encoder reads the question and a chunk together, which is much more accurate but too slow to run on everything. Searching wide with FAISS and then reranking 20 gets most of the accuracy at a small cost. Its scores aren't percentages: a sentence answering "What is a context window?" scored 9.69, and an unrelated sentence about seahorses scored -11.22.

**4. What does overlap really do here, and can a chunk be too long?**

Overlap only matters when the splitter has to cut through the middle of a paragraph. In this article, 161 of the 190 blocks are 500 characters or shorter (the median is 192), so most chunks end at a paragraph break and share nothing. That's why 0 to 150 overlap barely changed anything. As for length, the embedding model reads at most 512 tokens and quietly drops the rest. My longest chunk was 152 tokens at 500 characters and 280 at 1,000, so nothing got cut, but chunks around 2,000 characters could start losing their endings.

**5. How does the prompt shape what Qwen says?**

- **Staying in the context.** The prompt tells Qwen to answer only from the retrieved text. It held up even when the search missed: at 200-character chunks, where the hallucination sentence wasn't retrieved, it answered "I don't know." instead of guessing.
- **The exact wording matters.** I tested two versions of the fallback. With "say that you don't know", the model literally replied "You don't know." With the reply quoted, `reply exactly: "I don't know."`, it said "I don't know." every time. Small models take instructions very literally.
- **Thinking and the token limit.** `qwen3:4b` from Ollama is the reasoning-only build. It has the same image ID as `qwen3:4b-thinking-2507` and lists thinking as always on, so `"think": False` doesn't stop it. Its reasoning comes back inside the answer ending in a `</think>` tag, and it counts against the 500-token limit, so on harder questions it runs out before it answers. The app removes everything before `</think>` and says plainly when it ran out.
