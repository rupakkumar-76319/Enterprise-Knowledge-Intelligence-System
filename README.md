Harness: In LangChain and AI engineering, a harness refers to all the code, configuration, tools, and execution logic that surround a language model to turn it into a working agent

data\document\Persuasion.pdf

Stage 1: Basic working RAG foundation — JSON-based storage, ingestion, chunking, embeddings/retrieval, answer generation.
Stage 2: Improve Stage 1's retrieval quality — hybrid search, BM25, multi-query, reranking, etc.
Stage 3: Improve context and answer intelligence — context compression, better context selection, citations, memory, etc.
Stage 4: Turn the accumulated system into a production-ready RAG platform — API, authentication, caching, monitoring, security, scalability, etc.
Stage 5: Extend the production system into an agentic enterprise knowledge system — LangGraph, tools, planning, multi-step reasoning, evaluation, continuous updates.

Stage 1 — Basic Document RAG: Document Knowledge Assistant -- Done
    Document ingestion
    Document loading
    Chunking
    Embeddings
    Store processed data in JSON files
    Basic retrieval from the stored JSON data
    LLM-generated answers
Stage 2 — Advanced Retrieval RAG: Intelligent Retrieval Engine
    Better chunking
    Metadata filtering
    Hybrid search / BM25
    Query rewriting
    Multi-query retrieval
    Reranking
Stage 3 — Context-Aware RAG: Context Intelligence Engine
    Connect/use the Stage 1 JSON data
    Context compression
    Parent-child retrieval
    Document-aware prompts
    Memory
    Citations
    Better-quality context
Stage 4 — Production RAG Platform: Production-Ready RAG API
    FastAPI
    Authentication
    Caching
    Retries
    Monitoring & logging
    Security
    Async processing
    Token/API-cost optimization
Stage 5 — Agentic Enterprise RAG: Enterprise Knowledge Intelligence System
    LangGraph
    Planning & tool use
    Multi-step search
    Answer evaluation
    Continuous knowledge updates
    Multiple enterprise knowledge sources

So in stage two, First we go for text splitter or we can say chunk strategy So first we try on the recursive method and in this recursive method we got total page is 345 and the total chunk size is 638. Now we go for the Character Text splitter and we got total page is 345 and total chunk is 332. Now we go for token text splitter and we got total page is 345 and total chunk is 332. This is the most important thing like "is_separator_regex" not accept by token text splitter.

---

## Stage 1 Review Notes (temporary)


My questions from finishing Stage 1, in the order I asked them, with short answers. The numbers come from testing with Persuasion. Delete this section after reviewing.

**1. What is the error in my terminal?**
Three `FileNotFoundError`s, each from a missing file. The LangChain warnings did not cause any failure.
- `vectors.json`: it was missing at that moment. Running `load_data` and `embeding_chunk` created it again.
- `question_vectors.json`: nothing creates it since I removed the save step from `user_Question_embedding.py`.
- `retrieval.json`: only `retrieval.py` writes it, and that file crashes first.

**2. Where did my project get stuck?**
On the old step-by-step route. The new `generation.py` no longer creates the files that route needs, so each old step waits for a file from a step that no longer makes it.

**3. Was the old version wrong if I had to run everything each time?**
Yes. The old `generation.py` asked my question but took its context from the saved `retrieval.json`, which belonged to an earlier question. To get it right, I had to run three commands and type the question twice. On a website, users could also overwrite each other's files.

**4. Is my workflow correct? (PDF → fmjl → vectors, then question → embed → top 3 → prompt → Gemini → answer)**
Yes, this is the standard RAG design. The PDF is embedded once, and only the question is embedded each time. Gemini does not search the PDF. It only reads the chunks I send, so wrong chunks mean a wrong or missing answer.

**5. Does the problem come from fmjl?**
No, the fmjl file is fine. `TextLoader` read it as plain text, so `hash`, `bbox` and `noise` were embedded with the book. 1079 of 1282 chunks had raw JSON.

**6. Can we use `lazy_load()`?**
Yes, in my own loader. `TextLoader.lazy_load()` still gives one huge Document. One Document per page works best here: 873 chunks, and each keeps its page number. It saves little memory on a small file like this one.

**7. What is `FmjlLoader`, and why was it made?**
It reads the fmjl file line by line, keeps the `md` text from `paragraph` and `heading` elements, skips the rest, and gives back one Document per page. Result: 332 Documents, 873 chunks, no raw JSON.

**8. Can I put it in `base.py` and import it anywhere?**
Yes, with `from Developer.spilliting.base import FmjlLoader`. Run commands from the project root with `python -m`.

**9. Why did I get `TypeError: unexpected keyword argument 'encoding'`?**
`encoding="utf-8"` was left over from the `TextLoader` line. Use `FmjlLoader(file_path)`. `base.py` already opens the file as UTF-8.

**10. What should I run next after rebuilding?**
`generation`, once per test question. With top 3, the book name and author were correct, Sir Walter was correct but thin, and Lady Russell and Anne's marriage were refused.

**11. Does "I do not have sufficient information..." come from the API key?**
No. That sentence is Gemini's reply, and it is written in my prompt in `generation.py`. A bad key crashes instead, with `ClientError: 400 ... API key not valid`. The sentence means the chunks did not clearly contain the answer. The passages that introduce Lady Russell rank 14th and 19th, and the one that gives Anne's marriage (page 332) ranks 39th. With `[:50]`, Lady Russell and Sir Walter got much fuller answers.

**12. Is Stage 1 finished?**
Yes. Every Stage 1 item in this README now works together. Still open:
- Test "Who does Anne Elliot marry?" with `[:50]`, because it has not been run yet.
- Gemini can still make mistakes. It said Lady Russell is Anne's god-daughter, but Anne is hers. Check answers against the book.
- Before Stage 2: commit the work, save the test questions and their correct answers, and build reranking first.

**13. Why do `retrieval.py` and `top_k_retrieval.py` give errors when I can't see a fault?**
They have no coding mistake. They open files that do not exist, and Python only finds that out when the code runs. A traceback shows where the code crashed, not the cause, so `storage.py` looks guilty but is fine. `top_k_retrieval.py` is no longer needed, because `generation.py` picks the top chunks itself.

**14. Is the main block of `retrieval.py` no longer used?**
Yes, and it cannot work without `question_vectors.json`. Importing the file skips that block, so it causes no harm. Keep `cosine_similarties`, because `generation.py` uses it.

**15. If I change something, do I only need to run `generation.py`?**
Partly. `generation.py` reads the saved `chunks.json` and `vectors.json`, not the PDF.

| What I change | What to run |
|---|---|
| Just asking a new question | `generation` |
| Prompt, top-k, temperature or Gemini model | `generation` |
| Chunk size, or `FmjlLoader` | `load_data` → `embeding_chunk` → `generation` |
| Embedding model (change it in both `embeding_chunk.py` and `user_Question_embedding.py`) | `embeding_chunk` → `generation` |
| A new or updated PDF | `fmjl` → `load_data` → `embeding_chunk` → `generation` |

Rule: if the change affects what is stored, rebuild first. If it only affects how a question gets answered, just run `generation`.