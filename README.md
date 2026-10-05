# rag-blog-qa

A small Retrieval-Augmented Generation (RAG) pipeline: ingest a text document into a Pinecone vector index, then answer questions about it with Claude, grounded in the retrieved chunks.

**Stack:** LangChain (LCEL) · Pinecone (serverless) · `sentence-transformers/all-MiniLM-L6-v2` · Claude Haiku 4.5 · `uv`

## Architecture

```
            INGESTION (ingestion.py, run once)
 ┌──────────┐   ┌────────────────────┐   ┌─────────────────┐   ┌───────────────┐
 │ .txt doc │──▶│ CharacterTextSplit │──▶│ MiniLM-L6-v2    │──▶│ Pinecone      │
 │TextLoader│   │ 1200 chars, 0 lap  │   │ 384-d, L2-norm  │   │ upsert vectors│
 └──────────┘   └────────────────────┘   └─────────────────┘   │ + text/source │
                                                                └───────┬───────┘
                                                                        │
            QUERY (main.py, per question)                               │
 ┌──────────┐   ┌─────────────────┐   ┌────────────────────┐            │
 │ question │──▶│ MiniLM-L6-v2    │──▶│ Retriever (top-3,  │◀───────────┘
 └──────────┘   │ embed query     │   │ cosine similarity) │
                └─────────────────┘   └─────────┬──────────┘
                                                │ 3 chunks
                                                ▼
 ┌────────┐   ┌──────────────┐   ┌───────────────────────────────┐
 │ answer │◀──│ Claude Haiku │◀──│ Prompt: context + question    │
 └────────┘   │ temperature=0│   └───────────────────────────────┘
              └──────────────┘
```

### Ingestion
1. `TextLoader` reads `mediumblog1.txt` into a `Document`.
2. `CharacterTextSplitter` splits on `\n\n`, merging up to ~1200 characters per chunk (no overlap).
3. `HuggingFaceEmbeddings` encodes each chunk to a 384-dim, L2-normalised vector.
4. `PineconeVectorStore.from_documents` upserts vectors with `text` and `source` metadata.

### Query
The chain is composed with LCEL:

```
RunnablePassthrough.assign(context = itemgetter("question") | retriever | format_docs)
  | prompt_template | llm | StrOutputParser()
```

The same embedding model embeds the question, Pinecone returns the 3 nearest chunks, and Claude answers from that context.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) and run `uv sync`.
2. Create a Pinecone **dense** index with **dimension 384** and **cosine** metric (the dimension must match the embedding model).
3. Create a `.env` file:
   ```
   PINECONE_API_KEY=...
   INDEX_NAME=your-index-name
   ANTHROPIC_API_KEY=...
   ```
4. Put the text you want to query in `mediumblog1.txt` (not included in this repo).

## Usage

```bash
uv run ingestion.py   # chunk, embed and upsert the document
uv run main.py        # ask a question (edit `question` in main.py)
```

## Notes
- Ingestion and querying must use the same embedding model, or retrieval will fail or return poor matches.
- Possible improvements: chunk overlap / `RecursiveCharacterTextSplitter`, a relevance-score threshold, reranking, and a prompt that restricts answers to the context.
