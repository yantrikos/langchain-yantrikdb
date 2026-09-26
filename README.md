# langchain-yantrikdb

[![PyPI](https://img.shields.io/pypi/v/langchain-yantrikdb)](https://pypi.org/project/langchain-yantrikdb/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-yantrikdb)](https://pypi.org/project/langchain-yantrikdb/)
[![Downloads](https://img.shields.io/pypi/dm/langchain-yantrikdb)](https://pypi.org/project/langchain-yantrikdb/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A vector store treats your agent's memory as an append-only pile. Store
"the rate limit is 100/min" today and "the rate limit is 500/min" next
month, and both sit there forever, equally weighted — retrieval returns
whichever embeds closer to the query, and nothing ever notices they
disagree.

This package plugs [YantrikDB](https://github.com/yantrikos/yantrikdb) — a
cognitive memory engine — into LangChain's standard interfaces. Same
`VectorStore` API your chains already use, but stored records behave like
memories:

- **Temporal decay** — each record has a half-life; ranking blends
  similarity with decay, recency, and importance, so stale facts lose to
  fresh ones at equal similarity.
- **Conflict candidates** — `store.think()` flags pairs of very similar
  records that may contradict each other and hands them back with rids and
  a suggested action for you to review. This is a similarity flag, not
  natural-language inference: see the caveat below.
- **Consolidation** — near-duplicates get merged instead of accumulating.
- **Explainable retrieval** — every hit can tell you *why* it surfaced
  (`"semantically similar (0.72)"`, `"recent"`, `"important (decay=0.50)"`).

No external services and no model download: the engine is an embedded
Rust core (SQLite-backed, single file) with a bundled embedder. Bring
your own LangChain `Embeddings` if you want a larger model.

## 60 seconds

```bash
pip install langchain-yantrikdb
```

```python
from langchain_yantrikdb import YantrikDBVectorStore

store = YantrikDBVectorStore(db_path="./memory.db")

store.add_texts([
    "The deploy target is eu-west-1",
    "The database is PostgreSQL 16",
])

docs = store.similarity_search("where do we deploy?", k=1)
print(docs[0].page_content)   # The deploy target is eu-west-1

retriever = store.as_retriever()  # drop into any chain
```

## The part a plain vector store can't do

Store two facts that disagree, then ask the engine to think:

```python
store.add_texts([
    "The API rate limit is 100 requests per minute",
    "The API rate limit is 500 requests per minute",
])

report = store.think()
for trigger in report["triggers"]:
    print(trigger["reason"], "->", trigger["suggested_action"])
# Two memories about 'API' are 97% similar but may contradict each other
# (rid_a=..., rid_b=...) -> review_conflict
```

What that is, and what it isn't: `think()` flags pairs of records whose
embeddings are very similar (97% here) and leaves the decision to you. The label depends on the engine release: 0.18.x reports this pair as
`potential_conflict` / `review_conflict`, while 0.13.x (what PyPI's 0.1.0
installs) reports it as `redundancy` / `consolidate_or_forget`.

`store.conflicts()` is the separate, stricter list. YantrikDB detects
contradictions between structured or recognized single-valued claims,
including polarity and temporal conflicts, and records them for review;
arbitrary sentence pairs and multi-valued relations are intentionally not
flagged. For plain sentences like the ones above it stays empty.

And ask retrieval to explain itself:

```python
for doc, why in store.explain_search("what is the rate limit?", k=2):
    print(doc.page_content, why["why_retrieved"])
# ... ['semantically similar (0.72)', 'recent', 'important (decay=0.50)', 'keyword_match']
```

Scores returned by `similarity_search_with_score` are the same blended
score the engine ranks by (similarity x decay x recency x importance,
in [0, 1]) — documented, not raw cosine in disguise.

## Chat history

`YantrikDBChatMessageHistory` persists sessions in the same database
file, one namespace per session. Tool calls and `additional_kwargs`
survive the round trip.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_yantrikdb import YantrikDBChatMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: YantrikDBChatMessageHistory(
        session_id, db_path="./memory.db"
    ),
    input_messages_key="input",
    history_messages_key="history",
)
```

The buffer keeps the most recent 1,000 messages per session (configurable
via `max_turns`). For the long-term layer — the one that decays,
consolidates, and gets contradiction-checked — distill what matters into
a `YantrikDBVectorStore` on the same file.

## Your own embeddings

```python
from langchain_openai import OpenAIEmbeddings

store = YantrikDBVectorStore(
    db_path="./memory.db",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    namespace="docs",
)
```

The embedding dimension is probed at construction and must stay
consistent for the lifetime of the database file. With `embedding=None`
the bundled embedder is used — adequate for agent-memory recall, smaller
than sentence-transformer models. Its dimension is fixed when the file is
created (64 with engine 0.13.x, 256 with 0.18.x), and an existing file
keeps its original dimension when you upgrade the engine.

## When NOT to use this

- **Static document RAG at scale.** If the corpus doesn't change and you
  just need nearest-neighbour over a million chunks, a dedicated vector
  database is the better tool. YantrikDB's decay and consolidation add
  nothing to documents that never go stale.
- **You need caller-supplied ids from PyPI 0.1.0.** That release raises
  `NotImplementedError` on `add_texts(ids=...)`. `main` accepts them as
  upsert keys (re-adding an id replaces the document); until the next
  release, install from git:
  `pip install git+https://github.com/yantrikos/langchain-yantrikdb`.
- **MMR retrieval.** `max_marginal_relevance_search` is not implemented.
- **Exact score reproducibility.** Blended scores move as records age —
  that is the point, but it breaks tests that pin exact score values.

## Interface coverage

| LangChain surface | Status |
| --- | --- |
| `add_texts` / `add_documents` | supported (engine-assigned ids; caller-supplied ids as upsert keys on `main` only) |
| `similarity_search` / `_with_score` / `_by_vector` | supported |
| `similarity_search_with_relevance_scores` | supported (scores already in [0, 1]) |
| `delete(ids)` / `delete()` (namespace-wide) | supported (tombstone) |
| `get_by_ids` | supported |
| `from_texts` | supported |
| `as_retriever` | supported |
| async variants | inherited executor-backed defaults |
| `max_marginal_relevance_search` | not implemented |
| `BaseChatMessageHistory` | supported, per-session namespaces |

Extras beyond the standard interface: `explain_search()`, `think()`,
`conflicts()`, and `store.db` for the full engine API (record links,
knowledge graph, memory packs).

Tested against langchain-core 0.3.x and 1.x on Python 3.10-3.14.

## Related projects

- [yantrikdb](https://github.com/yantrikos/yantrikdb) — the engine
  itself: Rust core, Python bindings, CLI, REST server.
- [yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp) — the same
  memory as an MCP server for Claude Code, Cursor, and other MCP hosts.
- [yantrikdb-hermes-plugin](https://github.com/yantrikos/yantrikdb-hermes-plugin)
  — memory provider for hermes-agent.

## License

MIT (this integration). The YantrikDB engine is Apache-2.0.

---

Pranab Sarkar, Independent Researcher
