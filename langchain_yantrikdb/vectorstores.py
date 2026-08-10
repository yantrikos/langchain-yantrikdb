"""YantrikDB as a LangChain VectorStore.

YantrikDB is a cognitive memory engine, not a plain vector index. Documents
stored through this class become *memories*: each carries an importance
weight and a temporal half-life, retrieval blends semantic similarity with
decay, recency, and importance, and the engine can consolidate near-duplicate
records and flag contradictions between them (``think()``). Every hit can
explain why it was retrieved (``explain_search()``).

If all you need is nearest-neighbour lookup over static documents, a plain
vector store is simpler. Reach for this class when the corpus is an agent's
accumulating memory — facts that change, repeat, and contradict over time.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from yantrikdb import YantrikDB

DEFAULT_NAMESPACE = "langchain"


class _EmbeddingsShim:
    """Adapt a LangChain ``Embeddings`` object to the ``encode(text)``
    interface YantrikDB expects from a Python-side embedder."""

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings

    def encode(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)


class YantrikDBVectorStore(VectorStore):
    """LangChain ``VectorStore`` backed by the YantrikDB memory engine.

    Two embedding modes:

    * **Bundled (default).** With ``embedding=None`` the store opens the
      database via ``YantrikDB.with_default``, which attaches the engine's
      bundled 64-dimension embedder. No model download, no extra
      dependencies — ``pip install langchain-yantrikdb`` is the whole setup.
    * **External.** Pass any LangChain ``Embeddings`` implementation and the
      store attaches it to the engine (``embed_query`` per text). The
      database's embedding dimension is probed from the embedder at
      construction and must stay consistent for the lifetime of the
      database file.

    What differs from a plain vector store:

    * Records decay: each memory has a half-life, and retrieval scores blend
      similarity with decay, recency, and importance. A stale record loses
      to a fresh one at equal similarity.
    * ``think()`` runs the engine's cognition pass — consolidation of
      near-duplicates, contradiction scanning, pattern mining — over what
      you stored through this same interface.
    * ``explain_search()`` returns, per hit, the engine's ``why_retrieved``
      reasons and the score breakdown.

    Example:
        .. code-block:: python

            from langchain_yantrikdb import YantrikDBVectorStore

            store = YantrikDBVectorStore(db_path="./memory.db")
            store.add_texts(["The deploy target is eu-west-1"])
            docs = store.similarity_search("where do we deploy?", k=1)

    Args:
        db_path: Path to the database file. ``":memory:"`` for an ephemeral
            in-memory database. Ignored when ``db`` is given.
        embedding: Optional LangChain ``Embeddings``. ``None`` uses the
            engine's bundled embedder.
        namespace: YantrikDB namespace this store reads and writes. Two
            stores on the same file with different namespaces are fully
            isolated.
        db: An already-open ``yantrikdb.YantrikDB`` instance to reuse
            (for sharing one engine between a store and a chat history).
            The caller keeps ownership; ``close()`` becomes a no-op.
        importance: Default importance (0.0-1.0) stamped on added texts.
            Higher-importance memories decay slower and rank higher.
        memory_type: Default memory type for added texts. ``"semantic"``
            (facts) is the right default for documents; ``"episodic"``
            (events) decays faster.
        domain: Default domain tag for added texts.
        source: Default source tag for added texts (``"document"``).
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        embedding: Embeddings | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        db: YantrikDB | None = None,
        importance: float = 0.5,
        memory_type: str = "semantic",
        domain: str = "general",
        source: str = "document",
    ) -> None:
        self._embedding = embedding
        self._namespace = namespace
        self._importance = importance
        self._memory_type = memory_type
        self._domain = domain
        self._source = source
        if db is not None:
            self._db = db
            self._owns_db = False
        elif embedding is None:
            self._db = YantrikDB.with_default(db_path)
            self._owns_db = True
        else:
            probe = embedding.embed_query("dimension probe")
            self._db = YantrikDB(
                db_path=db_path,
                embedding_dim=len(probe),
                embedder=_EmbeddingsShim(embedding),
            )
            self._owns_db = True

    # -- lifecycle ---------------------------------------------------------

    @property
    def db(self) -> YantrikDB:
        """The underlying ``yantrikdb.YantrikDB`` engine, for direct use of
        engine APIs this class does not wrap (links, packs, triggers, ...)."""
        return self._db

    @property
    def namespace(self) -> str:
        """The YantrikDB namespace this store operates in."""
        return self._namespace

    def close(self) -> None:
        """Close the underlying database (no-op when constructed with
        ``db=`` — the caller owns that handle)."""
        if self._owns_db:
            self._db.close()

    # -- VectorStore: write path ------------------------------------------

    @property
    def embeddings(self) -> Embeddings | None:
        """The external LangChain ``Embeddings`` if one was supplied,
        else ``None`` (the engine's bundled embedder is in use)."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        domain: str | None = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store texts as memories. Returns the engine-assigned record ids.

        YantrikDB assigns every record a UUIDv7 rid; caller-supplied ``ids``
        are not supported and raise ``NotImplementedError``. Use the
        returned rids for ``get_by_ids`` / ``delete``.

        Args:
            texts: Texts to store.
            metadatas: Optional per-text metadata dicts; round-trip intact
                onto retrieved ``Document.metadata``.
            ids: Not supported — YantrikDB assigns rids.
            importance: Override the store's default importance for this
                batch (0.0-1.0). Importance slows decay and raises rank.
            memory_type: Override memory type (``"semantic"``,
                ``"episodic"``, ``"procedural"``).
            domain: Override domain tag.
            source: Override source tag.
        """
        if ids is not None:
            raise NotImplementedError(
                "YantrikDB assigns UUIDv7 record ids; caller-supplied ids are "
                "not supported. Use the ids returned by add_texts()."
            )
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) does not match "
                f"texts length ({len(texts)})"
            )
        rids: list[str] = []
        for i, text in enumerate(texts):
            rids.append(
                self._db.record(
                    text,
                    memory_type=memory_type or self._memory_type,
                    importance=self._importance if importance is None else importance,
                    metadata=(metadatas[i] if metadatas else None) or {},
                    namespace=self._namespace,
                    domain=domain or self._domain,
                    source=source or self._source,
                )
            )
        return rids

    # -- VectorStore: read path -------------------------------------------

    def _recall(self, k: int, **recall_kwargs: Any) -> list[dict]:
        recall_kwargs.setdefault("namespace", self._namespace)
        # Over-fetch candidates, then truncate to k after the engine's
        # blended ordering. Two reasons: (1) the engine ranks its candidate
        # set by the blended score, so a record with slightly lower
        # similarity but much higher importance/recency can deserve a top-k
        # slot yet miss an exact-k candidate set; (2) on yantrikdb 0.13.x,
        # top_k=1 against a 2-record namespace can return the wrong record
        # (ANN candidate selection quirk — reproducible, reported upstream).
        fetch_k = max(k * 4, 16)
        return self._db.recall(top_k=fetch_k, **recall_kwargs)[:k]

    @staticmethod
    def _hit_to_document(hit: dict) -> Document:
        return Document(
            id=hit["rid"],
            page_content=hit["text"],
            metadata=hit.get("metadata") or {},
        )

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Retrieve the ``k`` best memories for ``query``.

        Ranking is the engine's blended retrieval score — semantic
        similarity weighted by temporal decay, recency, and importance —
        not raw cosine distance. A record you stored a year ago and never
        touched ranks below an equally-similar record from yesterday.

        Extra keyword arguments are passed to ``YantrikDB.recall``
        (``memory_type=``, ``domain=``, ``source=``, ``certainty_min=``,
        ``namespace=`` to override the store default, ...).
        """
        return [self._hit_to_document(h) for h in self._recall(k, query=query, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Like ``similarity_search`` but with each hit's blended retrieval
        score (0.0-1.0; similarity x decay x recency x importance).
        Higher is better."""
        return [
            (self._hit_to_document(h), h["score"])
            for h in self._recall(k, query=query, **kwargs)
        ]

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Retrieve by a precomputed query vector. The vector's dimension
        must match the database's embedding dimension."""
        return [
            self._hit_to_document(h)
            for h in self._recall(k, query_embedding=embedding, **kwargs)
        ]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # Engine scores are already normalized relevance in [0, 1].
        return lambda score: max(0.0, min(1.0, score))

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Fetch memories by rid. Missing or deleted (tombstoned) rids are
        skipped, per the LangChain contract."""
        docs: list[Document] = []
        for rid in ids:
            rec = self._db.get(rid)
            if rec is None or rec.get("consolidation_status") == "tombstoned":
                continue
            docs.append(
                Document(
                    id=rec["rid"],
                    page_content=rec["text"],
                    metadata=rec.get("metadata") or {},
                )
            )
        return docs

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Forget memories. ``ids=None`` forgets every record in this
        store's namespace. Forgetting is a tombstone (soft delete) — the
        record stops being retrievable immediately."""
        if ids is not None:
            for rid in ids:
                self._db.forget(rid)
            return True
        cursor: str | None = None
        while True:
            page = self._db.list_records(
                namespace=self._namespace, since_rid=cursor, limit=200
            )
            for rec in page["records"]:
                if rec.get("consolidation_status") != "tombstoned":
                    self._db.forget(rec["rid"])
            cursor = page["next_cursor"]
            if cursor is None:
                return True

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        db_path: str = ":memory:",
        namespace: str = DEFAULT_NAMESPACE,
        **kwargs: Any,
    ) -> "YantrikDBVectorStore":
        """Build a store and add ``texts`` in one call.

        Defaults to an in-memory database; pass ``db_path`` for a
        persistent one — persistence is the point of a memory engine.
        """
        store = cls(db_path=db_path, embedding=embedding, namespace=namespace, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    # -- YantrikDB differentiators ----------------------------------------

    def explain_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, dict]]:
        """``similarity_search`` plus the engine's explanation per hit.

        Each hit's dict has:

        * ``"score"`` — the blended retrieval score.
        * ``"scores"`` — the breakdown (similarity, decay, recency,
          importance, graph proximity, and each one's contribution).
        * ``"why_retrieved"`` — human-readable reasons, e.g.
          ``["semantically similar (0.90)", "recent", "important (decay=0.80)"]``.

        Useful for debugging retrieval and for showing an agent *why* a
        memory surfaced.
        """
        return [
            (
                self._hit_to_document(h),
                {
                    "score": h.get("score"),
                    "scores": h.get("scores"),
                    "why_retrieved": h.get("why_retrieved"),
                },
            )
            for h in self._recall(k, query=query, **kwargs)
        ]

    def think(
        self,
        *,
        run_consolidation: bool = True,
        run_conflict_scan: bool = True,
        run_pattern_mining: bool = True,
    ) -> dict:
        """Run the engine's cognition pass over stored memories.

        Consolidates near-duplicates, scans for contradictions between
        records, and mines patterns. Returns the engine's report — the
        ``"triggers"`` list contains findings such as two memories being
        97% similar with conflicting content, each with source rids and a
        suggested action.

        Note: ``think()`` operates engine-wide, not per-namespace. On a
        database file shared across namespaces it sees all of them.
        """
        return self._db.think(
            config={
                "run_consolidation": run_consolidation,
                "run_conflict_scan": run_conflict_scan,
                "run_pattern_mining": run_pattern_mining,
            }
        )

    def conflicts(self, *, status: str | None = None, limit: int = 50) -> list[dict]:
        """Contradictions the engine has detected between stored memories
        in this store's namespace (populated by ``think()``)."""
        return self._db.get_conflicts(
            status=status, namespace=self._namespace, limit=limit
        )
