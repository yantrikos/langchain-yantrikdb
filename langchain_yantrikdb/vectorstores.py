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

# Metadata key under which a caller-supplied document id is stored. YantrikDB
# mints its own UUIDv7 rid for every record and that rid is not caller-
# choosable, so an externally-chosen id has to live alongside the record
# instead. It is stripped again on the way out, so it never appears in the
# metadata a caller gets back.
_LC_ID_KEY = "__lc_id__"


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
        ids: list[str | None] | None = None,
        importance: float | None = None,
        memory_type: str | None = None,
        domain: str | None = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store texts as memories. Returns one id per text, in order.

        Ids are **upsert keys**. Adding the same id twice replaces the earlier
        memory instead of duplicating it, so a repeated ``add_texts`` is
        idempotent, and re-adding an id with new text mutates that document
        in place.

        Where an entry of ``ids`` is ``None`` — or ``ids`` is omitted entirely
        — the engine's own UUIDv7 rid becomes that document's id. Both kinds
        of id are accepted everywhere this class takes one (``get_by_ids``,
        ``delete``), and whichever applies is what comes back as
        ``Document.id``.

        Args:
            texts: Texts to store.
            metadatas: Optional per-text metadata dicts; round-trip intact
                onto retrieved ``Document.metadata``. Never mutated.
            ids: Optional per-text ids, used as upsert keys. May contain
                ``None`` for texts that should take an engine-assigned rid.
            importance: Override the store's default importance for this
                batch (0.0-1.0). Importance slows decay and raises rank.
            memory_type: Override memory type (``"semantic"``,
                ``"episodic"``, ``"procedural"``).
            domain: Override domain tag.
            source: Override source tag.
        """
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) does not match "
                f"texts length ({len(texts)})"
            )
        if ids is not None and len(ids) != len(texts):
            raise ValueError(
                f"ids length ({len(ids)}) does not match "
                f"texts length ({len(texts)})"
            )

        supplied: list[str | None] = [None] * len(texts) if ids is None else list(ids)

        # Upsert: tombstone whatever those ids already name before the
        # replacements land. One paged scan resolves the whole batch, so a
        # bulk re-add costs a single pass rather than one pass per id.
        reused = {i for i in supplied if i is not None}
        if reused:
            for rid in self._resolve_external_ids(reused).values():
                self._db.forget(rid)

        assigned: list[str] = []
        for i, text in enumerate(texts):
            # Copied, never aliased: add_documents hands us Document.metadata
            # directly, and the caller's object must not grow a private key.
            metadata = dict(metadatas[i]) if metadatas else {}
            external = supplied[i]
            if external is not None:
                metadata[_LC_ID_KEY] = external
            rid = self._db.record(
                text,
                memory_type=memory_type or self._memory_type,
                importance=self._importance if importance is None else importance,
                metadata=metadata,
                namespace=self._namespace,
                domain=domain or self._domain,
                source=source or self._source,
            )
            assigned.append(rid if external is None else external)
        return assigned

    # -- id resolution -----------------------------------------------------

    def _resolve_external_ids(self, wanted: set[str]) -> dict[str, str]:
        """Map caller-supplied ids to engine rids with one paged scan of this
        namespace, stopping as soon as every id is accounted for.

        The engine indexes records by rid and by vector, not by an arbitrary
        metadata key, so resolving an external id is a scan by construction.
        Ids this namespace does not carry are simply absent from the result.
        """
        found: dict[str, str] = {}
        cursor: str | None = None
        while wanted - found.keys():
            page = self._db.list_records(
                namespace=self._namespace, since_rid=cursor, limit=200
            )
            for rec in page["records"]:
                if rec.get("consolidation_status") == "tombstoned":
                    continue
                external = (rec.get("metadata") or {}).get(_LC_ID_KEY)
                if external in wanted:
                    found.setdefault(external, rec["rid"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        return found

    def _active(self, rid: str) -> dict | None:
        """The live record for an engine rid, or ``None`` when it names
        nothing or has been forgotten. An unknown id reads as ``None`` rather
        than raising, which is what the LangChain id contract wants."""
        rec = self._db.get(rid)
        if rec is None or rec.get("consolidation_status") == "tombstoned":
            return None
        return rec

    def _records_for(self, ids: Sequence[str]) -> dict[str, dict]:
        """Resolve a mix of engine rids and caller-supplied ids to live
        records, keyed by the id as it was asked for. Ids naming nothing are
        omitted."""
        found: dict[str, dict] = {}
        unresolved: set[str] = set()
        for identifier in ids:
            rec = self._active(identifier)
            if rec is not None:
                found[identifier] = rec
            else:
                unresolved.add(identifier)
        if unresolved:
            for external, rid in self._resolve_external_ids(unresolved).items():
                rec = self._active(rid)
                if rec is not None:
                    found[external] = rec
        return found

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
    def _to_document(record: dict) -> Document:
        """Engine record — or recall hit, same shape — to ``Document``, with
        any caller-supplied id lifted back out of metadata."""
        metadata = dict(record.get("metadata") or {})
        external = metadata.pop(_LC_ID_KEY, None)
        return Document(
            id=record["rid"] if external is None else external,
            page_content=record["text"],
            metadata=metadata,
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
        return [self._to_document(h) for h in self._recall(k, query=query, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Like ``similarity_search`` but with each hit's blended retrieval
        score (0.0-1.0; similarity x decay x recency x importance).
        Higher is better."""
        return [
            (self._to_document(h), h["score"])
            for h in self._recall(k, query=query, **kwargs)
        ]

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Retrieve by a precomputed query vector. The vector's dimension
        must match the database's embedding dimension."""
        return [
            self._to_document(h)
            for h in self._recall(k, query_embedding=embedding, **kwargs)
        ]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        # Engine scores are already normalized relevance in [0, 1].
        return lambda score: max(0.0, min(1.0, score))

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Fetch memories by id, in the order asked for.

        Accepts caller-supplied ids and engine rids interchangeably. Ids that
        name nothing — never stored, or already forgotten — are skipped
        rather than raising, per the LangChain contract.
        """
        found = self._records_for(ids)
        return [self._to_document(found[i]) for i in ids if i in found]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Forget memories. ``ids=None`` forgets every record in this
        store's namespace. Forgetting is a tombstone (soft delete) — the
        record stops being retrievable immediately.

        Accepts caller-supplied ids and engine rids interchangeably. Ids that
        name nothing are ignored rather than raising.
        """
        if ids is not None:
            for rec in self._records_for(ids).values():
                self._db.forget(rec["rid"])
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
        ids: list[str | None] | None = None,
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
                self._to_document(h),
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
