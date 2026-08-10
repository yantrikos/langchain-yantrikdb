"""Real-engine tests for YantrikDBVectorStore — no mocks on the core path."""

import asyncio
import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_yantrikdb import YantrikDBVectorStore


class HashEmbeddings(Embeddings):
    """Deterministic toy embedder: identical text -> identical vector.

    Self-contained (no numpy) so it behaves the same on every
    langchain-core line the suite runs against.
    """

    def __init__(self, size: int = 32) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * (self.size * 4 // len(digest) + 1))[: self.size]
        return [b / 255.0 - 0.5 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

FACTS = [
    ("The deploy target is eu-west-1", {"topic": "infra"}),
    ("The database is PostgreSQL 16", {"topic": "infra"}),
    ("Standup is at 09:30 on weekdays", {"topic": "process"}),
]


@pytest.fixture()
def store(tmp_path):
    s = YantrikDBVectorStore(db_path=str(tmp_path / "test.db"))
    yield s
    s.close()


def _seed(store):
    texts = [t for t, _ in FACTS]
    metadatas = [m for _, m in FACTS]
    return store.add_texts(texts, metadatas=metadatas)


def test_add_and_similarity_search(store):
    rids = _seed(store)
    assert len(rids) == 3
    assert all(isinstance(r, str) and r for r in rids)

    docs = store.similarity_search("what region do we deploy to?", k=1)
    assert len(docs) == 1
    assert docs[0].page_content == "The deploy target is eu-west-1"
    assert docs[0].metadata == {"topic": "infra"}
    assert docs[0].id == rids[0]


def test_similarity_search_with_score(store):
    _seed(store)
    results = store.similarity_search_with_score("which database do we use?", k=3)
    assert len(results) == 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert results[0][0].page_content == "The database is PostgreSQL 16"


def test_similarity_search_with_relevance_scores(store):
    _seed(store)
    results = store.similarity_search_with_relevance_scores("deploy region", k=2)
    assert all(0.0 <= s <= 1.0 for _, s in results)


def test_add_documents_and_retriever(store):
    store.add_documents(
        [Document(page_content=t, metadata=m) for t, m in FACTS]
    )
    retriever = store.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke("when is standup?")
    assert docs[0].page_content == "Standup is at 09:30 on weekdays"


def test_caller_ids_rejected(store):
    with pytest.raises(NotImplementedError):
        store.add_texts(["x"], ids=["my-id"])


def test_get_by_ids(store):
    rids = _seed(store)
    docs = store.get_by_ids([rids[1], "0" * 32])  # one real, one missing
    assert len(docs) == 1
    assert docs[0].id == rids[1]
    assert docs[0].page_content == "The database is PostgreSQL 16"


def test_delete_by_id(store):
    rids = _seed(store)
    assert store.delete([rids[0]]) is True
    assert store.get_by_ids([rids[0]]) == []
    docs = store.similarity_search("what region do we deploy to?", k=3)
    assert all(d.page_content != "The deploy target is eu-west-1" for d in docs)


def test_delete_all_in_namespace(store):
    rids = _seed(store)
    assert store.delete() is True
    assert store.get_by_ids(rids) == []
    assert store.similarity_search("anything at all", k=5) == []


def test_namespace_isolation(tmp_path):
    db_path = str(tmp_path / "shared.db")
    a = YantrikDBVectorStore(db_path=db_path, namespace="tenant_a")
    try:
        a.add_texts(["tenant A secret plan"])
        b = YantrikDBVectorStore(db=a.db, namespace="tenant_b")
        b.add_texts(["tenant B grocery list"])
        hits_b = b.similarity_search("secret plan", k=5)
        assert all("tenant A" not in d.page_content for d in hits_b)
        assert len(a.similarity_search("secret plan", k=5)) == 1
        b.close()  # no-op: does not own the handle
        assert len(a.similarity_search("secret plan", k=5)) == 1
    finally:
        a.close()


def test_from_texts_in_memory():
    store = YantrikDBVectorStore.from_texts(
        [t for t, _ in FACTS], metadatas=[m for _, m in FACTS]
    )
    try:
        docs = store.similarity_search("deployment region", k=1)
        assert docs[0].page_content == "The deploy target is eu-west-1"
    finally:
        store.close()


def test_external_langchain_embeddings(tmp_path):
    emb = HashEmbeddings(size=32)
    store = YantrikDBVectorStore(
        db_path=str(tmp_path / "ext.db"), embedding=emb, namespace="ext"
    )
    try:
        assert store.embeddings is emb
        store.add_texts(["alpha document", "beta document"])
        # Deterministic embeddings: an identical query vector must retrieve
        # the identical text as the top hit.
        docs = store.similarity_search("alpha document", k=1)
        assert docs[0].page_content == "alpha document"
        # By-vector path with a caller-computed vector.
        docs2 = store.similarity_search_by_vector(emb.embed_query("beta document"), k=1)
        assert docs2[0].page_content == "beta document"
    finally:
        store.close()


def test_explain_search_surfaces_why_retrieved(store):
    _seed(store)
    results = store.explain_search("deploy region", k=2)
    assert len(results) == 2
    doc, why = results[0]
    assert isinstance(doc, Document)
    assert isinstance(why["why_retrieved"], list) and why["why_retrieved"]
    assert all(isinstance(reason, str) for reason in why["why_retrieved"])
    assert 0.0 <= why["score"] <= 1.0
    assert "similarity" in why["scores"]


def test_think_flags_contradictory_near_duplicates(store):
    rids = store.add_texts(
        [
            "The API rate limit is 100 requests per minute",
            "The API rate limit is 500 requests per minute",
        ],
        importance=0.8,
    )
    report = store.think()
    assert isinstance(report, dict)
    triggers = report.get("triggers", [])
    flagged = [
        t
        for t in triggers
        if set(rids) <= set(t.get("source_rids", []))
    ]
    assert flagged, f"engine did not flag the contradictory pair; triggers={triggers}"
    assert flagged[0].get("suggested_action")


def test_conflicts_listing_shape(store):
    _seed(store)
    store.think()
    assert isinstance(store.conflicts(), list)


def test_async_defaults_work(store):
    _seed(store)
    docs = asyncio.run(store.asimilarity_search("deploy region", k=1))
    assert docs[0].page_content == "The deploy target is eu-west-1"


def test_metadata_length_mismatch(store):
    with pytest.raises(ValueError):
        store.add_texts(["a", "b"], metadatas=[{}])
