"""LangChain's official standard test suite, run against a real engine.

``langchain_tests.integration_tests.VectorStoreIntegrationTests`` is the
conformance suite every listed LangChain vector store is measured against. It
is run here exactly as published — no overridden test bodies, no mocks, no
skips beyond the suite's own capability switches. Each test gets its own
temp-directory database, so "the store starts empty" is a real property of a
real engine rather than something arranged by a fake.

There is no equivalent published standard suite for
``BaseChatMessageHistory``; ``YantrikDBChatMessageHistory`` is covered by the
hand-written tests in ``tests/test_chat_message_history.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_yantrikdb import YantrikDBVectorStore


class TestYantrikDBVectorStore(VectorStoreIntegrationTests):
    """YantrikDBVectorStore against the full published conformance suite."""

    @pytest.fixture()
    def vectorstore(self, tmp_path: Path) -> Generator[VectorStore, None, None]:
        """An empty store on its own database file.

        ``tmp_path`` is function-scoped, so every test opens a database that
        has never held a record — the suite's empty-store assertions are
        checked against a genuinely fresh engine, not a cleared one. The
        ``delete()`` in the teardown is therefore belt-and-braces; it also
        exercises the namespace-wide delete path on every test.
        """
        store = YantrikDBVectorStore(
            db_path=str(tmp_path / "standard.db"),
            embedding=self.get_embeddings(),
        )
        try:
            yield store
        finally:
            store.delete()
            store.close()
