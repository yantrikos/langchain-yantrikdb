"""LangChain integration for YantrikDB, a cognitive memory engine.

Provides:

* :class:`YantrikDBVectorStore` — the engine as a LangChain ``VectorStore``.
  Stored texts become memories with importance and temporal decay; the
  engine consolidates near-duplicates and flags contradictions, and every
  retrieval can explain why each hit surfaced.
* :class:`YantrikDBChatMessageHistory` — session-scoped chat persistence
  via YantrikDB namespaces, for ``RunnableWithMessageHistory``.
"""

from langchain_yantrikdb.chat_message_histories import YantrikDBChatMessageHistory
from langchain_yantrikdb.vectorstores import YantrikDBVectorStore

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("langchain-yantrikdb")
except Exception:  # pragma: no cover - source tree without installed dist
    __version__ = "0.0.0+unknown"

__all__ = [
    "YantrikDBVectorStore",
    "YantrikDBChatMessageHistory",
]
