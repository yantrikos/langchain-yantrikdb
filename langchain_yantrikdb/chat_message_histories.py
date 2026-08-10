"""Chat message history persisted in YantrikDB's conversation buffer."""

from __future__ import annotations

import json
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
    messages_from_dict,
)

from yantrikdb import YantrikDB

DEFAULT_MAX_TURNS = 1000


class YantrikDBChatMessageHistory(BaseChatMessageHistory):
    """Session-scoped chat history stored in a YantrikDB database.

    Each session maps to its own YantrikDB namespace
    (``"{prefix}:{session_id}"``), so any number of sessions share one
    database file with full isolation. Messages go into the engine's
    conversation buffer — stored verbatim, not embedded — and are
    serialized with LangChain's message format, so tool calls and
    ``additional_kwargs`` survive the round trip.

    The buffer is a ring: only the most recent ``max_turns`` messages are
    kept per session (default 1000). For a durable long-term record,
    distill what matters into a :class:`~langchain_yantrikdb.YantrikDBVectorStore`
    on the same database file — that is the layer that consolidates,
    decays, and gets contradiction-checked.

    Example:
        .. code-block:: python

            from langchain_yantrikdb import YantrikDBChatMessageHistory

            history = YantrikDBChatMessageHistory(
                session_id="user-42", db_path="./memory.db"
            )
            history.add_user_message("My deploy target is eu-west-1")
            history.messages  # -> [HumanMessage(...)]

    Args:
        session_id: Identifier for this conversation session.
        db_path: Path to the database file. Ignored when ``db`` is given.
        db: An already-open ``yantrikdb.YantrikDB`` to reuse (e.g. the
            engine behind a ``YantrikDBVectorStore``). The caller keeps
            ownership; ``close()`` becomes a no-op.
        namespace_prefix: Prefix for the per-session namespace.
        max_turns: Ring-buffer size — messages kept per session.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str | None = None,
        *,
        db: YantrikDB | None = None,
        namespace_prefix: str = "chat",
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        if db is None and db_path is None:
            raise ValueError("Provide db_path (or an open db=) — chat history needs a database.")
        self.session_id = session_id
        self._namespace = f"{namespace_prefix}:{session_id}"
        self._max_turns = max_turns
        if db is not None:
            self._db = db
            self._owns_db = False
        else:
            self._db = YantrikDB.with_default(db_path)
            self._owns_db = True

    @property
    def db(self) -> YantrikDB:
        """The underlying ``yantrikdb.YantrikDB`` engine."""
        return self._db

    @property
    def namespace(self) -> str:
        """The per-session YantrikDB namespace."""
        return self._namespace

    def close(self) -> None:
        """Close the underlying database (no-op when constructed with
        ``db=`` — the caller owns that handle)."""
        if self._owns_db:
            self._db.close()

    # -- BaseChatMessageHistory -------------------------------------------

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """All stored messages for this session, oldest first."""
        turns = self._db.recent_turns(self._namespace, limit=self._max_turns)
        out: list[BaseMessage] = []
        for turn in turns:
            out.extend(_turn_to_messages(turn))
        return out

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append messages to the session buffer."""
        for message in messages:
            self._db.record_turn(
                self._namespace,
                message.type,
                json.dumps(message_to_dict(message)),
                max_turns=self._max_turns,
            )

    def clear(self) -> None:
        """Delete every message in this session (other sessions untouched)."""
        self._db.clear_turns(self._namespace)


def _turn_to_messages(turn: dict) -> list[BaseMessage]:
    """Decode one stored turn back into LangChain messages.

    Turns written by this class carry a full ``message_to_dict`` payload.
    Turns written by other YantrikDB clients (plain role/content strings)
    fall back to a message type inferred from the role.
    """
    content = turn.get("content", "")
    try:
        payload = json.loads(content)
        if isinstance(payload, dict) and "type" in payload and "data" in payload:
            return messages_from_dict([payload])
    except (json.JSONDecodeError, TypeError):
        pass
    role = (turn.get("role") or "").lower()
    if role in ("ai", "assistant"):
        return [AIMessage(content=content)]
    if role == "system":
        return [SystemMessage(content=content)]
    return [HumanMessage(content=content)]
