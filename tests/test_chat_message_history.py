"""Real-engine tests for YantrikDBChatMessageHistory."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_yantrikdb import YantrikDBChatMessageHistory


@pytest.fixture()
def history(tmp_path):
    h = YantrikDBChatMessageHistory("session-1", db_path=str(tmp_path / "chat.db"))
    yield h
    h.close()


def test_requires_database():
    with pytest.raises(ValueError):
        YantrikDBChatMessageHistory("nowhere")


def test_round_trip_order_and_types(history):
    history.add_messages(
        [
            SystemMessage(content="You are terse."),
            HumanMessage(content="What region do we deploy to?"),
            AIMessage(content="eu-west-1"),
        ]
    )
    msgs = history.messages
    assert [m.type for m in msgs] == ["system", "human", "ai"]
    assert msgs[1].content == "What region do we deploy to?"
    assert msgs[2].content == "eu-west-1"


def test_tool_calls_and_kwargs_survive(history):
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"q": "region"}, "id": "call-1"}],
        additional_kwargs={"custom": "value"},
    )
    history.add_messages([ai])
    (restored,) = history.messages
    assert isinstance(restored, AIMessage)
    assert restored.tool_calls == ai.tool_calls
    assert restored.additional_kwargs.get("custom") == "value"


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "persist.db")
    h1 = YantrikDBChatMessageHistory("s", db_path=path)
    h1.add_user_message("remember me")
    h1.close()

    h2 = YantrikDBChatMessageHistory("s", db_path=path)
    try:
        msgs = h2.messages
        assert len(msgs) == 1
        assert msgs[0].content == "remember me"
    finally:
        h2.close()


def test_sessions_isolated(tmp_path):
    path = str(tmp_path / "multi.db")
    a = YantrikDBChatMessageHistory("alice", db_path=path)
    try:
        b = YantrikDBChatMessageHistory("bob", db=a.db)
        a.add_user_message("alice says hi")
        b.add_user_message("bob says hello")
        assert [m.content for m in a.messages] == ["alice says hi"]
        assert [m.content for m in b.messages] == ["bob says hello"]
        b.clear()
        assert b.messages == []
        assert len(a.messages) == 1
    finally:
        a.close()


def test_clear_only_this_session(history):
    history.add_user_message("hello")
    history.clear()
    assert history.messages == []


def test_max_turns_ring_buffer(tmp_path):
    h = YantrikDBChatMessageHistory(
        "ring", db_path=str(tmp_path / "ring.db"), max_turns=3
    )
    try:
        for i in range(5):
            h.add_user_message(f"msg {i}")
        msgs = h.messages
        assert [m.content for m in msgs] == ["msg 2", "msg 3", "msg 4"]
    finally:
        h.close()


def test_plain_turns_from_other_clients_readable(history):
    # Another YantrikDB client may write raw role/content turns into the
    # same namespace; the history must degrade gracefully, not crash.
    history.db.record_turn(history.namespace, "ai", "plain text turn", max_turns=10)
    (msg,) = history.messages
    assert isinstance(msg, AIMessage)
    assert msg.content == "plain text turn"
