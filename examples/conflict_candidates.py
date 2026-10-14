"""Conflict candidates through the LangChain VectorStore interface.

think() flags pairs of records whose embeddings are very similar and may
be worth a human look — it is a similarity flag, not natural-language
contradiction detection. See the README's "What that is, and what it
isn't" caveat, and store.conflicts() for the stricter, structured check.

Run: python examples/conflict_candidates.py
(no API keys, no model downloads — the bundled embedder does the work)
"""

from langchain_yantrikdb import YantrikDBVectorStore

store = YantrikDBVectorStore(db_path=":memory:")

# Month 1: someone stores a fact.
store.add_texts(
    ["The API rate limit is 100 requests per minute"], importance=0.8
)

# Month 2: someone stores an updated — and possibly contradictory — fact.
store.add_texts(
    ["The API rate limit is 500 requests per minute"], importance=0.8
)

# A plain vector store now holds both forever and serves whichever
# embeds closer to the query. think() flags the pair as a candidate:
report = store.think()
for trigger in report.get("triggers", []):
    print("FLAGGED:", trigger["reason"])
    print("  suggested action:", trigger["suggested_action"])
    print("  urgency:", round(trigger["urgency"], 2))

# Retrieval explains itself:
print("\nWhy did each hit surface?")
for doc, why in store.explain_search("what is the rate limit?", k=2):
    print(f"  {doc.page_content!r}")
    print(f"    score={why['score']:.3f} because {why['why_retrieved']}")

store.close()
