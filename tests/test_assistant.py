"""The assistant's pipeline pieces that need no model and no database."""
import os

os.environ.setdefault("PIA_API_TOKEN", "test-token")
os.environ.setdefault("DB_HOST", "127.0.0.1")

import assistant as A
from auth import User, Visibility


def test_plan_from_model_json_and_defaults():
    p = A.plan_from({"entities": ["Iran", "United States"], "days": "30", "kind": "why_connected"}, "x")
    assert p.names == ["Iran", "United States"] and p.days == 30 and p.kind == "why_connected"
    p = A.plan_from({}, "anything new today?")
    assert p.names == [] and p.days == 7 and p.kind == "what_is_new"        # no names + "new" → what is new
    p = A.plan_from({"entities": ["a", "b", "c", "d", "e", "f"], "days": 99999, "kind": "nonsense"}, "x")
    assert len(p.names) == A.MAX_ENTITIES and p.days == 3650 and p.kind == "other"


def test_json_object_tolerates_fences_and_prose():
    assert A._json_object('```json\n{"entities": ["X"], "days": 7, "kind": "who_is"}\n```')["entities"] == ["X"]
    assert A._json_object('Sure! {"entities": []} hope that helps')["entities"] == []
    assert A._json_object("not json") == {}


def test_context_numbers_items_and_keeps_the_budget(monkeypatch):
    monkeypatch.setattr(A, "CONTEXT_CHARS", 60)
    c = A.Context()
    assert c.add("event", "e1", "first", "x" * 30, entity_id="a", other_id="b") == 1
    assert c.add("event", "e2", "second", "y" * 25) == 2
    assert c.add("event", "e3", "third", "z" * 20) is None          # over budget: dropped, numbering stays dense
    assert c.render().startswith("[1] xxx") and "[2] yyy" in c.render() and "[3]" not in c.render()


def test_citations_and_sources():
    c = A.Context()
    for i in range(4):
        c.add("event", f"e{i}", f"l{i}", f"text {i}", entity_id="a", other_id="b", source_id="bbc.co.uk")
    assert A.cited("Iran warned the US [2][4]. Nothing else [9].", 4) == [2, 4]     # [9] does not exist
    srcs = A.sources_for(c, [2, 4])
    assert [s["n"] for s in srcs] == [2, 4] and srcs[0]["kind"] == "event" and srcs[0]["other_id"] == "b"
    assert len(A.sources_for(c, [])) == 4                                           # nothing cited → everything offered


def test_visibility_fragment_is_used_in_gather_sql():
    v = Visibility(User("u", "vic", "viewer", "h"), ["reporter:crow"])
    assert "reporter:crow" in v.sql("ev.source_id")
    assert not v.allows("reporter:crow") and v.allows("bbc.co.uk")
