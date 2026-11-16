"""memory.search's term leg: which governed definitions a query names, in what order,
where they go in the served list, and what the leg reports when it adds nothing."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, cast

import pytest

from server.memory.api.models_search import MemorySearchInput, _MemorySearchHit
from server.memory.app.search import term_leg
from server.memory.app.search.context import SearchContext
from server.memory.domain.definition_check import SharedDefinition
from server.memory.domain.promotion_source import promotion_source
from server.settings import settings

NOW = datetime(2026, 10, 1)


def _row(key: str, term: str, aliases: list[str] | None = None, age_days: int = 1) -> dict:
    return {
        "memory_key": key,
        "value": json.dumps({"term": term, "aliases": aliases or [], "body": f"{term} is ..."}),
        "promoted_from": [promotion_source("o", "n", "-", "-", "-", key)],
        "promoted_at": NOW - timedelta(days=age_days),
        "promotion_event": f"e-{key}",
        "total": 0,
    }


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"rows": [], "lapsed": {}, "checked": [], "values_read": []}

    def names(row: dict) -> dict:
        value = json.loads(row["value"])
        return {**row, "term": value["term"], "aliases": value["aliases"]}

    async def shared_definitions(
        org: str, namespace: str, limit: int, *, strong: bool
    ) -> list[dict]:
        assert strong is False
        return [names(r) for r in state.get("listed", state["rows"])[:limit]]

    async def definition_values(org: str, namespace: str, keys: list[str]) -> dict[str, str]:
        state["values_read"].append(sorted(keys))
        return {r["memory_key"]: r["value"] for r in state["rows"] if r["memory_key"] in keys}

    async def source_reasons(
        org: str, namespace: str, defs: list[Any], *, strong: bool
    ) -> dict[str, Any]:
        assert strong is False  # the search path reads without sequential consistency
        state["checked"].append(sorted(d.memory_key for d in defs))
        return {d.memory_key: state["lapsed"].get(d.memory_key) for d in defs}

    monkeypatch.setattr(term_leg.shared_repo, "shared_definitions", shared_definitions)
    monkeypatch.setattr(term_leg.shared_repo, "definition_values", definition_values)
    monkeypatch.setattr(term_leg, "source_reasons", source_reasons)
    return state


async def _keys(query: str, excluded: frozenset[str] = frozenset()) -> tuple[str, list[str]]:
    leg = await term_leg._read("o", "n", query, set(excluded), None, "m")
    return leg.status, [h.memory_key for h in leg.hits]


async def test_a_query_naming_no_term_reads_no_source(store: dict[str, Any]) -> None:
    store["rows"] = [_row("definition/arr", "ARR")]
    assert await _keys("what did we ship") == ("complete", [])
    assert store["checked"] == [] and store["values_read"] == []


async def test_an_empty_namespace_reports_empty(store: dict[str, Any]) -> None:
    assert await _keys("what is ARR") == ("empty", [])


async def test_names_are_served_in_query_order_the_longer_first(store: dict[str, Any]) -> None:
    store["rows"] = [
        _row("definition/arr", "ARR", ["annual recurring revenue"]),
        _row("definition/arr-growth", "ARR growth"),
        _row("definition/churn", "Churn rate"),
    ]
    status, keys = await _keys("churn rate against ARR growth")
    assert status == "complete"
    assert keys == ["definition/churn", "definition/arr-growth", "definition/arr"]
    # only the named definitions' sources are read
    assert store["checked"] == [["definition/arr", "definition/arr-growth", "definition/churn"]]


async def test_a_lapsed_or_excluded_definition_is_not_served(store: dict[str, Any]) -> None:
    store["rows"] = [_row("definition/arr", "ARR"), _row("definition/mrr", "MRR")]
    store["lapsed"] = {"definition/mrr": "changed"}
    assert await _keys("ARR and MRR") == ("complete", ["definition/arr"])
    assert await _keys("ARR and MRR", frozenset({"definition/arr"})) == ("complete", [])


async def test_a_name_two_hold_goes_to_the_latest_promotion(store: dict[str, Any]) -> None:
    store["rows"] = [
        _row("definition/new", "ARR", age_days=1),
        _row("definition/old", "Annual revenue", ["ARR"], age_days=5),
    ]
    assert await _keys("what is ARR") == ("complete", ["definition/new"])
    # the older one keeps the names it holds alone
    assert await _keys("annual revenue") == ("complete", ["definition/old"])


async def test_a_copy_not_promoted_from_its_namespace_wide_row_is_not_served(
    store: dict[str, Any],
) -> None:
    row = _row("definition/arr", "ARR")
    row["promoted_from"] = [promotion_source("o", "n", "bot", "-", "-", "definition/arr")]
    store["rows"] = [row]
    assert await _keys("what is ARR") == ("complete", [])


async def test_past_the_ceiling_the_leg_reports_capped(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "memory_shared_definitions_max", 1)
    store["rows"] = [_row("definition/arr", "ARR"), _row("definition/mrr", "MRR", age_days=3)]
    assert await _keys("ARR and MRR") == ("capped", ["definition/arr"])


def _input(**over: Any) -> MemorySearchInput:
    return MemorySearchInput(**{"org": "o", "namespace": "n", "query": "what is ARR", **over})


@pytest.mark.parametrize(
    "over",
    [
        {"memory_type": "decision"},
        {"tags": ["x"]},
        {"category": "c"},
        {"as_of": NOW},
        {"min_body_chars": 10},
        {"recent_first": 2},
        {"anchor_first": 1},
        {"legs": ["sessions"]},
        {"page": "p1"},
        {"query": "what is \ud800"},
    ],
)
def test_a_field_asking_for_other_rows_turns_the_leg_off(over: dict[str, Any]) -> None:
    assert term_leg.leg_on(_input(**over)) is False


@pytest.mark.parametrize(
    "over",
    [
        {},
        {"tags": ["x"], "tags_soft": True},
        {"category": "c", "category_soft": True},
        {"agent": "bot", "user_id": "u", "session_id": "s", "scope": "shared"},
        {"exclude_keys": ["k"], "min_similarity": 0.4, "legs": ["memory", "sessions"]},
    ],
)
def test_a_field_saying_whose_rows_to_read_keeps_it_on(over: dict[str, Any]) -> None:
    assert term_leg.leg_on(_input(**over)) is True


def test_a_merged_pass_runs_no_leg() -> None:
    merged = _input()
    merged._merged_into_parent = True
    assert term_leg.leg_on(merged) is False


async def test_a_leg_past_its_bound_or_failing_adds_nothing() -> None:
    async def slow() -> term_leg.TermLeg:
        await asyncio.sleep(1)
        return term_leg.TermLeg("complete")

    async def broken() -> term_leg.TermLeg:
        raise RuntimeError("down")

    def leg(coro: Any) -> asyncio.Task[term_leg.TermLeg]:
        return asyncio.create_task(coro)

    assert (await term_leg.outcome(leg(slow()), 0.01)).status == "timed_out"
    assert (await term_leg.outcome(leg(broken()), 1)).status == "failed"
    assert (await term_leg.outcome(None, 1)).status == "off"


def _hit(key: str, rank: float, *, source: str = "fact", parent: str = "") -> _MemorySearchHit:
    return _MemorySearchHit(
        memory_key=key,
        memory_type="semantic",
        content_excerpt=key,
        distance=0.5,
        similarity=0.5,
        rank_score=rank,
        source=source,
        parent_memory_key=parent,
    )


def _governed(key: str) -> _MemorySearchHit:
    row = _row(key, key)
    d = SharedDefinition(
        memory_key=key,
        value=json.loads(row["value"]),
        promoted_from=tuple(row["promoted_from"]),
        promoted_at=row["promoted_at"],
        promotion_event=row["promotion_event"],
    )
    return term_leg._hit(d, "n")


def test_governed_hits_lead_after_a_computed_row_and_push_the_rest_to_the_tail() -> None:
    derived = _hit("derived_aggregate", 0.9 + 1e-6, source="computed")  # ceiling + eps
    a, b, c = _hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)
    tail = [_hit("d", 0.6)]
    governed = [_governed("definition/x"), _governed("definition/y")]
    served, rest = term_leg.merge([derived, a, b, c], governed, tail, "n", top_k=4)
    assert [h.memory_key for h in served] == [
        "derived_aggregate",
        "definition/x",
        "definition/y",
        "a",
    ]
    assert [h.memory_key for h in rest] == ["b", "c", "d"]
    assert derived.rank_score > governed[0].rank_score > governed[1].rank_score > a.rank_score


def test_another_row_under_a_governed_key_is_left_out() -> None:
    private = _hit("definition/x", 0.9)
    chunk = _hit("chunk:s:1", 0.8, source="chunk", parent="definition/x")
    other = _hit("a", 0.7)
    served, rest = term_leg.merge(
        [private, chunk, other], [_governed("definition/x")], [_hit("definition/x", 0.1)], "n", 5
    )
    assert [h.memory_key for h in served] == ["definition/x", "a"]
    assert served[0].governed is True
    assert rest == []


def test_governed_hits_past_top_k_are_cut_the_later_first_and_lead_the_next_page() -> None:
    governed = [_governed("definition/x"), _governed("definition/y")]
    served, rest = term_leg.merge([_hit("a", 0.9)], governed, [], "n", top_k=1)
    assert [h.memory_key for h in served] == ["definition/x"]
    assert [h.memory_key for h in rest] == ["definition/y", "a"]


def test_a_governed_hit_is_marked_and_shaped_as_a_shared_row() -> None:
    hit = _governed("definition/x")
    wire = hit.model_dump()
    assert wire["governed"] is True and wire["memory_type"] == "definition"
    assert wire["similarity"] == 0.0 and wire["distance"] == 1.0
    assert "governed" not in _hit("a", 0.5).model_dump()


def test_loom_search_keeps_governed_hits_first_through_its_floors() -> None:
    from server.memory.app.loom_search.answer_first import answer_first

    governed = {"memory_key": "definition/x", "governed": True, "similarity": 0.0}
    lead = {"memory_key": "a", "similarity": 0.9, "namespace": "n", "source": "fact"}
    weak = {"memory_key": "b", "similarity": 0.05, "namespace": "n", "source": "fact"}
    payload = {"memory": {"results": [lead, governed, weak]}, "knowledge": {"results": []}}
    out = answer_first(payload, "n")
    assert [h["memory_key"] for h in out["memory"]["results"]] == ["definition/x", "a"]


def test_governed_hits_count_within_the_tracker_verdicts_memory_cap() -> None:
    from server.memory.app.loom_search.answer_first import _restore_governed
    from server.tracker.domain.defect_answer import VERDICT_MEMORY_MAX

    governed = [{"memory_key": f"definition/{i}", "governed": True} for i in range(2)]
    rest = [{"memory_key": f"r{i}"} for i in range(VERDICT_MEMORY_MAX)]
    payload = {
        "tracker": {"verdict": "duplicate"},
        "memory": {"results": rest},
        "withheld": {"reason": "tracker_verdict_leads"},
    }
    _restore_governed(payload, governed)
    keys = [h["memory_key"] for h in payload["memory"]["results"]]
    assert keys[:2] == ["definition/0", "definition/1"] and len(keys) == VERDICT_MEMORY_MAX
    assert payload["withheld"]["memory"] == 2


def test_a_merged_decision_never_goes_above_a_governed_hit() -> None:
    from server.memory.app.loom_search.repo_decisions import merge_memory_hits

    governed = _governed("definition/x")
    decision = _hit("d", 0.0)
    decision.similarity = 0.99
    hits, added = merge_memory_hits([governed, _hit("a", 0.5)], [decision], "n")
    assert [h.memory_key for h in hits][0] == "definition/x"
    assert added == [decision]


async def test_a_name_goes_to_its_latest_promotion_even_when_the_query_misses_that_one(
    store: dict[str, Any],
) -> None:
    """`CHURN` (newer) holds the name `churn` that `Churn` (older) also claims; the query
    matches only `Churn`, case-folded, and still serves neither copy of the name."""
    store["rows"] = [
        _row("definition/new", "CHURN", age_days=1),
        _row("definition/old", "Churn", age_days=5),
    ]
    assert await _keys("what is churn") == ("complete", [])
    assert store["checked"] == [["definition/new", "definition/old"]]
    # the same when the winner is excluded
    assert await _keys("what is churn", frozenset({"definition/new"})) == ("complete", [])


class _Ctx:
    """The fields of a SearchContext that `term_leg.join` reads and writes."""

    def __init__(self) -> None:
        self.input = _input(top_k=3)
        self.search_budget: Any = None
        self.page_tail: list[Any] = []
        self.candidates_total = 4
        self.actual_work: dict[str, Any] = {}
        self.term_leg_status = "off"


def _context() -> SearchContext:
    return cast(SearchContext, _Ctx())


async def test_join_places_the_hits_and_clears_the_abstain() -> None:
    governed = _governed("definition/x")

    async def found() -> term_leg.TermLeg:
        return term_leg.TermLeg("complete", [governed])

    ctx = _context()
    task = asyncio.create_task(found())
    hits, verdict = await term_leg.join(ctx, task, [_hit("a", 0.5)], (True, "no_match", "x"))
    assert [h.memory_key for h in hits] == ["definition/x", "a"]
    assert verdict == (False, "", "") and ctx.term_leg_status == "complete"
    assert ctx.candidates_total == 5 and ctx.actual_work["term_leg_served"] == 1


async def test_the_leg_scores_its_hits_by_their_shared_embedding(
    store: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    store["rows"] = [_row("definition/arr", "ARR"), _row("definition/mrr", "MRR")]

    async def vectors(
        org: str, ns: str, idents: list[Any], *, model: str, sequential: bool
    ) -> dict[Any, Any]:
        assert model == "m" and sequential is False
        return {i: [0.6, 0.8] for i in idents if i[3] == "definition/arr"}

    monkeypatch.setattr(term_leg, "live_vectors", vectors)
    leg = await term_leg._read("o", "n", "ARR and MRR", set(), [1.0, 0.0], "m")
    arr, mrr = leg.hits
    assert arr.similarity == pytest.approx(0.6) and arr.distance == pytest.approx(0.4)
    assert (mrr.similarity, mrr.distance) == (0.0, 1.0)  # no shared embedding


async def test_names_are_matched_from_the_value_that_is_served(store: dict[str, Any]) -> None:
    """A re-promotion between the two reads dropped the alias the query names."""
    store["rows"] = [_row("definition/arr", "ARR", ["annual recurring revenue"])]
    listed = dict(store["rows"][0])
    store["rows"][0]["value"] = json.dumps({"term": "ARR", "aliases": [], "body": "..."})
    store["listed"] = [listed]
    assert await _keys("annual recurring revenue") == ("complete", [])


async def test_join_with_no_hit_keeps_the_verdict() -> None:
    async def none() -> term_leg.TermLeg:
        return term_leg.TermLeg("empty")

    ctx = _context()
    task = asyncio.create_task(none())
    hits, verdict = await term_leg.join(ctx, task, [_hit("a", 0.5)], (True, "no_match", "x"))
    assert [h.memory_key for h in hits] == ["a"] and verdict == (True, "no_match", "x")
    assert ctx.term_leg_status == "empty"


def test_a_dated_reading_order_keeps_governed_hits_first() -> None:
    from server.memory.app.ask.rendering import evidence_order

    governed = _governed("definition/x")
    governed.created_at = governed.valid_at = "2019-01-01"  # promoted long ago
    old, new = _hit("old", 0.5), _hit("new", 0.5)
    old.valid_at, new.valid_at = "2020-01-01", "2026-01-01"
    ordered, mode = evidence_order([old, governed, new], None, "what is the latest ARR")
    assert mode == "reverse_chronological"
    assert [h.memory_key for h in ordered] == ["definition/x", "new", "old"]


def test_restated_value_check_neither_retires_nor_is_retired_by_a_governed_hit() -> None:
    from server.memory.app.ask.supersession import resolve_templated_conflicts

    governed = _governed("definition/ceo")
    governed.content_excerpt = "The CEO of Acme is Ann Lee"
    newer = _hit("ceo-note", 0.5)
    newer.content_excerpt, newer.created_at = "The CEO of Acme is Bob Ray", "2026-01-01"
    kept, retired = resolve_templated_conflicts([governed, newer])
    assert kept == [governed, newer] and retired == []


def test_the_ask_pool_puts_governed_hits_first_whichever_search_served_them() -> None:
    from server.memory.app.ask.pool import governed_first

    governed = _governed("definition/x")
    governed.similarity = 0.0
    pool = [_hit(f"r{i}", 0.9) for i in range(30)]  # reranked past a 15-row window
    # a sub-search's governed hit, appended after the primary search's rows
    ordered = governed_first([*pool, governed])
    assert ordered[0] is governed and len(ordered) == 31
    # a stage that dropped it decided so: it is not put back
    assert governed_first(pool) == pool


async def test_join_waits_the_bound_and_a_re_entrant_pass_only_its_time_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.memory.app.search.budget import SearchBudget

    async def slow() -> term_leg.TermLeg:
        await asyncio.sleep(0.05)
        return term_leg.TermLeg("complete")

    monkeypatch.setattr(settings, "memory_term_leg_wait_ms", 1000)
    first = _context()
    first.search_budget = SearchBudget(max_rounds=0, max_wall_ms=0)
    await term_leg.join(first, asyncio.create_task(slow()), [], (True, "", ""))
    assert first.term_leg_status == "complete"  # a first pass is not gated by the budget
    again = _context()
    again.input._pass_depth = 1
    again.search_budget = SearchBudget(max_rounds=0, max_wall_ms=0)
    await term_leg.join(again, asyncio.create_task(slow()), [], (True, "", ""))
    assert again.term_leg_status == "timed_out"


@pytest.mark.parametrize("bodies", ["unit", "source"])
async def test_bodies_leave_a_governed_hit_s_text_alone(
    bodies: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another row under the governed key would answer a read by key: none is made."""
    from server.memory import store as memory_store
    from server.memory.app.search import payload

    read: list[list[str]] = []

    async def get_many_by_keys(identity: Any, keys: list[str], **kw: Any) -> list[Any]:
        read.append(keys)
        return []

    class _Links:
        async def query_all(self, sql: str, params: dict[str, Any]) -> list[Any]:
            read.append(params["keys"])
            return []

    monkeypatch.setattr(memory_store, "get_many_by_keys", get_many_by_keys)
    monkeypatch.setattr(payload, "get_client", _Links)
    governed = _governed("definition/x")
    text = governed.content_excerpt
    hits = [governed, _hit("a", 0.5)]
    await payload._expand_unit_bodies(hits, _input(bodies=bodies))
    if bodies == "source":
        await payload._expand_source_bodies(hits, _input(bodies=bodies))
    assert read and all("definition/x" not in keys for keys in read)
    assert governed.content_excerpt == text


async def test_tags_and_usage_are_not_read_by_a_governed_hit_s_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.memory.app.search import payload

    keys_read: list[list[str]] = []

    async def read_row_tags(org: str, nss: list[str], keys: list[str]) -> dict[Any, Any]:
        keys_read.append(keys)
        return {}

    monkeypatch.setattr(payload, "_read_row_tags", read_row_tags)
    governed = _governed("definition/x")
    await payload._fill_hit_tags([governed], _input(include_tags=True), ["n"])
    assert keys_read == [] and governed.tags == []
    clients: list[None] = []

    def client() -> Any:
        clients.append(None)
        raise RuntimeError("no ClickHouse in a unit test")

    monkeypatch.setattr(payload, "get_client", client)
    await payload._fill_hit_usage([governed], _input(include_usage=True), ["n"])
    assert clients == []  # a governed hit's counters are not read by its key
    assert governed.used_count == 0 and governed.harm_count == 0


async def test_a_definition_in_another_script_is_served_as_written(store: dict[str, Any]) -> None:
    store["rows"] = [_row("definition/sales", "売上")]
    # stored as memory.promote stores it: every non-ASCII character escaped
    store["rows"][0]["value"] = json.dumps({"term": "売上", "aliases": [], "body": "月次の売上"})
    leg = await term_leg._read("o", "n", "今月の売上は", set(), None, "m")
    assert [h.memory_key for h in leg.hits] == ["definition/sales"]
    assert "月次の売上" in leg.hits[0].content_excerpt


def test_a_governed_hit_a_computed_row_pushes_out_leads_the_tail() -> None:
    derived = _hit("derived_aggregate", 9.0, source="computed")
    governed = [_governed("definition/x")]
    served, rest = term_leg.merge([derived, _hit("a", 0.5)], governed, [], "n", top_k=1)
    assert [h.memory_key for h in served] == ["derived_aggregate"]
    assert [h.memory_key for h in rest] == ["definition/x", "a"]


async def test_a_parked_governed_hit_stays_governed_and_reads_no_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A governed hit cut to the next page keeps its flag, and its tags are still not read
    by its key there."""
    from server.memory.app import search_pages
    from server.memory.app.search import payload

    keys_read: list[list[str]] = []

    async def read_row_tags(org: str, nss: list[str], keys: list[str]) -> dict[Any, Any]:
        keys_read.append(keys)
        return {}

    monkeypatch.setattr(payload, "_read_row_tags", read_row_tags)
    _, tail = term_leg.merge(
        [_hit("a", 0.5)], [_governed("definition/x"), _governed("definition/y")], [], "n", 1
    )
    page = [search_pages.unparked(search_pages._parked(h)) for h in tail]
    assert [(h.memory_key, h.governed) for h in page if h] == [
        ("definition/y", True),
        ("a", None),
    ]
    await payload._fill_hit_tags([h for h in page if h], _input(include_tags=True), ["n"])
    assert keys_read == [["a"]]
    assert page[0] is not None and page[0].tags == []


def test_the_verdict_cap_holds_when_the_verdict_cut_nothing() -> None:
    from server.memory.app.loom_search.answer_first import _restore_governed
    from server.tracker.domain.defect_answer import VERDICT_MEMORY_MAX

    governed = [{"memory_key": f"definition/{i}", "governed": True} for i in range(2)]
    rest = [{"memory_key": f"r{i}"} for i in range(VERDICT_MEMORY_MAX - 1)]
    payload: dict[str, Any] = {"tracker": {"verdict": "new"}, "memory": {"results": rest}}
    _restore_governed(payload, governed)
    assert len(payload["memory"]["results"]) == VERDICT_MEMORY_MAX
    assert payload["withheld"] == {"memory": 1, "reason": "tracker_verdict_leads"}


async def test_whole_bodies_serve_a_long_definition_whole(store: dict[str, Any]) -> None:
    from server.memory.app.search import payload

    body = "Annual recurring revenue, counted per contract. " * 40  # past any excerpt
    store["rows"] = [_row("definition/arr", "ARR")]
    store["rows"][0]["value"] = json.dumps({"term": "ARR", "aliases": [], "body": body})
    leg = await term_leg._read("o", "n", "what is ARR", set(), None, "m", whole=True)
    (hit,) = leg.hits
    assert body in hit.content_excerpt
    returned, elided, tokens = await payload._expand_unit_bodies([hit], _input(bodies="unit"))
    assert (returned, elided) == (1, 0) and tokens > 0 and not hit.body_truncated


def test_loom_search_and_ask_keep_a_computed_row_above_governed_hits() -> None:
    from server.memory.app.ask.pool import governed_first
    from server.memory.app.loom_search.answer_first import answer_first

    count = {"memory_key": "derived_aggregate", "source": "computed", "similarity": 0.95}
    governed = {"memory_key": "definition/arr", "governed": True, "similarity": 0.0}
    fact = {"memory_key": "a", "similarity": 0.9, "namespace": "n", "source": "fact"}
    out = answer_first({"memory": {"results": [count, governed, fact]}}, "n")
    assert [h["memory_key"] for h in out["memory"]["results"]] == [
        "derived_aggregate",
        "definition/arr",
        "a",
    ]
    hit = _governed("definition/arr")
    derived = _hit("derived_aggregate", 1.0, source="computed")
    ordered = governed_first([_hit("a", 0.9), derived, hit])
    assert [h.memory_key for h in ordered] == ["derived_aggregate", "definition/arr", "a"]


async def test_launch_serves_whole_bodies_when_the_search_asks_for_them(
    store: dict[str, Any],
) -> None:
    body = "Annual recurring revenue, counted per contract. " * 40
    store["rows"] = [_row("definition/arr", "ARR")]
    store["rows"][0]["value"] = json.dumps({"term": "ARR", "aliases": [], "body": body})
    for bodies, whole in (("unit", True), ("source", True), ("full", False)):
        task = term_leg.launch(_input(bodies=bodies), None, "m")
        assert task is not None
        (hit,) = (await term_leg.outcome(task, 5)).hits
        assert (body in hit.content_excerpt) is whole
