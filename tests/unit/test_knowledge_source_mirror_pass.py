"""The mirrored sources' share of a sync pass: they run first, apart from the namespace
sources' document bound, rotate, keep their state under the mirror, spend as the mirror,
and read their lists on from a cursor (docs/architecture/document-sources.md, Every
complete listing reads the lists again)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

import pytest

from server.clients import gdrive
from server.code.app.index.pass_budget import PassBudget
from server.knowledge.app import source_sync
from server.knowledge.store import mirrors as mirror_store
from server.llm import customer
from server.settings import settings
from tests.unit.test_knowledge_source_sync import DOCS, _detail, _file, _source
from tests.unit.test_knowledge_source_sync import world as world  # the fixture
from tests.unit.test_knowledge_source_sync_mirror import MIRROR, READERS, _mirrored, _own
from tests.unit.test_knowledge_source_sync_mirror import mirror as mirror  # the fixture


def _pass(world: Any, *sources: Any) -> Any:
    world.sources = list(sources)
    return asyncio.run(source_sync.run_sync())


def _files(n: int, **kw: Any) -> list[dict[str, Any]]:
    return [_file(f"doc{i}", DOCS, f"2026-01-0{i}T00:00:00.000Z", **kw) for i in range(1, n + 1)]


def test_the_mirrored_sources_run_first_under_their_own_document_bound(
    world: Any, mirror: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "knowledge_source_sync_documents_per_pass", 3)
    monkeypatch.setattr(settings, "knowledge_source_mirror_documents_per_pass", 1)
    world.files = _files(3, permissions=READERS)
    result = _pass(world, _source(name="notes"), _mirrored())
    homes = [written.namespace for written, _ in world.upserts]
    # The mirror's one document first, then every namespace document: the mirror's size
    # takes nothing from the namespace sources' bound.
    assert homes == [MIRROR, *[_source().namespace] * 3]
    assert result.sources == 2 and result.cut


def test_a_leg_takes_its_share_of_the_pass_and_never_ends_past_it() -> None:
    whole = PassBudget(started=0.0, stop_after=1e12)
    leg = source_sync._leg(whole, 1 / 3)
    assert leg.stop_after - leg.started == pytest.approx(1e12 / 3, rel=1e-6)
    ending = PassBudget(started=0.0, stop_after=leg.started + 1)
    assert source_sync._leg(ending, 1 / 3).stop_after == ending.stop_after


def test_the_mirrored_source_synced_longest_ago_goes_first_after_the_asked(
    world: Any, mirror: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real = source_sync._sync_one

    async def _seen(src: Any, *a: Any) -> Any:
        order.append(src.name)
        return await real(src, *a)

    monkeypatch.setattr(source_sync, "_sync_one", _seen)

    def row(**detail: str) -> dict[str, str]:
        return {"state_value": "", "detail": json.dumps(detail)}

    world.states = {
        "a": row(last_synced_at="2026-01-03T00:00:00+00:00"),
        "b": row(last_synced_at="2026-01-01T00:00:00+00:00"),
        "c": row(last_synced_at="2026-01-05T00:00:00+00:00", requested_at="2026-01-06T00:00:00"),
    }
    _pass(world, *(_mirrored(name=n) for n in "abc"))
    assert order == ["c", "b", "a"]


def test_a_mirrored_source_s_namespace_row_holds_only_its_codes(
    world: Any, mirror: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fails(token: str, file_id: str, **kw: Any) -> Any:
        raise gdrive.VendorError("503")

    monkeypatch.setattr(gdrive, "file_permissions", _fails)
    world.files = _files(2)  # listed without lists: each needs a read, and each fails
    world.states["decks"] = {"state_value": "", "detail": json.dumps({"requested_at": "t"})}
    _pass(world, _mirrored())
    shown = _detail(world)
    assert set(shown) == {"last_synced_at", "status"}  # the request was answered
    own = _own(world)
    assert own["changed"] == 2 and own["error"].startswith("permissions:")


def test_a_source_added_again_under_the_name_inherits_no_state(world: Any, mirror: Any) -> None:
    world.files = _files(1, permissions=READERS)
    revision = world.files[0]["modifiedTime"]
    # The removed source gave the file up; its state is under its own id.
    given_up = {"file_waits": {"doc1": [revision, 99, "export_failed"]}}
    world.states["id-decks"] = {"state_value": "", "detail": json.dumps(given_up)}
    _pass(world, _mirrored(source_id="id-again"))
    assert [written.namespace for written, _ in world.upserts] == [MIRROR]


def test_a_mirrored_turn_spends_as_its_mirror(
    world: Any, mirror: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    real = source_sync.sync_file

    async def _attributed(*a: Any, **kw: Any) -> Any:
        seen.append(customer.attribution().namespace)
        return await real(*a, **kw)

    monkeypatch.setattr(source_sync, "sync_file", _attributed)
    world.files = _files(1, permissions=READERS)
    _pass(world, _mirrored())
    assert seen == [MIRROR]


@pytest.fixture
def slow_lists(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Each file's list read separately, and the deadline after `per_turn` of them."""
    reads: dict[str, Any] = {"per_turn": 0, "now": 0, "ids": [], "rows": []}

    async def _read(token: str, file_id: str, **kw: Any) -> Any:
        reads["now"] += 1
        reads["ids"].append(file_id)
        return READERS

    async def _listing(org: str, ns: str, source_id: str, name: str, started: Any) -> None:
        reads["rows"].append(started)

    def _exhausted(self: PassBudget) -> bool:
        return bool(reads["per_turn"]) and reads["now"] >= reads["per_turn"]

    monkeypatch.setattr(gdrive, "file_permissions", _read)
    monkeypatch.setattr(mirror_store, "record_listing", _listing)
    monkeypatch.setattr(PassBudget, "exhausted", _exhausted)
    return reads


def _turn_with(world: Any, reads: dict[str, Any], per_turn: int) -> None:
    reads.update(per_turn=per_turn, now=0)
    reads["ids"].clear()
    world.files = _files(4)  # listed without lists every time
    _pass(world, _mirrored())


def test_the_lists_go_on_from_the_cursor_and_the_cycle_s_row_lands_past_the_wrap(
    world: Any, mirror: Any, slow_lists: dict[str, Any]
) -> None:
    _turn_with(world, slow_lists, 0)  # every file written, each with its list read
    assert len(slow_lists["rows"]) == 1
    first = slow_lists["rows"][-1]
    _turn_with(world, slow_lists, 2)  # doc1 and doc2 read; doc3 is where it stopped
    assert slow_lists["ids"] == ["doc1", "doc2"] and len(slow_lists["rows"]) == 1
    began = datetime.fromisoformat(_own(world)["lists_since"])
    assert began > first and _own(world)["lists_cursor"].endswith("/doc3/edit#id-decks")
    _turn_with(world, slow_lists, 3)  # doc3, doc4, then doc1 past the wrap: the cycle is done
    assert slow_lists["ids"] == ["doc3", "doc4", "doc1"]
    assert slow_lists["rows"][-1] == began  # the row carries the start of the cycle
    assert _own(world)["lists_cursor"].endswith("/doc2/edit#id-decks")


def test_a_turn_that_vouches_for_nothing_keeps_the_cycle(
    world: Any, mirror: Any, slow_lists: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _turn_with(world, slow_lists, 0)
    _turn_with(world, slow_lists, 2)
    kept = (_own(world)["lists_cursor"], _own(world)["lists_since"])

    async def _owed(org: str, namespace: str, uri_suffix: str) -> str:
        return "owed"

    monkeypatch.setattr(source_sync.documents, "publications_owed", _owed)
    rows = len(slow_lists["rows"])
    _turn_with(world, slow_lists, 0)  # every list read, but a write was not in ClickHouse
    assert len(slow_lists["rows"]) == rows
    assert (_own(world)["lists_cursor"], _own(world)["lists_since"]) == kept
