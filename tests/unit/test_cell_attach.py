"""`loom cell attach` (server/cli/cell_attach.py, cell_attach_beats.py, cell_attach_pg.py) on
fakes: every refusal the heartbeats, the Postgres reads and the router's answers make, the
waits and their budget, the placement batches and the rescan's watch. The real Postgres,
ClickHouse and router behaviour is tests/integration/test_cell_attach.py. CH-free."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from server.cli import cell_attach, cell_scan
from server.cli import cell_attach_beats as beats
from server.cli import cell_attach_pg as apg
from server.cli.cell_ch import Refusal, Refused
from server.cli.cell_scan import Table
from server.core import cell, instances
from server.router.limits import ATTACH_BATCH, MAX_BODY_BYTES
from server.settings import settings

FIELDS = {**cell.DEFAULTS, "deployment_id": "dep-1"}
CURRENT = cell.as_value(FIELDS)
NEXT = cell.as_value({**FIELDS, "oidc_client_id": "next"})


def _value(**keys: Any) -> dict[str, Any]:
    return {**CURRENT, "serves": True, "marked": True, "router": True, "router_why": "", **keys}


def _beat(
    i: str = "p1",
    *,
    cell_name: str | None = "first",
    live: bool = True,
    stopped: bool = False,
    mode: str = "recording",
    value: dict[str, Any] | None = None,
    keys_fp: str = "k",
) -> beats.Beat:
    return beats.Beat(
        instance_id=i,
        live=live,
        stopped=stopped,
        keys_fp=keys_fp,
        last_beat="2026-01-01 00:00:00.000",
        cell=cell_name,
        mode=mode if cell_name is not None else "",
        value=(value if value is not None else _value()) if cell_name is not None else None,
    )


def _judge(found: list[beats.Beat], stopped_unset: beats.StoppedUnset = "refuse") -> Any:
    return beats.judge(found, "first", (CURRENT, NEXT), stopped_unset)


def _reasons(verdict: beats.Verdict) -> list[str]:
    return [str(r) for r in verdict.refusals]


# -- the heartbeats --------------------------------------------------------------------


def test_a_healthy_cell_passes_and_reports_its_keys() -> None:
    nxt = {**_value(), **NEXT}
    verdict = _judge([_beat("a"), _beat("b", value=nxt)])
    assert (verdict.refusals, verdict.booting, verdict.live, verdict.keys_fps) == ([], [], 2, {"k"})


def test_a_live_process_with_loom_cell_unset_or_another_cell_refuses_at_once() -> None:
    verdict = _judge([_beat("a", cell_name=""), _beat("b", cell_name="other")], "wait")
    assert _reasons(verdict) == [
        "process a: runs with LOOM_CELL unset: set LOOM_CELL=first on it",
        "process b: runs with LOOM_CELL=other: set LOOM_CELL=first on it",
    ]


def test_a_stopped_unset_process_refuses_or_is_waited_on_as_asked() -> None:
    stopped = [
        _beat("a", cell_name="", live=False, stopped=True),
        _beat("b", cell_name=None, live=False, stopped=True),
    ]
    refused = _judge(stopped, "refuse")
    assert [r.subject for r in refused.refusals] == ["process a", "process b"]
    assert "may still be writing" in refused.refusals[0].reason
    waited = _judge(stopped, "wait")
    assert (waited.refusals, waited.aging) == ([], ["a", "b"])


def test_a_live_instance_with_no_cell_row_is_booting() -> None:
    verdict = _judge([_beat("a", cell_name=None)])
    assert (verdict.refusals, verdict.booting) == ([], ["a"])


def test_a_cell_row_without_the_current_keys_refuses_even_while_booting() -> None:
    old = {k: v for k, v in _value().items() if k not in ("serves", "marked", "router")}
    verdict = _judge([_beat("a", value=old, mode="")])
    assert _reasons(verdict) == [
        "process a: reports no serves, marked or router key: upgrade it to this release"
    ]


@pytest.mark.parametrize(
    "keys,booting,refused",
    [
        ({"marked": False}, [], "lose the tenancy marker"),
        ({"marked": None}, ["a"], ""),
        ({"router": False, "router_why": "a 404"}, [], "cannot read its router: a 404"),
        ({"router": None}, ["a"], ""),
        # A workers-only pod places no org: it needs no router credential.
        ({"serves": False, "router": False}, [], ""),
        ({"serves": False, "router": None}, [], ""),
    ],
)
def test_marked_and_router_refuse_or_wait(
    keys: dict[str, Any], booting: list[str], refused: str
) -> None:
    verdict = _judge([_beat("a", value=_value(**keys))])
    assert verdict.booting == booting
    assert (refused in " ".join(_reasons(verdict))) and bool(verdict.refusals) == bool(refused)


def test_no_tenancy_mode_yet_is_booting() -> None:
    # A pod whose first cell-state read failed: mode '', marked and router null.
    verdict = _judge([_beat("a", mode="", value=_value(marked=None, router=None))])
    assert (verdict.refusals, verdict.booting) == ([], ["a"])


def test_a_fingerprint_on_neither_current_nor_next_refuses() -> None:
    drifted = {**_value(), **cell.as_value({**FIELDS, "oidc_client_id": "drifted"})}
    assert _reasons(_judge([_beat("a", value=drifted)])) == [
        "process a: runs neither loom.cell_self's current nor its next configuration"
    ]


def test_the_final_refusals() -> None:
    scim = {**_value(), **cell.as_value({**FIELDS, "scim": True})}
    verdict = beats.judge(
        [_beat("a", value=scim)], "first", (cell.as_value({**FIELDS, "scim": True}),), "refuse"
    )
    assert [r.reason for r in beats.final_refusals(verdict)] == [
        "SCIM is on: attach with SCIM on needs the per-cell SCIM release"
    ]
    two = _judge([_beat("a"), _beat("b", keys_fp="other")])
    assert "root-secret fingerprints" in beats.final_refusals(two)[0].reason
    assert "no live process" in beats.final_refusals(_judge([]))[0].reason


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()

    async def _sleep(s: float) -> None:
        c.now += s

    monkeypatch.setattr(apg, "_sleep", _sleep)
    return c


@pytest.mark.asyncio
async def test_settle_waits_for_a_booting_process(clock: _Clock) -> None:
    reads = [[_beat("a", cell_name=None)], [_beat("a", cell_name=None)], [_beat("a")]]

    async def read() -> list[beats.Beat]:
        return reads.pop(0)

    wait = apg.Wait(10, clock)
    verdict = await beats.settle(read, _judge, wait)
    assert (verdict.live, wait.spent) == (1, 2.0)


@pytest.mark.asyncio
async def test_settle_refuses_at_the_end_of_wait_naming_what_it_waited_on(clock: _Clock) -> None:
    async def read() -> list[beats.Beat]:
        return [_beat("a", cell_name=None), _beat("b", cell_name="", live=False, stopped=True)]

    with pytest.raises(Refused) as e:
        await beats.settle(read, lambda f: _judge(f, "wait"), apg.Wait(3, clock))
    assert [str(r) for r in e.value.refusals] == [
        "process a: still booting at the end of --wait",
        "process b: stopped with LOOM_CELL unset, not aged out within --wait",
    ]


@pytest.mark.asyncio
async def test_settle_extends_once_for_a_process_first_seen_booting(clock: _Clock) -> None:
    seen = 0

    async def read() -> list[beats.Beat]:
        nonlocal seen
        seen += 1
        # A stopped process aging out first; then one booting from the third read (2 s
        # spent), done at the seventh, past the first deadline.
        if seen < 3:
            return [_beat("a"), _beat("x", cell_name="", live=False, stopped=True)]
        return [_beat("a"), _beat("b", cell_name=None)] if seen < 7 else [_beat("a"), _beat("b")]

    wait = apg.Wait(3, clock)
    verdict = await beats.settle(read, lambda f: _judge(f, "wait"), wait, extend_s=4)
    assert (verdict.live, wait.seconds) == (2, 6)

    async def forever() -> list[beats.Beat]:
        return [_beat("b", cell_name=None)]

    wait = apg.Wait(2, clock)
    with pytest.raises(Refused, match="still booting"):
        await beats.settle(forever, _judge, wait, extend_s=1)
    # Seen booting from the start: first sight plus the extension is inside the deadline.
    assert wait.seconds == 2


@pytest.mark.asyncio
async def test_wait_counts_only_its_own_pauses(clock: _Clock) -> None:
    wait = apg.Wait(2, clock)
    clock.now += 100  # a scan's time
    assert await wait.pause() and await wait.pause()
    assert not await wait.pause()
    assert wait.spent == 2.0


# -- the rescan's watch ----------------------------------------------------------------


def test_the_prune_contract_is_an_hour() -> None:
    assert instances.PRUNE_AFTER_S == 3600


@pytest.mark.asyncio
async def test_the_watch_refuses_every_unset_process_it_saw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rounds = [
        [
            _beat("a", cell_name=""),
            _beat("c"),
            _beat("d", cell_name=None, live=False, stopped=True),
        ],
        # a's rows were swept; c's cell row alone was swept: it was seen with the cell.
        [_beat("c", cell_name=None, live=False, stopped=True)],
    ]

    async def read_beats(_ch: Any, *, since: str | None = None) -> list[beats.Beat]:
        assert since == "2026-01-01 00:00:00.000"
        return rounds.pop(0)

    monkeypatch.setattr(beats, "read_beats", read_beats)
    clock = _Clock()
    watch = beats.Watch(None, "2026-01-01 00:00:00.000", "first", clock)
    await watch.read_once()
    await watch.read_once()
    assert [r.subject for r in watch.refusals()] == ["process a", "process d"]
    clock.now += beats.watch_period_s() + 1
    monkeypatch.setattr(beats, "read_beats", read_beats)
    rounds.append([])
    await watch.read_once()
    assert watch.refusals()[-1].subject == "heartbeats"


def test_the_watch_refuses_up_front_with_a_very_long_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "instance_heartbeat_seconds", 1180)
    assert beats.Watch(None, "", "first").refusals_up_front()
    monkeypatch.setattr(settings, "instance_heartbeat_seconds", 10)
    assert beats.Watch(None, "", "first").refusals_up_front() == []


# -- batches, ordering, arguments --------------------------------------------------------


def test_batches_keep_the_count_and_the_byte_bound() -> None:
    many = [f"org-{i:05d}" for i in range(ATTACH_BATCH + 1)]
    assert [len(b) for b in cell_attach.batches(many, "dep")] == [ATTACH_BATCH, 1]
    # Escaping grows a non-ASCII name six times: the encoded body is what counts.
    wide = ["é" * 2000 + str(i) for i in range(10)]
    out = cell_attach.batches(wide, "dep")
    assert len(out) > 1 and sum(map(len, out)) == 10
    for b in out:
        body = json.dumps({"deployment_id": "dep", "orgs": b}, separators=(",", ":"))
        assert len(body.encode()) <= MAX_BODY_BYTES


def test_move_sources_are_scanned_first() -> None:
    tables = [Table("loom", n, "MergeTree") for n in ("a", "memory_extraction_queue", "b", "src")]
    ordered = cell_scan.order_sources(tables, {"loom.src"})
    assert [t.name for t in ordered] == ["memory_extraction_queue", "src", "a", "b"]


def _args(**over: Any) -> argparse.Namespace:
    fields: dict[str, Any] = {
        "cell": "first",
        "wait": 10,
        "scan_budget": 100,
        "dry_run": False,
        "ch_allow_user": [],
        "ch_allow_user_directory": [],
        "ch_allow_object": [],
        "ch_allow_replica": [],
        "pg_allow_role": [],
        "router": "http://127.0.0.1:7900",
    }
    return argparse.Namespace(**{**fields, **over})


def test_the_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "cell", "first")
    monkeypatch.setattr(settings, "storage_authority", "postgres")
    monkeypatch.setattr(settings, "pg_dsn", "postgresql://cell.internal/loom")
    assert cell_attach._args_problem(_args()) == ""
    assert "LOOM_CELL" in cell_attach._args_problem(_args(cell="other"))
    assert "--scan-budget" in cell_attach._args_problem(_args(scan_budget=0))
    monkeypatch.setattr(settings, "storage_authority", "clickhouse")
    assert "Postgres" in cell_attach._args_problem(_args())


def test_an_org_past_the_limit_is_named_with_its_sources() -> None:
    found = cell_scan.ScanResult()
    found.add("postgres loom.tenancy", "x" * (cell.ORG_MAX_BYTES + 1))
    (refusal,) = cell_attach._too_long(found)
    assert refusal.subject.startswith(f"org of {cell.ORG_MAX_BYTES + 1} bytes")
    assert "found in postgres loom.tenancy" in refusal.reason


# -- the router ------------------------------------------------------------------------


def _router(*responses: httpx.Response) -> cell_attach.Router:
    answers = list(responses)

    def handler(_request: httpx.Request) -> httpx.Response:
        return answers.pop(0)

    return cell_attach.Router(
        httpx.URL("http://127.0.0.1:7900"), "first", httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_the_router_calls_refuse_as_the_router_answers() -> None:
    ok = {"cell": "first", "state": "attaching"}
    assert await _router(httpx.Response(200, json=ok)).call("") == ok
    with pytest.raises(Refused, match="not upgraded"):
        await _router(httpx.Response(404, json={"error": "no route"})).call("/attach/place", {})
    with pytest.raises(Refused, match=r"refused /attach/begin \(409\): use loom cell update"):
        await _router(httpx.Response(409, json={"error": "use loom cell update"})).call(
            "/attach/begin", {}
        )
    with pytest.raises(Refused, match="did not answer as the router does"):
        await _router(httpx.Response(502, text="bad gateway")).call("")


# -- the Postgres reads, on a fake session ---------------------------------------------


@dataclass
class FakeConn:
    """Answers by the first needle the SQL holds; a callable answer is called with the
    arguments; an exception answer is raised."""

    routes: list[tuple[str, Any]] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)

    def _answer(self, sql: str, args: tuple[Any, ...]) -> Any:
        for needle, answer in self.routes:
            if needle in sql:
                if isinstance(answer, BaseException):
                    raise answer
                return answer(*args) if callable(answer) else answer
        raise AssertionError(f"unrouted: {sql[:80]}")

    async def fetch(self, sql: str, *args: Any) -> Any:
        return self._answer(sql, args)

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        return self._answer(sql, args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        return self._answer(sql, args)

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append(sql)
        for needle, answer in self.routes:
            if needle in sql:
                if isinstance(answer, BaseException):
                    raise answer
                return answer(*args) if callable(answer) else str(answer)
        return "OK"

    def transaction(self) -> Any:
        conn = self

        class _Tx:
            async def __aenter__(self) -> None:
                conn.executed.append("BEGIN")

            async def __aexit__(self, *exc: Any) -> None:
                conn.executed.append("ROLLBACK" if exc[0] else "COMMIT")

        return _Tx()


def _seq(*values: Any) -> Callable[..., Any]:
    pending = list(values)
    return lambda *_a: pending.pop(0)


@pytest.mark.asyncio
async def test_the_session_checks_refuse_a_pooler() -> None:
    moved = FakeConn(
        [
            ("pg_backend_pid() AS pid", _seq((1, "t0"), (2, "t1"))),
            ("current_setting($1, true)", None),
        ]
    )
    refusals = await apg.direct_refusals(moved, "attach")
    assert [r.reason.split(":")[0] for r in refusals] == [
        "attach's session moved between backends",
        "attach's session loses loom.tenancy_checked",
    ]
    direct = FakeConn([("pg_backend_pid() AS pid", (1, "t0")), ("current_setting($1, true)", "on")])
    assert await apg.direct_refusals(direct, "attach") == []
    assert "RESET ALL" in direct.executed


@pytest.mark.asyncio
async def test_the_lock_refuses_when_held_or_not_on_this_backend() -> None:
    holder = [{"pid": 7, "application_name": "loom", "client": "10.0.0.1", "backend_start": "t"}]
    held = FakeConn([("pg_try_advisory_lock", False), ("FROM pg_locks l LEFT JOIN", holder)])
    with pytest.raises(Refused, match="held by pid 7 \\(loom, from 10.0.0.1"):
        await apg.take_lock(held, "attach")
    pooled = FakeConn([("pg_try_advisory_lock", True), ("SELECT EXISTS", False)])
    with pytest.raises(Refused, match="a pooler"):
        await apg.take_lock(pooled, "attach")
    assert any("tcp_keepalives_idle" in s for s in pooled.executed)


@pytest.mark.asyncio
async def test_a_database_not_utf8_refuses() -> None:
    assert await apg.encoding_refusals(FakeConn([("pg_encoding_to_char", "UTF8")])) == []
    (r,) = await apg.encoding_refusals(FakeConn([("pg_encoding_to_char", "SQL_ASCII")]))
    assert "SQL_ASCII, not UTF8" in r.reason


def test_privilege_holders_outside_the_allowed_refuse() -> None:
    holders = apg.Holders(
        own=["loom"],
        others={"ci": ["may insert into loom.tenancy"], "servers": ["owns loom.memory_current"]},
    )
    assert [r.subject for r in apg.privilege_refusals(holders, ["servers"])] == ["postgres role ci"]
    assert apg.privilege_refusals(holders, ["ci", "servers"]) == []


@pytest.mark.asyncio
async def test_settings_and_subscriptions_refuse() -> None:
    conn = FakeConn(
        [
            (
                "pg_db_role_setting",
                [{"role": "ci", "db": "every database", "cfg": "Loom.Tenancy_Checked=on"}],
            ),
            ("pg_subscription", [{"subname": "mirror", "subenabled": False}]),
        ]
    )
    refusals = await apg.unmarked_refusals(conn, True)
    assert [r.subject for r in refusals] == [
        "postgres unmarked session",
        "postgres setting for ci in every database",
        "postgres subscription mirror",
    ]


@pytest.mark.asyncio
async def test_own_role_lacks_and_prepared_transactions_refuse() -> None:
    conn = FakeConn(
        [
            ("SELECT current_user", "attacher"),
            ("WHERE NOT held", [{"need": "membership in pg_read_all_stats"}]),
            ("has_table_privilege($1, 'SELECT')", lambda t: t != "loom.memory_current"),
            ("max_prepared_transactions", 5),
        ]
    )
    refusals = await apg.own_role_refusals(conn, ["loom.memory_current", "loom.tenancy"])
    assert [str(r) for r in refusals] == [
        "postgres role attacher: attach's role lacks membership in pg_read_all_stats",
        "postgres role attacher: attach's role lacks SELECT on loom.memory_current",
        "postgres max_prepared_transactions: is 5: a prepared transaction outlives every "
        "session, so step 4 could not wait for it; set it to 0",
    ]


def _row(**over: Any) -> dict[str, Any]:
    return {
        "pid": 9,
        "backend_type": "client backend",
        "application_name": "ci",
        "xact_start": "t",
        "ours": True,
        "xid_old": False,
        "xmin_old": False,
        "hidden": False,
        **over,
    }


@pytest.mark.asyncio
async def test_step_4_refuses_what_it_cannot_wait_for(clock: _Clock) -> None:
    base = [("max_prepared_transactions", 0), ("pg_prepared_xacts", 0)]
    hidden = FakeConn([*base, ("pg_stat_activity s", [_row(hidden=True)])])
    with pytest.raises(Refused, match="hidden from attach's role"):
        await apg.wait_transactions(hidden, 100, apg.Wait(5, clock))
    prepared = FakeConn([("max_prepared_transactions", 0), ("pg_prepared_xacts", 1)])
    with pytest.raises(Refused, match="prepared transaction"):
        await apg.wait_transactions(prepared, 100, apg.Wait(5, clock))
    held = FakeConn(
        [
            *base,
            (
                "pg_stat_activity s",
                [_row(xmin_old=True), _row(pid=10, ours=False, xid_old=True), _row(pid=11)],
            ),
        ]
    )
    with pytest.raises(Refused) as e:
        await apg.wait_transactions(held, 100, apg.Wait(2, clock))
    assert [r.reason for r in e.value.refusals] == [
        "may have read `recording`: still open",
        "another database's transaction, holding the snapshot horizon",
    ]
    done = FakeConn([*base, ("pg_stat_activity s", [_row(pid=10, ours=False, xid_old=True)])])
    await apg.wait_transactions(done, 100, apg.Wait(0, clock))


class _Timeout(Exception):
    sqlstate = "55P03"


@pytest.mark.asyncio
async def test_the_mode_commit_retries_a_busy_lock_and_ends_an_earlier_run(clock: _Clock) -> None:
    tries = _seq(_Timeout(), "UPDATE 1")

    def lock(*_a: Any) -> str:
        answer = tries()
        if isinstance(answer, Exception):
            raise answer
        return "LOCK TABLE"

    holders = _seq(
        [{"pid": 5, "application_name": "attach", "xact_start": "t", "rolsuper": False}], []
    )
    conn = FakeConn(
        [
            ("LOCK TABLE", lock),
            ("UPDATE loom.tenancy_mode", "UPDATE 1"),
            ("l.mode = 'ExclusiveLock'", holders),
            ("rolsuper FROM pg_roles", False),
            ("pg_terminate_backend", True),
        ]
    )
    confirmed: list[bool] = []

    async def confirm() -> None:
        confirmed.append(True)

    async def ready() -> bool:
        return True

    logged: list[str] = []
    await apg.commit_enforced(conn, confirm, ready, apg.Wait(10, clock), logged.append)
    assert logged == ["the mode's lock is busy; retrying"]
    # Before each try and the one termination.
    assert len(confirmed) == 3


@pytest.mark.asyncio
async def test_a_superusers_mode_transaction_is_named_not_ended() -> None:
    conn = FakeConn(
        [
            (
                "l.mode = 'ExclusiveLock'",
                [{"pid": 5, "application_name": "psql", "xact_start": "t", "rolsuper": True}],
            ),
            ("rolsuper FROM pg_roles", False),
        ]
    )

    async def confirm() -> None:
        pass

    with pytest.raises(Refused, match="a superuser must end it"):
        await apg.end_mode_transactions(conn, confirm)


@pytest.mark.asyncio
async def test_the_mode_commit_refuses_past_wait_and_skips_a_try_while_booting(
    clock: _Clock,
) -> None:
    conn = FakeConn([("l.mode = 'ExclusiveLock'", [])])

    async def confirm() -> None:
        pass

    async def booting() -> bool:
        return False

    with pytest.raises(Refused, match="not taken within --wait"):
        await apg.commit_enforced(conn, confirm, booting, apg.Wait(2, clock), lambda _t: None)
    assert not any("LOCK TABLE" in s for s in conn.executed)


@pytest.mark.asyncio
async def test_the_durable_read_must_see_enforced_and_x_above_its_xid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.core import pg_profile

    async def synchronous() -> str:
        return "synchronous"

    monkeypatch.setattr(pg_profile, "standby_now", synchronous)
    run = cell_attach.Run(None, _args(), None, lambda _t: None)  # type: ignore[arg-type]
    run.lock_conn = FakeConn([("pg_snapshot_xmax", "100")])

    async def read(mode: str, xid: int) -> apg.Durable:
        return apg.Durable(mode=mode, first=(True, "first"), xid=xid)

    monkeypatch.setattr(apg, "durable_read", lambda: read("recording", 50))
    with pytest.raises(Refused, match="did not land"):
        await run.durable()
    monkeypatch.setattr(apg, "durable_read", lambda: read("enforced", 100))
    with pytest.raises(Refused, match="is not above 100"):
        await run.durable()
    monkeypatch.setattr(apg, "durable_read", lambda: read("enforced", 99))
    assert await run.durable() == 100

    async def waiting() -> str:
        return "waiting"

    monkeypatch.setattr(pg_profile, "standby_now", waiting)
    with pytest.raises(Refused, match="not synchronous before"):
        await run.durable()


@pytest.mark.asyncio
async def test_a_serving_row_is_done_only_once_the_cell_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import asynccontextmanager

    from server.core import pg

    state = ["enforced"]

    @asynccontextmanager
    async def dedicated(**_kw: Any) -> Any:
        yield None

    async def read_mode(_conn: Any) -> tuple[str, Any]:
        return state[0], (True, "first")

    monkeypatch.setattr(pg, "dedicated", dedicated)
    monkeypatch.setattr(apg, "read_mode", read_mode)
    run = cell_attach.Run(None, _args(), None, lambda _t: None)  # type: ignore[arg-type]
    assert await run.done()
    state[0] = "recording"
    with pytest.raises(Refused, match="run loom cell restore"):
        await run.done()


def test_refusal_text() -> None:
    assert str(Refusal("a", "b")) == "a: b"


@pytest.mark.asyncio
async def test_a_root_secret_change_during_attach_refuses_at_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = cell_attach.Run(None, _args(), None, lambda _t: None)  # type: ignore[arg-type]
    run.begun_keys_fp = "before"

    async def settle(*_a: Any, **_kw: Any) -> beats.Verdict:
        return beats.Verdict(live=1, keys_fps={"after"})

    monkeypatch.setattr(beats, "settle", settle)
    with pytest.raises(Refused, match="a root-secret rotation"):
        await run.final_checks(beats.Watch(None, "", "first"))
    run.begun_keys_fp = "after"
    await run.final_checks(beats.Watch(None, "", "first"))
