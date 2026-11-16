"""`loom cell attach` (server/cli/cell_attach.py) against a real Postgres with a synchronous
standby, a real ClickHouse and the router served in process: a prepared first cell in
`recording` with orgs in both stores attaches to `serving` with every org placed and the
triggers created; a rerun on `serving` exits 0; a refusal after step 3 leaves `enforced` and
a rerun resumes from there; a second run is refused at the attach lock; the Postgres reads
attach makes refuse what they name (an unchecked writer's setting, a subscription, a role
that can make an org held, a transaction that may have read `recording`).

The cell's Postgres and the directory are databases of the test's own; the configured
ClickHouse databases are scratch ones; the cell's processes are heartbeat rows the test
writes. Needs LOOM_PG_DSN (a server with a synchronous standby), LOOM_ROUTER_PG_DSN and
LOOM_CH_HOST."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from server.cli import cell as cell_cli
from server.cli import cell_attach, cell_attach_pg, cell_prepare
from server.cli.cell_ch import Refused
from server.core import cell, cell_probe, cell_triggers, instances, pg
from server.core.ch import get_client
from server.core.schema import pg_migrate
from server.router import app as router_app
from server.router import directory
from server.settings import settings
from tests.integration._pg_database import create_database, drop_database

OPERATOR = "lco_" + "2" * 128
CREDENTIAL_SHA = hashlib.sha256(b"lcr_first").hexdigest()
ROUTER = "http://127.0.0.1:7901"
HOST = "loom.example"


class _Network(httpx.AsyncBaseTransport):
    """What attach and the router reach: the router at ROUTER, the shared host's health
    route, the cell's upstream main port (:7777) and its probe listener (any other)."""

    def __init__(self) -> None:
        self.listener = httpx.ASGITransport(app=cell_probe.listener_app())
        self.router = httpx.ASGITransport(
            app=router_app.build_app(router_app.Router(probe_transport=self))
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1" and request.url.port == 7901:
            return await self.router.handle_async_request(request)
        if request.url.host == HOST or request.url.port == 7777:
            return httpx.Response(200, json=cell_probe.answer())
        return await self.listener.handle_async_request(request)


async def _admin(sql: str, *args: Any) -> Any:
    import asyncpg

    conn = await asyncpg.connect(os.environ["LOOM_PG_DSN"])
    try:
        return await conn.execute(sql, *args)
    finally:
        await conn.close()


@pytest.fixture
async def cell_stores(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[dict[str, Any]]:
    cell_dsn = os.environ.get("LOOM_PG_DSN") or pytest.skip("LOOM_PG_DSN not set")
    router_dsn = os.environ.get("LOOM_ROUTER_PG_DSN") or pytest.skip("LOOM_ROUTER_PG_DSN not set")
    os.environ.get("LOOM_CH_HOST") or pytest.skip("LOOM_CH_HOST not set")
    cell_db, own_cell = await create_database(cell_dsn)
    router_db, own_router = await create_database(router_dsn)
    deployment = f"dep-{uuid.uuid4().hex[:8]}"
    tag = f"it-{uuid.uuid4().hex[:8]}-"
    for name, value in {
        "cell": "first",
        "storage_authority": "postgres",
        "pg_dsn": own_cell,
        "document_s3_bucket": "",
        "router_pg_dsn": own_router,
        "router_shared_host": HOST,
        "router_upstream_probe_url": "http://listener.internal:7790",
        "router_upstream_url": "http://upstream.internal:7777",
        "router_operator_sha256": hashlib.sha256(OPERATOR.encode()).hexdigest(),
        "router_operator_token": OPERATOR,
        "ch_cluster": "",
        "public_base_url": f"https://{HOST}",
        "oauth_issuer": "",
    }.items():
        monkeypatch.setattr(settings, name, value)
    await pg.close_pool()
    monkeypatch.setattr(directory, "_pool", None)
    # A step's poll interval and the waits stay short.
    monkeypatch.setattr(cell_attach_pg, "POLL_S", 0.2)
    ch = get_client()
    await ch.connect()
    ids: list[str] = []
    try:
        await pg_migrate.migrate()
        await pg.execute(
            "INSERT INTO loom.authority (authority, epoch, deployment_id) "
            "VALUES ('postgres', 1, $1)",
            deployment,
        )
        await directory.migrate()
        await ch.insert(
            "loom.storage_authority",
            [{"authority": "postgres", "epoch": 10**12, "deployment_id": deployment}],
        )
        cell.set_deployment_id(deployment)
        yield {"ch": ch, "ids": ids, "tag": tag, "deployment": deployment, "db": cell_db}
    finally:
        cell.set_deployment_id("")
        await ch.execute(
            "ALTER TABLE loom.storage_authority DELETE WHERE deployment_id = {d:String} "
            "SETTINGS mutations_sync = 2",
            {"d": deployment},
        )
        for table in (instances.TABLE, instances.CELL_TABLE):
            await ch.execute(
                f"ALTER TABLE {table} DELETE WHERE has({{ids:Array(String)}}, instance_id)",
                {"ids": ids},
            )
        await ch.execute(
            "DELETE FROM loom.namespaces WHERE startsWith(org, {p:String})", {"p": tag}
        )
        await ch.disconnect()
        await pg.close_pool()
        await directory.close()
        await drop_database(cell_dsn, cell_db)
        await drop_database(router_dsn, router_db)


def _value(**keys: Any) -> str:
    reported = {
        **cell.as_value(cell.mine()),
        "serves": True,
        "marked": True,
        "router": True,
        "router_why": "",
        **keys,
    }
    return json.dumps(reported, sort_keys=True)


async def _beat(
    stores: dict[str, Any],
    instance_id: str,
    *,
    cell_name: str | None = "first",
    mode: str = "recording",
    stopped: bool = False,
    **keys: Any,
) -> None:
    """One process's heartbeat rows, as `instances.tick` writes them."""
    ch = stores["ch"]
    if instance_id not in stores["ids"]:
        stores["ids"].append(instance_id)
    await ch.insert(
        instances.TABLE, [{**instances.row(stopped=stopped), "instance_id": instance_id}]
    )
    if cell_name is not None:
        await ch.insert(
            instances.CELL_TABLE,
            [
                {
                    "instance_id": instance_id,
                    "cell": cell_name,
                    "tenancy_mode": mode,
                    "fingerprint": _value(**keys),
                }
            ],
        )


async def _prepare(stores: dict[str, Any], network: _Network) -> None:
    pid = f"it-{uuid.uuid4().hex[:8]}"
    await _beat(stores, pid, cell_name="first", mode="open")
    args = argparse.Namespace(
        cell="first",
        host=HOST,
        credential_sha256=CREDENTIAL_SHA,
        probe_url="",
        wait=0,
        router=ROUTER,
    )
    router = cell_cli.operator_router(args)
    assert isinstance(router, httpx.URL)
    assert await cell_prepare.prepare(stores["ch"], args, router, network) == 0
    stores["pid"] = pid
    await _beat(stores, pid, mode="recording")


def _args(**over: Any) -> argparse.Namespace:
    fields: dict[str, Any] = {
        "cell": "first",
        "wait": 20,
        "scan_budget": 600,
        "dry_run": False,
        "ch_allow_user": [settings.ch_user or "default"],
        "ch_allow_user_directory": [],
        "ch_allow_object": [],
        "ch_allow_replica": [],
        "pg_allow_role": [],
        "router": ROUTER,
    }
    return argparse.Namespace(**{**fields, **over})


async def _attach(stores: dict[str, Any], network: _Network, **over: Any) -> int:
    args = _args(**over)
    url = cell_cli.operator_router(args)
    assert isinstance(url, httpx.URL)
    return await cell_attach.attach(stores["ch"], args, url, network)


async def _mode() -> str:
    return str(await pg.fetchval("SELECT mode FROM loom.tenancy_mode"))


async def _placed() -> set[str]:
    import asyncpg

    conn = await asyncpg.connect(settings.router_pg_dsn)
    try:
        rows = await conn.fetch("SELECT org FROM loom_router.placements WHERE cell = 'first'")
    finally:
        await conn.close()
    return {r["org"] for r in rows}


async def _row() -> Any:
    return await directory.first_cell_named("first")


@pytest.mark.asyncio
async def test_a_recording_first_cell_attaches_to_serving(cell_stores, capsys) -> None:
    stores, network = cell_stores, _Network()
    tag = stores["tag"]
    await _prepare(stores, network)
    capsys.readouterr()
    # Orgs a recording cell holds: a tenancy row alone, a Postgres data row, a ClickHouse
    # row, and one name no request can carry (skipped and named).
    await pg.execute("INSERT INTO loom.tenancy (org) VALUES ($1)", tag + "signed-in")
    await pg.execute("INSERT INTO loom.scope_generations (org, scope) VALUES ($1, '*')", tag + "pg")
    await stores["ch"].insert(
        "loom.namespaces",
        [
            {"org": tag + "ch", "namespace": "n", "owner_email": ""},
            {"org": tag + "bad\x00org", "namespace": "n", "owner_email": ""},
        ],
    )
    assert await _attach(stores, network) == 0
    report = json.loads(capsys.readouterr().out)
    mine = {tag + "signed-in", tag + "pg", tag + "ch"}
    assert report["state"] == "serving"
    assert await _mode() == "enforced"
    placed = await _placed()
    assert {o for o in placed if o.startswith(tag)} == mine
    held = {r["org"] for r in await pg.fetch("SELECT org FROM loom.tenancy")}
    assert held == placed
    assert any("clickhouse loom.namespaces" in k for k in report["skipped"]["malformed"])
    row = await _row()
    assert (row.state, row.mode, row.placed) == ("serving", "enforced", len(placed))
    async with pg.dedicated() as conn:
        assert await cell_triggers.missing(conn) == []
    # A rerun on `serving` is done.
    assert await _attach(stores, network) == 0
    assert json.loads(capsys.readouterr().out)["done"] is True


@pytest.mark.asyncio
async def test_a_refusal_after_step_3_leaves_enforced_and_a_rerun_resumes(
    cell_stores, monkeypatch, capsys
) -> None:
    stores, network = cell_stores, _Network()
    await _prepare(stores, network)
    org = stores["tag"] + "a"
    await pg.execute("INSERT INTO loom.scope_generations (org, scope) VALUES ($1, '*')", org)
    # A process with LOOM_CELL unset appears once the mode is committed: refused at once.
    real = cell_attach.Run.durable

    async def durable_then_unset(self: Any) -> int:
        xmax = await real(self)
        await _beat(stores, f"it-unset-{uuid.uuid4().hex[:6]}", cell_name="")
        return xmax

    monkeypatch.setattr(cell_attach.Run, "durable", durable_then_unset)
    assert await _attach(stores, network) == 1
    assert "runs with LOOM_CELL unset" in capsys.readouterr().err
    assert await _mode() == "enforced"
    row = await _row()
    assert (row.state, row.mode) == ("attaching", "enforced")
    # Placed live once enforced; the rerun resumes from the durable read.
    monkeypatch.setattr(cell_attach.Run, "durable", real)
    for i in list(stores["ids"]):
        if i.startswith("it-unset-"):
            await _beat(stores, i, cell_name="first", mode="enforced")
    await _beat(stores, stores["pid"], mode="enforced")
    assert await _attach(stores, network) == 0
    assert org in await _placed()
    assert (await _row()).state == "serving"


@pytest.mark.asyncio
async def test_a_second_run_is_refused_at_the_lock(cell_stores, capsys) -> None:
    stores, network = cell_stores, _Network()
    await _prepare(stores, network)
    async with pg.dedicated() as holder:
        await cell_attach_pg.take_lock(holder, "the test")
        assert await _attach(stores, network) == 1
        assert "attach lock: held by pid" in capsys.readouterr().err
    assert await _mode() == "recording"


@pytest.mark.asyncio
async def test_a_dry_run_writes_nothing(cell_stores, capsys) -> None:
    stores, network = cell_stores, _Network()
    await _prepare(stores, network)
    capsys.readouterr()
    org = stores["tag"] + "dry"
    await pg.execute("INSERT INTO loom.scope_generations (org, scope) VALUES ($1, '*')", org)
    assert await _attach(stores, network, dry_run=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["per_source"]["postgres loom.scope_generations"] == {
        "count": 1,
        "sample": [org],
    }
    assert await _mode() == "recording"
    assert await _placed() == set()
    assert await pg.fetchval("SELECT count(*) FROM loom.tenancy") == 0
    async with pg.dedicated() as conn:
        assert any(d.kind == "absent" for d in await cell_triggers.missing(conn))


@pytest.mark.asyncio
async def test_an_org_past_the_limit_refuses_before_any_tenancy_row(cell_stores, capsys) -> None:
    stores, network = cell_stores, _Network()
    await _prepare(stores, network)
    long = "o" * (cell.ORG_MAX_BYTES + 1)
    await pg.execute("INSERT INTO loom.scope_generations (org, scope) VALUES ($1, '*')", long)
    await pg.execute("INSERT INTO loom.scope_generations (org, scope) VALUES ('org-ok', '*')")
    assert await _attach(stores, network) == 1
    assert f"longer than {cell.ORG_MAX_BYTES} bytes" in capsys.readouterr().err
    assert await _mode() == "recording"
    assert await pg.fetchval("SELECT count(*) FROM loom.tenancy") == 0
    assert (await _row()).placed == 0


@pytest.mark.asyncio
async def test_settings_and_subscriptions_that_pass_unchecked_writers_refuse(
    cell_stores, capsys
) -> None:
    stores, network = cell_stores, _Network()
    await _prepare(stores, network)
    db = stores["db"]
    role = f"it_role_{uuid.uuid4().hex[:6]}"
    await _admin(f"CREATE ROLE {role} LOGIN")
    try:
        # Stored as first spelled: a fresh session that never set the placeholder.
        await _admin(f'ALTER ROLE {role} IN DATABASE "{db}" SET "Loom.Tenancy_Checked" = \'on\'')
        async with pg.dedicated() as conn:
            setting = await conn.fetchval(
                "SELECT setconfig::text FROM pg_db_role_setting s JOIN pg_roles r "
                "ON r.oid = s.setrole WHERE r.rolname = $1",
                role,
            )
            assert "Loom.Tenancy_Checked" in setting
            refusals = await cell_attach_pg.unmarked_refusals(conn, False)
        assert [r.subject for r in refusals] == [f"postgres setting for {role} in {db}"]
        # Another database's entry does not refuse.
        await _admin(f'ALTER ROLE {role} IN DATABASE "{db}" RESET ALL')
        await _admin(
            f"ALTER ROLE {role} IN DATABASE postgres SET session_replication_role = 'replica'"
        )
        async with pg.dedicated() as conn:
            assert await cell_attach_pg.unmarked_refusals(conn, False) == []
        # A role that can insert a tenancy row, a column grant alone, refuses attach.
        await pg.execute(f"GRANT USAGE ON SCHEMA loom TO {role}")
        await pg.execute(f"GRANT INSERT (org) ON loom.tenancy TO {role}")
        assert await _attach(stores, network) == 1
        err = capsys.readouterr().err
        assert f"postgres role {role}: may insert into loom.tenancy" in err
        assert await _mode() == "recording"
        assert await _attach(stores, network, pg_allow_role=[role], dry_run=True) == 0
    finally:
        await pg.execute(f"REVOKE ALL ON loom.tenancy FROM {role}")
        await pg.execute(f"REVOKE ALL ON SCHEMA loom FROM {role}")
        await _admin(f"ALTER ROLE {role} IN DATABASE postgres RESET ALL")
        await _admin(f"DROP ROLE {role}")


@pytest.mark.asyncio
async def test_a_disabled_subscription_refuses(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    name = f"it_sub_{uuid.uuid4().hex[:6]}"
    await pg.execute(
        f"CREATE SUBSCRIPTION {name} CONNECTION 'host=127.0.0.1 dbname=none' "
        "PUBLICATION none WITH (connect = false, enabled = false)"
    )
    try:
        async with pg.dedicated() as conn:
            refusals = await cell_attach_pg.unmarked_refusals(conn, False)
        assert [str(r) for r in refusals] == [
            f"postgres subscription {name}: disabled: its apply worker fires no origin "
            "trigger; drop it"
        ]
    finally:
        await pg.execute(f"ALTER SUBSCRIPTION {name} SET (slot_name = NONE)")
        await pg.execute(f"DROP SUBSCRIPTION {name}")


@pytest.mark.asyncio
async def test_step_4_waits_for_a_transaction_that_may_have_read_recording(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    async with pg.dedicated() as writer, pg.dedicated() as poller:
        tx = writer.transaction()
        await tx.start()
        await writer.execute("INSERT INTO loom.scope_generations (org, scope) VALUES ('w', '*')")
        # X is read after a later transaction completed, as after the durable read.
        await poller.execute("BEGIN; SELECT pg_current_xact_id(); COMMIT")
        xmax = await cell_attach_pg.snapshot_xmax(poller)
        with pytest.raises(Refused, match="may have read `recording`"):
            await cell_attach_pg.wait_transactions(poller, xmax, cell_attach_pg.Wait(0.5))
        await tx.rollback()
        # A session idle outside a transaction holds nothing: the wait passes.
        await cell_attach_pg.wait_transactions(poller, xmax, cell_attach_pg.Wait(0.5))


@pytest.mark.asyncio
async def test_the_session_checks_and_the_encoding(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    async with pg.dedicated() as conn:
        assert await cell_attach_pg.direct_refusals(conn, "attach") == []
        assert await cell_attach_pg.encoding_refusals(conn) == []
    async with pg.dedicated(marked=False) as conn:
        refusals = await cell_attach_pg.direct_refusals(conn, "attach")
        assert [r.reason.split(":")[0] for r in refusals] == [
            f"attach's session loses {pg.TENANCY_MARKER}"
        ]
    name = f"it_latin_{uuid.uuid4().hex[:6]}"
    await _admin(
        f"CREATE DATABASE {name} ENCODING 'LATIN1' LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
    )
    try:
        import asyncpg

        from tests.integration._pg_database import with_database

        conn = await asyncpg.connect(with_database(os.environ["LOOM_PG_DSN"], name))
        try:
            (refusal,) = await cell_attach_pg.encoding_refusals(conn)
            assert "LATIN1, not UTF8" in refusal.reason
        finally:
            await conn.close()
    finally:
        await _admin(f"DROP DATABASE {name}")


@pytest.mark.asyncio
async def test_the_mode_commit_waits_out_a_for_share_holder(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    logged: list[str] = []

    async def ok() -> bool:
        return True

    async with pg.dedicated() as holder, pg.dedicated() as conn:
        tx = holder.transaction()
        await tx.start()
        await holder.execute("SELECT mode FROM loom.tenancy_mode FOR SHARE")

        async def release() -> None:
            await asyncio.sleep(3)
            await tx.rollback()

        task = asyncio.create_task(release())
        await cell_attach_pg.commit_enforced(conn, ok, ok, cell_attach_pg.Wait(20), logged.append)
        await task
    assert await _mode() == "enforced"
    assert logged == ["the mode's lock is busy; retrying"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "grant,why",
    [
        ("GRANT INSERT (org) ON loom.tenancy TO {r}", "may insert into loom.tenancy"),
        ("GRANT UPDATE (org) ON loom.tenancy TO {r}", "may update loom.tenancy (org)"),
        ("GRANT UPDATE (mode) ON loom.tenancy_mode TO {r}", "may update loom.tenancy_mode (mode)"),
        ("GRANT INSERT (mode) ON loom.tenancy_mode TO {r}", "may insert into loom.tenancy_mode"),
        (
            "GRANT INSERT, TRUNCATE ON loom.tenancy_mode TO {r}",
            "may insert into loom.tenancy_mode",
        ),
        (
            "GRANT TRIGGER ON loom.scope_generations TO {r}",
            "holds TRIGGER on loom.scope_generations",
        ),
        ("GRANT CREATE ON SCHEMA public TO {r}", "may create in schema public"),
        ("GRANT CREATE ON SCHEMA loom TO {r}", "may create in schema loom"),
        (
            "GRANT SET ON PARAMETER session_replication_role TO {r}",
            "may set session_replication_role",
        ),
        (
            "SET loom.tenancy_checked = 'off'; "
            "GRANT ALTER SYSTEM ON PARAMETER loom.tenancy_checked TO {r}",
            "may ALTER SYSTEM SET loom.tenancy_checked",
        ),
        ("GRANT pg_write_server_files TO {r}", "is a member of pg_write_server_files"),
        # A member of the owner role, ADMIN only: it can grant itself the rest.
        ("GRANT {owner} TO {r} WITH ADMIN TRUE, INHERIT FALSE, SET FALSE", "owns loom.tenancy"),
    ],
)
async def test_each_right_that_makes_an_org_held_is_named(
    cell_stores, grant: str, why: str
) -> None:
    await _prepare(cell_stores, _Network())
    role = f"it_r_{uuid.uuid4().hex[:6]}"
    owner = await pg.fetchval("SELECT current_user")
    await _admin(f"CREATE ROLE {role} LOGIN")
    try:
        async with pg.dedicated() as conn:
            await conn.execute("GRANT USAGE ON SCHEMA loom TO " + role)
            for sql in grant.format(r=role, owner=owner).split("; "):
                await conn.execute(sql)
            holders = await cell_attach_pg.privilege_holders(conn, exempt_self=True)
        assert owner in holders.own
        assert any(why in w for w in holders.others.get(role, [])), holders.others
    finally:
        async with pg.dedicated() as conn:
            await conn.execute(f"DROP OWNED BY {role}")
        await _admin(f"DROP OWNED BY {role}")
        await _admin(f"DROP ROLE {role}")


@pytest.mark.asyncio
async def test_a_function_owner_counts_though_it_owns_no_table(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    role = f"it_f_{uuid.uuid4().hex[:6]}"
    await _admin(f"CREATE ROLE {role} LOGIN")
    try:
        async with pg.dedicated() as conn:
            await conn.execute(f"ALTER FUNCTION loom.tenancy_refuse_unheld() OWNER TO {role}")
            holders = await cell_attach_pg.privilege_holders(conn, exempt_self=True)
            await conn.execute("ALTER FUNCTION loom.tenancy_refuse_unheld() OWNER TO CURRENT_USER")
        assert holders.others[role] == ["owns loom.tenancy_refuse_unheld()"]
    finally:
        await _admin(f"DROP ROLE {role}")


@pytest.mark.asyncio
async def test_a_repeatable_read_snapshot_from_before_x_holds_step_4(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    async with pg.dedicated() as reader, pg.dedicated() as poller:
        tx = reader.transaction(isolation="repeatable_read")
        await tx.start()
        await reader.fetchval("SELECT mode FROM loom.tenancy_mode")  # its snapshot, no xid
        await poller.execute("BEGIN; SELECT pg_current_xact_id(); COMMIT")
        xmax = await cell_attach_pg.snapshot_xmax(poller)
        with pytest.raises(Refused, match="may have read `recording`"):
            await cell_attach_pg.wait_transactions(poller, xmax, cell_attach_pg.Wait(0.5))
        await tx.rollback()


@pytest.mark.asyncio
async def test_prepare_is_refused_while_attach_holds_the_lock(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    args = argparse.Namespace(
        cell="first",
        host=HOST,
        credential_sha256=CREDENTIAL_SHA,
        probe_url="",
        wait=0,
        router=ROUTER,
    )
    router = cell_cli.operator_router(args)
    assert isinstance(router, httpx.URL)
    async with pg.dedicated() as holder:
        await cell_attach_pg.take_lock(holder, "attach")
        with pytest.raises(cell_prepare.Refused, match="attach lock: held by pid"):
            await cell_prepare.prepare(cell_stores["ch"], args, router, _Network())


@pytest.mark.asyncio
async def test_doctor_names_the_same_holders_settings_and_subscriptions(cell_stores) -> None:
    from server.cli import doctor_cell

    await _prepare(cell_stores, _Network())
    role = f"it_d_{uuid.uuid4().hex[:6]}"
    await _admin(f"CREATE ROLE {role} LOGIN")
    try:
        await pg.execute(f"GRANT USAGE ON SCHEMA loom TO {role}")
        await pg.execute(f"GRANT INSERT ON loom.tenancy TO {role}")
        await _admin(f"ALTER ROLE {role} SET session_replication_role = 'replica'")
        lines = {n: (g, d) for n, g, d in await doctor_cell._writer_reads(())}
        assert lines["cell_tenancy_privilege"][0] == "WARN"
        assert f"postgres role {role}" in lines["cell_tenancy_privilege"][1]
        assert lines["cell_unchecked_writers"][0] == "WARN"
        allowed = {n: g for n, g, _d in await doctor_cell._writer_reads((role,))}
        assert allowed["cell_tenancy_privilege"] == "PASS"
    finally:
        await pg.execute(f"REVOKE ALL ON loom.tenancy FROM {role}")
        await pg.execute(f"REVOKE ALL ON SCHEMA loom FROM {role}")
        await _admin(f"ALTER ROLE {role} RESET ALL")
        await _admin(f"DROP ROLE {role}")


@pytest.mark.asyncio
async def test_doctor_names_stray_orgs_smallest_tables_first(cell_stores) -> None:
    from server.cli import doctor_cell

    stores = cell_stores
    await _prepare(stores, _Network())
    tag = stores["tag"]
    await pg.execute("INSERT INTO loom.tenancy (org) VALUES ($1)", tag + "held")
    await pg.execute(
        "INSERT INTO loom.scope_generations (org, scope) VALUES ($1, '*')", tag + "held"
    )
    await pg.execute(
        "INSERT INTO loom.scope_generations (org, scope) VALUES ($1, '*')", tag + "stray"
    )
    lines = {n: (g, d) for n, g, d in await doctor_cell._stray_reads(stores["ch"], 300)}
    grade, detail = lines["cell_stray_orgs"]
    assert grade == "WARN" and tag + "stray" in detail and tag + "held" not in detail


@pytest.mark.asyncio
async def test_doctor_reads_smallest_first_and_leaves_the_largest_unchecked(
    cell_stores, monkeypatch
) -> None:
    from server.cli import cell_scan, doctor_cell

    stores = cell_stores
    await _prepare(stores, _Network())
    tag = stores["tag"]
    await pg.execute("CREATE TABLE loom.it_small (org text PRIMARY KEY)")
    await pg.execute("CREATE TABLE loom.it_large (org text PRIMARY KEY)")
    await pg.execute("INSERT INTO loom.it_small VALUES ($1)", tag + "small")
    await pg.execute(
        "INSERT INTO loom.it_large SELECT $1 || g FROM generate_series(1, 20000) g", tag + "l"
    )
    await pg.execute("ANALYZE loom.it_small, loom.it_large")
    # The budget runs out on the largest table, as it does on a real cell when that table's
    # read is what is too slow: every smaller one was read first.
    real, read = cell_scan.pg_orgs, []

    async def pg_orgs(table: str, budget: Any) -> list[str]:
        read.append(table)
        if table == "loom.it_large":
            await asyncio.sleep(budget.left() + 0.1)
        return await real(table, budget)

    monkeypatch.setattr(cell_scan, "pg_orgs", pg_orgs)
    lines = {n: (g, d) for n, g, d in await doctor_cell._stray_reads(stores["ch"], 3)}
    assert read[-1] == "loom.it_large" and "loom.it_small" in read
    assert tag + "small" in lines["cell_stray_orgs"][1]
    grade, detail = lines["cell_stray_orgs_unchecked"]
    assert grade == "WARN" and detail.endswith(": loom.it_large")


@pytest.mark.asyncio
async def test_a_role_with_the_grants_but_not_the_owner_role_is_refused_before_any_write(
    cell_stores,
) -> None:
    from server.core import cell_triggers

    await _prepare(cell_stores, _Network())
    role = f"it_o_{uuid.uuid4().hex[:6]}"
    await _admin(f"CREATE ROLE {role} LOGIN")
    try:
        async with pg.dedicated() as conn:
            await conn.execute(f"GRANT USAGE ON SCHEMA loom TO {role}")
            await conn.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA loom TO {role}"
            )
            await conn.execute(f"SET ROLE {role}")
            rows = list(await conn.fetch(cell_triggers.ORG_TABLES))
            refusals = await cell_attach_pg.own_role_refusals(conn, rows)
            lacking = [r.reason for r in refusals if "owner role" in r.reason]
            assert "attach's role lacks membership in the owner role of loom.tenancy" in lacking
            assert any("loom.scope_generations" in r for r in lacking)
            # The trigger pass tells the cause apart: denied, not a busy lock.
            outcome = await cell_triggers.ensure_table(
                conn, "scope_generations", recording_first=True
            )
            assert outcome.answer == "denied"
            await conn.execute("RESET ROLE")
            assert await _mode() == "recording"
    finally:
        async with pg.dedicated() as conn:
            await conn.execute(f"DROP OWNED BY {role}")
        await _admin(f"DROP OWNED BY {role}")
        await _admin(f"DROP ROLE {role}")


@pytest.mark.asyncio
async def test_doctor_names_a_table_it_cannot_read_and_reads_the_rest(cell_stores) -> None:
    from server.cli import doctor_cell

    stores = cell_stores
    await _prepare(stores, _Network())
    tag = stores["tag"]
    # An org table with no org-led index: its skip scan refuses, the read goes on.
    await pg.execute("CREATE TABLE loom.it_no_org_index (id int PRIMARY KEY, org text)")
    await pg.execute("INSERT INTO loom.it_no_org_index VALUES (1, $1)", tag + "unindexed")
    await pg.execute(
        "INSERT INTO loom.scope_generations (org, scope) VALUES ($1, '*')", tag + "stray"
    )
    lines = {n: (g, d) for n, g, d in await doctor_cell._stray_reads(stores["ch"], 300)}
    assert tag + "stray" in lines["cell_stray_orgs"][1]
    grade, detail = lines["cell_stray_orgs_unread"]
    assert grade == "WARN" and "postgres loom.it_no_org_index" in detail
    assert "cell_stray_orgs_unchecked" not in lines


@pytest.mark.asyncio
async def test_attach_exempts_the_role_it_runs_as_and_doctor_does_not(cell_stores) -> None:
    await _prepare(cell_stores, _Network())
    role = f"it_ci_{uuid.uuid4().hex[:6]}"
    await _admin(f"CREATE ROLE {role} LOGIN")
    try:
        async with pg.dedicated() as conn:
            await conn.execute(f"GRANT USAGE ON SCHEMA loom TO {role}")
            await conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA loom TO {role}")
            await conn.execute(f"GRANT INSERT ON loom.tenancy TO {role}")
            await conn.execute(f"SET ROLE {role}")
            as_attach = await cell_attach_pg.privilege_holders(conn, exempt_self=True)
            as_doctor = await cell_attach_pg.privilege_holders(conn, exempt_self=False)
            await conn.execute("RESET ROLE")
        # Attach runs as the operator's role; a CI role doctor runs as is warned about.
        assert role in as_attach.own and role not in as_attach.others
        assert role not in as_doctor.own
        assert "may insert into loom.tenancy" in as_doctor.others[role]
    finally:
        async with pg.dedicated() as conn:
            await conn.execute(f"DROP OWNED BY {role}")
        await _admin(f"DROP OWNED BY {role}")
        await _admin(f"DROP ROLE {role}")
