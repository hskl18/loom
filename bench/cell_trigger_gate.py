"""The first attach's trigger-cost gate, end to end: `memory.set` p50 and p99 through a server
of an attached cell, and a delivery-sized 1,000-row insert of a held org from a marked and an
unmarked session, with the cell's org triggers on and dropped in turn (the `loom.tenancy`
guard stays). Run it against an attached cell's server, which must not serve anything else,
with a writer token for a placed org; it drops and recreates the cell's org triggers, so
never against a deployment in use. Needs no provider.

    PYTHONPATH=$PWD .venv/bin/python bench/cell_trigger_gate.py \\
        --base-url http://127.0.0.1:7932 --pg-dsn <the cell's LOOM_PG_DSN> \\
        --token <writer token> --org <placed org> --namespace <its namespace>
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
import uuid

import asyncpg
import httpx

from server.core import cell_triggers

CALLS, WARMUP, ROUNDS, INSERTS = 400, 50, 3, 20


async def _triggers(dsn: str, on: bool) -> None:
    """Create or drop the org triggers, and check that what was asked holds: every guarded
    table carries both triggers enabled at origin, or none does."""
    conn = await asyncpg.connect(dsn)
    try:
        guarded = [t for t in await cell_triggers.planned_tables(conn) if t != "tenancy"]
        for table in guarded:
            for name in cell_triggers.ORG_TRIGGERS:
                await conn.execute(f'DROP TRIGGER IF EXISTS {name} ON loom."{table}"')
                if on:
                    await conn.execute(cell_triggers._create(f'loom."{table}"', name))
        enabled = await conn.fetchval(
            "SELECT count(*) FROM pg_trigger WHERE tgenabled = 'O' AND tgname = ANY($1::text[])",
            list(cell_triggers.ORG_TRIGGERS),
        )
    finally:
        await conn.close()
    expected = 2 * len(guarded) if on else 0
    if enabled != expected:
        raise SystemExit(f"{enabled} org triggers enabled, expected {expected}")


async def _memory_set(args: argparse.Namespace) -> tuple[float, float]:
    """p50 and p99 in ms of CALLS sequential `memory.set` calls, each a new key."""
    lat = []
    headers = {"Authorization": f"Bearer {args.token}"}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        for i in range(CALLS + WARMUP):
            body = {
                "org": args.org,
                "namespace": args.namespace,
                "agent": "bench",
                "memory_key": f"gate-{uuid.uuid4().hex[:10]}",
                "value": f"trigger gate {i}",
            }
            t = time.perf_counter()
            (await client.post(f"{args.base_url}/v1/memory.set", json=body)).raise_for_status()
            if i >= WARMUP:
                lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    return lat[len(lat) // 2], lat[int(0.99 * (len(lat) - 1))]


async def _batch(args: argparse.Namespace, marked: bool) -> float:
    """The median in ms of INSERTS 1,000-row inserts of the held org, in one statement each."""
    settings = {"loom.tenancy_checked": "on"} if marked else {}
    conn = await asyncpg.connect(args.pg_dsn, server_settings=settings)
    try:
        times = []
        for _ in range(INSERTS):
            t = time.perf_counter()
            await conn.execute(
                "INSERT INTO loom.scope_generations (org, scope) "
                "SELECT $1, 'gate-' || g || '-' || $2 FROM generate_series(1, 1000) g",
                args.org,
                uuid.uuid4().hex[:8],
            )
            times.append((time.perf_counter() - t) * 1000)
        return statistics.median(times)
    finally:
        await conn.close()


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", required=True)
    p.add_argument("--pg-dsn", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--org", required=True)
    p.add_argument("--namespace", required=True)
    args = p.parse_args()
    try:
        for _ in range(ROUNDS):
            for on in (True, False):
                await _triggers(args.pg_dsn, on)
                p50, p99 = await _memory_set(args)
                marked, unmarked = await _batch(args, True), await _batch(args, False)
                print(
                    f"triggers {'on ' if on else 'off'}: memory.set p50 {p50:.1f} ms "
                    f"p99 {p99:.1f} ms; 1,000-row insert {marked:.1f} ms marked, "
                    f"{unmarked:.1f} ms unmarked",
                    flush=True,
                )
    finally:
        await _triggers(args.pg_dsn, True)


if __name__ == "__main__":
    asyncio.run(main())
