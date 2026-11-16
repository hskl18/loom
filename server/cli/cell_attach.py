"""`loom cell attach`: turn a prepared first cell into a `serving` one
(docs/architecture/cells.md, "Attaching a cell"; the runbook is
docs/operations/cell-attach.md).

A run holds the attach lock on a direct session of the cell's Postgres for its life, and
goes, each step idempotent, so a rerun resumes by the directory row and the cell's mode:

1. the directory row: on `serving` it is done (exit 0) once the cell reads `enforced`;
2. the refusals, before any write: the stores are the registered ones, the session is
   direct and marked, the cell's processes (their heartbeats), the roles, settings and
   subscriptions that would let an unchecked writer through, the ClickHouse writers,
   the table list, durability, attach's own grants;
3. `attach/begin` (the router probes the stores and writes the fingerprints);
4. the mode: `recording`, or `enforced` on a resume;
5. the triggers (`cell_triggers.ensure_table`), under the migration lock one try at a time;
6. in `recording`, step 2's scan, refusing an org past `cell.ORG_MAX_BYTES`, and the
   tenancy rows of the orgs found; the triggers again; the pre-commit checks;
7. step 3: `attach/enforced`, then the mode commit (never on a resume): from it on the mode
   only rises, so every later refusal leaves the cell `enforced` and `attaching`;
8. the durable read of `enforced`, the heartbeats, step 4's wait for every transaction
   that could have read `recording`, the rescan, the tenancy rows and their placements
   (`attach/place`), the final checks and `attach/serving`.

`--wait` bounds the waits, `--scan-budget` each full scan. `--dry-run` runs the refusals
and the scan and prints what would be placed, writing nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from server.cli import cell as cell_cli
from server.cli import cell_attach_beats as beats
from server.cli import cell_attach_pg as apg
from server.cli import cell_scan
from server.cli._ch import writes
from server.cli.cell_ch import Refusal, Refused, read_topology
from server.core import cell, cell_triggers
from server.settings import settings

_WAIT_S = 600  # not a setting: a flag's default
_SCAN_BUDGET_S = 7200  # not a setting: a flag's default
_LOCK_STATEMENT_MS = 60_000  # not a setting: a bound on the lock session's catalog reads
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)  # not a setting: one health read


def _args_problem(args: argparse.Namespace) -> str:
    if not cell.NAME.fullmatch(args.cell or ""):
        return f"{args.cell!r} is no cell name (1 to 32 of a-z, 0-9, -)"
    if args.cell != settings.cell:
        return f"LOOM_CELL is {settings.cell!r}: attach runs with LOOM_CELL={args.cell}"
    if settings.storage_authority != "postgres" or not settings.pg_dsn:
        return (
            "a cell keeps its state in Postgres: set LOOM_STORAGE_AUTHORITY=postgres and "
            "LOOM_PG_DSN"
        )
    if args.wait < 0 or args.scan_budget <= 0:
        return "--wait is 0 or more seconds, --scan-budget more than 0"
    return ""


class Router:
    """The router's operator API for one cell."""

    def __init__(
        self, url: httpx.URL, name: str, transport: httpx.AsyncBaseTransport | None
    ) -> None:
        self.url, self.name, self.transport = url, name, transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=cell_cli._OPERATOR_TIMEOUT,
            trust_env=self.url.scheme == "https",
            transport=self.transport,
            headers={"Authorization": f"Bearer {settings.router_operator_token}"},
        )

    async def call(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """The router's 200 answer naming the cell; Refused with its refusal otherwise."""
        url = cell_cli.at(self.url, f"/cells/v1/operator/cells/{self.name}" + path)
        try:
            async with self._client() as client:
                r = await (client.get(url) if body is None else client.post(url, json=body))
        except httpx.HTTPError as e:
            raise _refused("router", f"no answer ({type(e).__name__}): rerun attach") from None
        answer = cell_cli.router_answer(r, {"cell": self.name}, cell_cli.OPERATOR_REFUSALS)
        if answer is None:
            raise _refused("router", cell_cli.not_the_router(r))
        if r.status_code == 404 and answer.get("code") != "NO_SUCH_CELL":
            raise _refused("router", f"a router replica is not upgraded: it does not answer {url}")
        if r.status_code != 200:
            raise _refused(
                "router", f"refused {path or 'the read'} ({r.status_code}): {answer['error']}"
            )
        return answer

    async def health(self, host: str) -> str:
        """The deployment id `https://<host>/v1/health/deployment` answers ('' when none)."""
        try:
            async with httpx.AsyncClient(
                timeout=_HEALTH_TIMEOUT, transport=self.transport
            ) as client:
                r = await client.get(f"https://{host}/v1/health/deployment")
            answer = r.json()
        except (httpx.HTTPError, ValueError):
            return ""
        return str(answer.get("deployment_id") or "") if isinstance(answer, dict) else ""


def _too_long(found: cell_scan.ScanResult) -> list[Refusal]:
    """Step 2's refusal of every org past `cell.ORG_MAX_BYTES`, before any tenancy row."""
    return [
        Refusal(
            f"org of {len(org.encode())} bytes ({org[:64]}…)",
            f"found in {', '.join(sorted(sources))}: longer than {cell.ORG_MAX_BYTES} bytes; "
            "remove it first (docs/operations/cell-attach.md, the over-long org procedure)",
        )
        for org, sources in sorted(found.too_long.items())
    ]


def _refused(subject: str, reason: str) -> Refused:
    return Refused([Refusal(subject, reason)])


def _encoded_size(body: Any) -> int:
    return len(json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode())


def batches(orgs: Iterable[str], deployment_id: str) -> list[list[str]]:
    """`orgs` in `attach/place` batches: at most ATTACH_BATCH each, and an encoded body
    within the router's body bound."""
    from server.router.limits import ATTACH_BATCH, MAX_BODY_BYTES

    out: list[list[str]] = []
    batch: list[str] = []
    for org in sorted(orgs):
        if batch and (
            len(batch) >= ATTACH_BATCH
            or _encoded_size({"deployment_id": deployment_id, "orgs": [*batch, org]})
            > MAX_BODY_BYTES
        ):
            out.append(batch)
            batch = []
        batch.append(org)
    if batch:
        out.append(batch)
    return out


class Run:
    """One attach run: its sessions, what it read, and what it prints."""

    def __init__(
        self,
        ch: Any,
        args: argparse.Namespace,
        router: Router,
        log: Callable[[str], None],
    ) -> None:
        self.ch, self.args, self.router, self.log = ch, args, router, log
        self.name = args.cell
        self.wait = apg.Wait(args.wait)
        self.lock_conn: Any = None
        self._lock_cm: Any = None
        self.row: dict[str, Any] = {}
        self.mode = ""
        self.own_fp: tuple[dict[str, Any] | None, dict[str, Any] | None] = (None, None)
        self.keys_fp = ""  # the live processes' one root-secret fingerprint, last read
        self.begun_keys_fp = ""  # the one this run's `attach/begin` wrote
        self.report: dict[str, Any] = {"cell": self.name}

    # -- the lock session --------------------------------------------------------------

    async def open_lock(self) -> None:
        from server.core import pg

        self._lock_cm = pg.dedicated(statement_timeout_ms=_LOCK_STATEMENT_MS)
        self.lock_conn = await self._lock_cm.__aenter__()
        if refusals := await apg.direct_refusals(self.lock_conn, "attach"):
            raise Refused(refusals)
        await apg.take_lock(self.lock_conn, "attach")

    async def close_lock(self) -> None:
        if self._lock_cm is not None:
            await self._lock_cm.__aexit__(None, None, None)
            self._lock_cm = self.lock_conn = None

    async def confirm_lock(self) -> None:
        """The lock is held still, or taken again (refusing when another run holds it now),
        and the cell's state read again: a cleanup, a prepare rerun or another attach could
        have run in the gap."""
        try:
            if await apg.lock_held(self.lock_conn):
                return
        except Exception as e:  # noqa: BLE001 — a lost session is taken again below
            self.log(f"the lock session was lost ({type(e).__name__}); taking the lock again")
        else:
            self.log("the attach lock was lost; taking it again")
        try:
            await self.close_lock()
        except Exception:  # noqa: BLE001 — the old session is gone either way
            self._lock_cm = self.lock_conn = None
        await self.open_lock()
        state, first = await apg.read_mode(self.lock_conn)
        if first != (True, self.name) or (self.mode == "enforced" and state != "enforced"):
            raise _refused(
                "cell state",
                f"changed while the lock was lost (now {state}, first row {first}): rerun attach",
            )
        row = await self.router.call("")
        if (
            row.get("deployment_id") != self.row.get("deployment_id")
            or row.get("state") != "attaching"
        ):
            raise _refused("directory row", "changed while the lock was lost: rerun attach")

    # -- the refusals ------------------------------------------------------------------

    async def read_own_row(self) -> None:
        own = await self.lock_conn.fetchrow("SELECT current_fp, next_fp FROM loom.cell_self")
        if own is None:
            raise _refused("postgres loom.cell_self", "absent: run loom cell prepare first")
        self.own_fp = (cell.json_object(own["current_fp"]), cell.json_object(own["next_fp"]))

    def judge_beats(self, stopped_unset: str) -> Callable[[list[beats.Beat]], beats.Verdict]:
        def _judge(found: list[beats.Beat]) -> beats.Verdict:
            if stopped_unset == "ignore":
                found = [b for b in found if b.live or b.cell == self.name]
            return beats.judge(
                found, self.name, self.own_fp, "wait" if stopped_unset == "wait" else "refuse"
            )

        return _judge

    async def check_beats(self, stopped_unset: str) -> beats.Verdict:
        verdict = await beats.settle(
            lambda: beats.read_beats(self.ch), self.judge_beats(stopped_unset), self.wait
        )
        (self.keys_fp,) = verdict.keys_fps
        return verdict

    async def stores_refusals(self) -> list[Refusal]:
        from server.core import pg_profile

        out = []
        row = self.row
        if not row.get("is_first") or row.get("state") != "attaching":
            out.append(
                Refusal(
                    "directory row",
                    f"is {'the first' if row.get('is_first') else 'a later'} cell, "
                    f"{row.get('state')}: this attach is the first cell's, while attaching",
                )
            )
        if row.get("other"):
            out.append(
                Refusal(
                    "directory row",
                    f"another cell, {row['other']}, is registered: remove it first "
                    "(loom cell remove)",
                )
            )
        expected = str(row.get("deployment_id") or "")
        recorded_pg = await self.lock_conn.fetchval("SELECT deployment_id FROM loom.authority")
        if recorded_pg != expected:
            out.append(
                Refusal(
                    "postgres", f"records deployment id {recorded_pg!r}, the directory {expected!r}"
                )
            )
        recorded = await pg_profile.recorded_authority(self.ch)
        if recorded is None or str(recorded["deployment_id"]) != expected:
            out.append(Refusal("clickhouse", f"records another deployment id than {expected!r}"))
        state, first = await apg.read_mode(self.lock_conn)
        if first != (True, self.name):
            out.append(
                Refusal("postgres loom.cell_first", f"reads {first}, not (true, {self.name})")
            )
        if state not in ("recording", "enforced"):
            out.append(Refusal("postgres cell state", f"reads {state}"))
        answered = await self.router.health(str(row.get("host") or ""))
        if answered != expected:
            out.append(
                Refusal(
                    f"https://{row.get('host')}/v1/health/deployment",
                    f"answers deployment id {answered!r}, not {expected!r}",
                )
            )
        return out

    async def durability_refusals(self) -> list[Refusal]:
        from server.core import pg, pg_profile

        out = []
        verdict = pg_profile.durability_verdict("synchronous", *await pg_profile.standby_state())
        if verdict.refusal:
            out.append(Refusal("postgres durability", verdict.refusal))
        try:
            await pg.check_durable_message()
        except Exception as e:  # noqa: BLE001 — a role that may not write it refuses
            out.append(Refusal("postgres durability", f"pg_logical_emit_message refused: {e}"))
        if (now := await pg_profile.standby_now()) != "synchronous":
            out.append(
                Refusal("postgres standbys", f"commits are acknowledged as {now}, not synchronous")
            )
        return out

    async def ch_limit_refusals(self) -> list[Refusal]:
        """A profile or quota that caps the rows or bytes attach's user reads would end a
        scan early."""
        row = await self.ch.query_one(
            "SELECT getSetting('max_rows_to_read') AS r, getSetting('max_bytes_to_read') AS b"
        )
        out = []
        if row and (int(row["r"]) or int(row["b"])):
            out.append(
                Refusal("clickhouse user", "a settings profile caps the rows or bytes it reads")
            )
        quota = await self.ch.query_all(
            "SELECT quota_name FROM system.quota_usage "
            "WHERE max_read_rows > 0 OR max_read_bytes > 0"
        )
        out += [
            Refusal(f"clickhouse quota {q['quota_name']}", "caps the rows or bytes attach reads")
            for q in quota
        ]
        return out

    async def writer_refusals(self) -> tuple[list[Refusal], Any, Any]:
        from server.cli import cell_writers

        topology = await read_topology(self.ch)
        if refusals := topology.refusals():
            return refusals, topology, {}
        names = await cell_writers.host_names(self.ch, topology) if topology.multi else {}
        allowed = cell_writers.Allowed(
            users=frozenset(self.args.ch_allow_user),
            user_directories=frozenset(self.args.ch_allow_user_directory),
            objects=frozenset(self.args.ch_allow_object),
            replicas=frozenset(self.args.ch_allow_replica),
        )
        return await cell_writers.check_writers(self.ch, topology, names, allowed), topology, names

    async def catalog_refusals(self) -> list[Refusal]:
        """The Postgres checks that run again before the mode commit and in step 4."""
        holders = await apg.privilege_holders(self.lock_conn)
        self.report["own_roles"] = holders.own
        out = apg.privilege_refusals(holders, self.args.pg_allow_role)
        out += await apg.unmarked_refusals(self.lock_conn, await apg.session_passes())
        return out

    async def refusals(self) -> tuple[Any, Any, cell_scan.TableList]:
        """Every refusal before any write, named together."""
        out = await self.stores_refusals()
        out += await apg.encoding_refusals(self.lock_conn)
        out += await self.catalog_refusals()
        out += await self.durability_refusals()
        rows = await self.lock_conn.fetch(cell_triggers.ORG_TABLES)
        out += await apg.own_role_refusals(self.lock_conn, list(rows))
        out += await self.ch_limit_refusals()
        out += beats.prune_window_refusals()
        if out:
            raise Refused(out)
        state, _ = await apg.read_mode(self.lock_conn)
        # A stopped process with LOOM_CELL unset refuses while the cell can still be
        # backed out; on a resume in `enforced` step 4 waits for it to age out.
        await self.check_beats("refuse" if state == "recording" else "ignore")
        writers, topology, names = await self.writer_refusals()
        out += writers
        listed = await cell_scan.list_tables(self.ch, self.lock_conn, topology, names)
        out += listed.refusals
        if out:
            raise Refused(out)
        return topology, names, listed

    # -- begin -------------------------------------------------------------------------

    async def begin(self) -> None:
        from server.cli.cell_prepare import _delete_nonces, _new_nonces, _write_nonces
        from server.router.limits import ATTACH_BEGIN_MAX_BYTES

        current, nxt = self.own_fp
        nonces = _new_nonces()
        body = {
            "deployment_id": self.row["deployment_id"],
            "current_fp": current,
            "next_fp": nxt,
            "keys_fp": self.keys_fp,
            "ch_allow_users": sorted(self.args.ch_allow_user),
            "ch_allow_user_directories": sorted(self.args.ch_allow_user_directory),
            "ch_allow_objects": sorted(self.args.ch_allow_object),
            "ch_allow_replicas": sorted(self.args.ch_allow_replica),
            "pg_allow_roles": sorted(self.args.pg_allow_role),
            "nonces": nonces,
        }
        if (size := _encoded_size(body)) > ATTACH_BEGIN_MAX_BYTES:
            raise _refused(
                "attach/begin",
                f"its body is {size} bytes, past the router's {ATTACH_BEGIN_MAX_BYTES}: the "
                "fingerprints or the allow lists are too large",
            )
        try:
            await _write_nonces(self.ch, nonces)
            self.row = {**self.row, **await self.router.call("/attach/begin", body)}
            self.begun_keys_fp = self.keys_fp
        finally:
            await _delete_nonces(self.ch, nonces)

    # -- the scans ---------------------------------------------------------------------

    async def scan(self, topology: Any, listed: cell_scan.TableList) -> cell_scan.ScanResult:
        budget = cell_scan.Budget(self.args.scan_budget)
        current, nxt = self.own_fp
        return await cell_scan.scan(
            self.ch, listed, topology, budget, cell_scan.configured_orgs(current, nxt)
        )

    def _named(self, found: cell_scan.ScanResult) -> dict[str, Any]:
        return {
            "malformed": {k: v[:20] for k, v in sorted(found.malformed.items())},
            "too_long": {
                f"{o[:64]}… ({len(o.encode())} bytes)": sorted(s) for o, s in found.too_long.items()
            },
        }

    async def triggers(self) -> None:
        if refusals := await apg.ensure_triggers(self.lock_conn, self.wait, self.log):
            raise Refused(refusals)

    # -- step 3 ------------------------------------------------------------------------

    async def ready_for_commit(self) -> bool:
        """The cheap heartbeat refusals right before a mode-commit try: a refusal raises,
        a booting process skips the try."""
        verdict = self.judge_beats("refuse")(await beats.read_beats(self.ch))
        if verdict.refusals:
            raise Refused(verdict.refusals)
        return not verdict.booting

    async def commit_mode(self) -> None:
        from server.core import pg

        # From here on the mode may have risen: only the durable read can say.
        self.mode = "unknown"
        await self.router.call("/attach/enforced", {"deployment_id": self.row["deployment_id"]})
        try:
            async with pg.dedicated(statement_timeout_ms=_LOCK_STATEMENT_MS) as conn:
                await apg.commit_enforced(
                    conn, self.confirm_lock, self.ready_for_commit, self.wait, self.log
                )
        except Refused:
            raise
        except Exception as e:  # noqa: BLE001 — the durable read says whether it landed
            self.log(f"the mode commit's outcome is unknown ({type(e).__name__}: {e})")
            async with pg.dedicated(statement_timeout_ms=_LOCK_STATEMENT_MS) as conn:
                await apg.end_mode_transactions(conn, self.confirm_lock)

    async def durable(self) -> int:
        """The durable read of `enforced`, retried within `--wait`; then X, the snapshot's
        xmax, above the read's own xid."""
        from server.core import pg, pg_profile

        with self.wait.counting():
            while True:
                if await pg_profile.standby_now() != "synchronous":
                    raise _refused("postgres standbys", "not synchronous before the durable read")
                try:
                    read = await apg.durable_read()
                except Exception as e:
                    # A try cut off by its bound, or an outage: its outcome is unknown, and
                    # the next try reads again. Anything else is no reason to retry.
                    outage = pg.is_outage(e) or isinstance(e, pg.PostgresUnavailableError)
                    if not (isinstance(e, TimeoutError) or outage):
                        raise
                    self.log(f"the durable read failed ({type(e).__name__}); retrying")
                    if not await self.wait.pause():
                        raise _refused(
                            "postgres durable read", "did not commit within --wait"
                        ) from e
                    continue
                if await pg_profile.standby_now() != "synchronous":
                    raise _refused("postgres standbys", "not synchronous after the durable read")
                break
        if read.first != (True, self.name) or read.mode != "enforced":
            raise _refused(
                "postgres cell state",
                f"the durable read saw mode {read.mode!r} and first row {read.first}: the mode "
                "commit did not land; rerun attach",
            )
        self.mode = "enforced"
        xmax = await apg.snapshot_xmax(self.lock_conn)
        if xmax <= read.xid:
            raise _refused("postgres snapshot", f"its xmax {xmax} is not above {read.xid}")
        return xmax

    # -- step 4 ------------------------------------------------------------------------

    async def cheap_checks(self, stopped_unset: str) -> None:
        from server.core import pg_profile

        out, _, _ = await self.writer_refusals()
        out += [
            Refusal("postgres triggers", d.text)
            for d in await cell_triggers.missing(self.lock_conn)
        ]
        out += await self.catalog_refusals()
        if out:
            raise Refused(out)
        await self.check_beats(stopped_unset)
        if (now := await pg_profile.standby_now()) != "synchronous":
            raise _refused("postgres standbys", f"commits are acknowledged as {now}")

    async def final_checks(self, watch: beats.Watch) -> None:
        from server.core import instances

        floor = apg.Wait(max(self.wait.left(), instances.boot_allowance_s()))

        async def _read() -> list[beats.Beat]:
            found = await watch.read_once()
            if refusals := watch.refusals():
                raise Refused(refusals)
            # A cell-correct process whose cell row alone was swept.
            return [b for b in found if not (b.cell is None and b.instance_id in watch.cell_ids)]

        verdict = await beats.settle(
            _read, self.judge_beats("refuse"), floor, extend_s=instances.boot_allowance_s()
        )
        if verdict.keys_fps != {self.begun_keys_fp}:
            raise _refused(
                "processes",
                "the root-secret fingerprint changed during attach (a root-secret rotation): "
                "rerun attach",
            )

    async def step4(self, xmax: int, topology: Any, names: Any) -> None:
        from server.core import pg

        async with pg.dedicated(statement_timeout_ms=_LOCK_STATEMENT_MS) as conn:
            await apg.wait_transactions(conn, xmax, self.wait)
        await self.cheap_checks("wait")
        since = str((await self.ch.query_one("SELECT toString(now64(3)) AS t"))["t"])
        watch = beats.Watch(self.ch, since, self.name)
        task = asyncio.create_task(watch.run())
        try:
            listed = await cell_scan.list_tables(self.ch, self.lock_conn, topology, names)
            found = await self.scan(topology, listed)
            orgs = set(found.orgs)
            self.report["tenancy_rows_written"] = await apg.write_tenancy(orgs)
            placed = 0
            for batch in batches(orgs, self.row["deployment_id"]):
                await self.confirm_lock()
                answer = await self.router.call(
                    "/attach/place", {"deployment_id": self.row["deployment_id"], "orgs": batch}
                )
                placed += int(answer.get("placed") or 0)
            self.report.update(orgs_found=len(orgs), placed_now=placed, skipped=self._named(found))
            writers, _, _ = await self.writer_refusals()
            if writers:
                raise Refused(writers)
        finally:
            task.cancel()
        await self.final_checks(watch)
        row = await self.router.call("")
        current, nxt = self.own_fp
        if (row.get("current_fp"), row.get("next_fp")) != (current, nxt):
            raise _refused(
                "postgres loom.cell_self",
                "differs from the directory's fingerprints (a fingerprinted setting changed "
                "during attach?)",
            )
        state, _ = await apg.read_mode(self.lock_conn)
        if state != "enforced":
            raise _refused("postgres cell state", f"reads {state}, not enforced")

    async def serving(self) -> dict[str, Any]:
        from server.cli.cell_prepare import _delete_nonces, _new_nonces, _write_nonces

        nonces = _new_nonces()
        try:
            await _write_nonces(self.ch, nonces)
            return await self.router.call(
                "/attach/serving", {"deployment_id": self.row["deployment_id"], "nonces": nonces}
            )
        finally:
            await _delete_nonces(self.ch, nonces)

    # -- the run -----------------------------------------------------------------------

    async def done(self) -> bool:
        """On a `serving` row: True once the cell reads `enforced`; Refused otherwise."""
        from server.core import pg

        async with pg.dedicated(statement_timeout_ms=_LOCK_STATEMENT_MS) as conn:
            state, _ = await apg.read_mode(conn)
        if state != "enforced":
            raise _refused(
                "postgres cell state",
                f"the directory says serving and the cell reads {state}: the directory and the "
                "cell's Postgres disagree, as after a restore of one of them; attach lowers "
                "and repairs neither",
            )
        return True

    async def run(self) -> int:
        self.row = await self.router.call("")
        if self.row.get("state") == "serving":
            await self.done()
            self.report.update(state="serving", done=True)
            return 0
        try:
            await self.open_lock()
            await self.read_own_row()
            topology, names, listed = await self.refusals()
            if self.args.dry_run:
                return await self.dry_run(topology, listed)
            await self.begin()
            state, _ = await apg.read_mode(self.lock_conn)
            self.mode = state
            await self.triggers()
            if state == "recording":
                found = await self.scan(topology, listed)
                if found.too_long:
                    raise Refused(_too_long(found))
                written = await apg.write_tenancy(found.orgs)
                self.log(f"step 2: {len(found.orgs)} orgs found, {written} tenancy rows written")
                await self.triggers()
                await self.precommit(found)
                await self.commit_mode()
            else:
                await self.router.call(
                    "/attach/enforced", {"deployment_id": self.row["deployment_id"]}
                )
            xmax = await self.durable()
            await self.check_beats("ignore")
            await self.step4(xmax, topology, names)
            answer = await self.serving()
            self.report.update(state=answer.get("state"), placed_orgs=answer.get("placed_orgs"))
            return 0
        finally:
            await self.close_lock()

    async def precommit(self, found: cell_scan.ScanResult) -> None:
        from server.core import pg_profile

        out = await self.catalog_refusals()
        if (now := await pg_profile.standby_now()) != "synchronous":
            out.append(Refusal("postgres standbys", f"commits are acknowledged as {now}"))
        if found.elapsed_s > self.args.scan_budget / 2:
            out.append(
                Refusal(
                    "--scan-budget",
                    f"step 2 took {found.elapsed_s:.0f} s, more than half of "
                    f"{self.args.scan_budget} s, and step 4 reads the same tables again: "
                    "raise it",
                )
            )
        if out:
            raise Refused(out)
        verdict = await self.check_beats("refuse")
        if refusals := beats.final_refusals(verdict):
            raise Refused(refusals)

    async def dry_run(self, topology: Any, listed: cell_scan.TableList) -> int:
        found = await self.scan(topology, listed)
        per_source: dict[str, list[str]] = {}
        for org, sources in found.orgs.items():
            for s in sources:
                per_source.setdefault(s, []).append(org)
        self.report.update(
            dry_run=True,
            scan_seconds=round(found.elapsed_s, 1),
            half_budget_ok=found.elapsed_s <= self.args.scan_budget / 2,
            orgs=len(found.orgs),
            per_source={
                s: {"count": len(o), "sample": sorted(o)[:5]} for s, o in sorted(per_source.items())
            },
            skipped=self._named(found),
        )
        return 0


async def attach(
    ch: Any,
    args: argparse.Namespace,
    url: httpx.URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    def log(text: str) -> None:
        print(f"loom cell attach: {text}", file=sys.stderr)

    run = Run(ch, args, Router(url, args.cell, transport), log)
    started = time.monotonic()
    try:
        code = await run.run()
    except Refused as e:
        for r in e.refusals:
            log(f"refusing: {r}")
        log(mode_line(run.mode))
        return 1
    except Exception as e:  # noqa: BLE001 — a store that fails names itself; a rerun resumes
        log(f"stopped ({type(e).__name__}: {e}); run it again")
        log(mode_line(run.mode))
        return 1
    run.report["seconds"] = round(time.monotonic() - started, 1)
    print(json.dumps(run.report, indent=2, sort_keys=True, default=str))
    return code


def mode_line(mode: str) -> str:
    """What a stopped run left the cell's mode at, by what it last knew."""
    if mode == "enforced":
        return "the cell stays enforced and attaching: rerun attach to resume"
    if mode == "unknown":
        return (
            "the mode commit's outcome is unknown: rerun attach, which resumes by the mode it reads"
        )
    if mode == "recording":
        return "the cell stays recording"
    return "the cell's state is unchanged"


def cmd_attach(args: argparse.Namespace) -> int:
    if problem := _args_problem(args):
        print(f"loom cell attach: {problem}", file=sys.stderr)
        return 2
    url = cell_cli.operator_router(args)
    if isinstance(url, str):
        print(f"loom cell attach: {url}", file=sys.stderr)
        return 2
    from server.cli._ch import run_connected
    from server.core.ch import statement_timeout

    async def _run(ch: Any) -> int:
        return await attach(ch, args, url)

    # Every ClickHouse statement, the connect included, may run as long as a scan may.
    with statement_timeout(args.scan_budget):
        return run_connected(_run, org_free=writes("attaches the cell; places the orgs it holds"))


def add_attach_parser(csub: Any) -> None:
    p = csub.add_parser("attach", help="Attach the prepared first cell: place its orgs, serve.")
    p.add_argument("--cell", required=True, help="This cell's name (LOOM_CELL)")
    p.add_argument("--wait", type=int, default=_WAIT_S, help="Seconds the waits share")
    p.add_argument(
        "--scan-budget", type=int, default=_SCAN_BUDGET_S, help="Seconds one full org scan may take"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the refusals and the scan and print what would be placed; write nothing "
        "(the router's own begin refusals do not run)",
    )
    for flag, what in (
        ("--ch-allow-user", "A ClickHouse user that may write the configured databases"),
        ("--ch-allow-user-directory", "A ClickHouse user directory the check cannot read"),
        ("--ch-allow-object", "A ClickHouse object (db.name) that writes the databases"),
        ("--ch-allow-replica", "A replica outside the cluster (db.table:replica)"),
        ("--pg-allow-role", "A Postgres role that may make an org held (the servers' role)"),
    ):
        p.add_argument(flag, action="append", default=[], help=what)
    p.add_argument("--router", help="The router's URL (default: LOOM_CELL_ROUTER_URL)")
    p.set_defaults(func=cmd_attach)
