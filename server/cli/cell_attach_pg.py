"""The first attach's reads and writes on the cell's Postgres (docs/architecture/cells.md,
"Attaching a cell"): the session checks, the one-run lock, the catalog checks attach and
`loom doctor` share (the roles that can make an org held or lower the mode, the settings
and subscriptions that pass an unchecked writer), the trigger pass, the mode commit, its
durable read and the wait for the transactions that could have read `recording`.

Every check returns its refusals as `cell_ch.Refusal` (subject, reason), as the org scan
and the ClickHouse writer check do, so attach and doctor print the same names."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from server.cli.cell_ch import Refusal, Refused
from server.core import cell, cell_triggers, pg
from server.core.cell import ATTACH_LOCK_KEY

_sleep = asyncio.sleep  # the seam the wait tests replace

# The pause between polls of a wait.
POLL_S = 1.0  # not a setting: a poll interval
# The migration lock's wait a trigger try (a boot holds it for its migrations), in ms; a
# try never waits longer than --wait has left.
_MIGRATION_LOCK_MS = 10_000  # not a setting: one try's bound
# The mode commit's lock wait a try, in ms, and its statement bound's margin above it.
_MODE_LOCK_MS = 2_000  # not a setting: one try's bound
_MODE_STATEMENT_MARGIN_MS = 3_000  # not a setting: a margin over the lock wait
# The server ends the lock session soon after its client is gone, releasing the lock.
_KEEPALIVE = {"tcp_keepalives_idle": "30", "tcp_keepalives_interval": "10"}


class Wait:
    """`--wait`: the wall-clock seconds attach's bounded waits share. Each wait runs inside
    `counting()`, which charges its elapsed time, lock tries included; time outside every
    wait (a scan) is never charged."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = float(seconds)
        self.spent = 0.0
        self._clock = clock
        self._since: float | None = None

    def used(self) -> float:
        """The seconds charged so far, the wait running now included."""
        running = 0.0 if self._since is None else self._clock() - self._since
        return self.spent + running

    def left(self) -> float:
        return max(0.0, self.seconds - self.used())

    @contextlib.contextmanager
    def counting(self) -> Iterator[None]:
        """Charge the block's wall time (a block inside another is charged once)."""
        if self._since is not None:
            yield
            return
        self._since = self._clock()
        try:
            yield
        finally:
            self.spent += self._clock() - self._since
            self._since = None

    def bound_ms(self, most_ms: int) -> int:
        """A lock timeout for one try: `most_ms`, or what is left when that is less."""
        return min(most_ms, int(self.left() * 1000))

    async def pause(self, seconds: float = POLL_S) -> bool:
        """Wait `seconds` (at most what is left); False once none is left."""
        step = min(seconds, self.left())
        if step <= 0:
            return False
        with self.counting():
            await _sleep(step)
        return True


def _split(key: int) -> str:
    """The `pg_locks` columns that hold a bigint advisory key, in this database."""
    return (
        f"l.locktype = 'advisory' AND l.classid = ({key} >> 32)::oid "
        f"AND l.objid = ({key} & 4294967295)::oid AND l.objsubid = 1 "
        "AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())"
    )


_LOCK_HOLDER = f"""
SELECT l.pid, a.application_name, a.backend_start, a.client_addr::text AS client
FROM pg_locks l LEFT JOIN pg_stat_activity a ON a.pid = l.pid
WHERE {_split(ATTACH_LOCK_KEY)} AND l.granted
"""

_IDENTITY = (
    "SELECT pg_backend_pid() AS pid, backend_start FROM pg_stat_activity "
    "WHERE pid = pg_backend_pid()"
)


async def direct_refusals(conn: Any, what: str) -> list[Refusal]:
    """Whether `conn` is one direct, marked session: the same backend across two
    transactions, and the tenancy marker kept through `RESET ALL` (a pooler that strips
    startup parameters drops it). Behind a transaction pooler a session-level lock would
    outlive the CLI on the pooler's server connection, and attach's tenancy rows would be
    refused as an unchecked writer's."""
    async with conn.transaction():
        first = await conn.fetchrow(_IDENTITY)
    async with conn.transaction():
        second = await conn.fetchrow(_IDENTITY)
    out = []
    if first is None or second is None or tuple(first) != tuple(second):
        out.append(
            Refusal(
                "postgres session",
                f"{what}'s session moved between backends: point LOOM_PG_DSN straight at "
                "Postgres, never through a pooler",
            )
        )
    await conn.execute("RESET ALL")  # a startup value survives it, a SET does not
    if await conn.fetchval("SELECT current_setting($1, true)", pg.TENANCY_MARKER) != "on":
        out.append(
            Refusal(
                "postgres session",
                f"{what}'s session loses {pg.TENANCY_MARKER}: run it with LOOM_CELL set, "
                "LOOM_PG_DSN straight at Postgres",
            )
        )
    return out


_HELD = f"""
SELECT EXISTS (SELECT 1 FROM pg_locks l
               WHERE {_split(ATTACH_LOCK_KEY)} AND l.granted AND l.pid = pg_backend_pid())
"""


async def lock_holder(conn: Any) -> str:
    """Who holds the attach lock ('' when no one), never a query's text."""
    rows = await conn.fetch(_LOCK_HOLDER)
    return "; ".join(
        f"pid {r['pid']} ({r['application_name'] or 'no name'}, from {r['client'] or 'local'}, "
        f"since {r['backend_start']})"
        for r in rows
    )


async def take_lock(conn: Any, what: str) -> None:
    """Take the attach lock on `conn` for the session's life, confirmed held by this
    backend; Refused naming its holder when another run holds it."""
    for name, value in _KEEPALIVE.items():
        await conn.execute(f"SET {name} = {value}")
    if not await conn.fetchval("SELECT pg_try_advisory_lock($1)", ATTACH_LOCK_KEY):
        holder = await lock_holder(conn) or "a session that just ended"
        raise Refused(
            [
                Refusal(
                    "attach lock",
                    f"held by {holder}: an attach or a prepare is running; {what} "
                    "runs one at a time",
                )
            ]
        )
    if not await lock_held(conn):
        raise Refused(
            [
                Refusal(
                    "attach lock",
                    "taken but not held by this session's backend: a pooler between "
                    "this command and Postgres; point LOOM_PG_DSN straight at it",
                )
            ]
        )


async def lock_held(conn: Any) -> bool:
    """Whether `conn`'s backend holds the attach lock."""
    return bool(await conn.fetchval(_HELD))


async def encoding_refusals(conn: Any) -> list[Refusal]:
    """A cell's database is UTF8: its `text` then holds no NUL or invalid UTF-8, so every
    tenancy row within the length limit is an org a request can name."""
    encoding = await conn.fetchval(
        "SELECT pg_encoding_to_char(encoding) FROM pg_database WHERE datname = current_database()"
    )
    if encoding == "UTF8":
        return []
    return [
        Refusal(
            "postgres database",
            f"its encoding is {encoding}, not UTF8: a cell's database is UTF8 "
            "(recreate it with ENCODING 'UTF8' and restore into it)",
        )
    ]


# -- who can make an org held, or lower the mode --------------------------------------

_SYSTEM_SCHEMA = (
    "n.nspname NOT IN ('pg_catalog', 'information_schema') "
    "AND n.nspname NOT LIKE 'pg\\_toast%' AND n.nspname NOT LIKE 'pg\\_temp\\_%'"
)

# Every right a login role may hold that lets it make an org held, lower the mode, or run
# its own SQL as the servers on their marked sessions: each row a role and why.
_RIGHTS = f"""
WITH owned AS (
    SELECT c.relowner AS owner, 'owns loom.' || c.relname AS why
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'loom' AND c.relkind IN ('r', 'p')
    UNION ALL
    SELECT p.proowner, 'owns loom.' || p.proname || '()' FROM pg_proc p
    WHERE p.oid IN (to_regprocedure('loom.tenancy_refuse_unheld()'),
                    to_regprocedure('loom.tenancy_refuse_unmarked()'),
                    to_regprocedure('loom.cell_first_immutable()'))
    UNION ALL
    SELECT d.datdba, 'owns the database' FROM pg_database d
    WHERE d.datname = current_database()
    UNION ALL
    SELECT n.nspowner, 'owns schema ' || n.nspname FROM pg_namespace n WHERE {_SYSTEM_SCHEMA}
),
login AS (SELECT oid, rolname FROM pg_roles WHERE rolcanlogin)
SELECT r.rolname AS role, w.why FROM login r, LATERAL (
    SELECT 'may insert into loom.tenancy'
    WHERE has_any_column_privilege(r.oid, 'loom.tenancy', 'INSERT')
    UNION ALL SELECT 'may update loom.tenancy (org)'
    WHERE has_column_privilege(r.oid, 'loom.tenancy', 'org', 'UPDATE')
    UNION ALL SELECT 'may update loom.tenancy_mode (mode)'
    WHERE has_column_privilege(r.oid, 'loom.tenancy_mode', 'mode', 'UPDATE')
    UNION ALL SELECT 'may insert into loom.tenancy_mode'
    WHERE has_any_column_privilege(r.oid, 'loom.tenancy_mode', 'INSERT')
    UNION ALL SELECT 'holds TRIGGER on loom.' || c.relname
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'loom' AND c.relkind IN ('r', 'p')
      AND has_table_privilege(r.oid, c.oid, 'TRIGGER')
    UNION ALL SELECT 'may create in the database'
    WHERE has_database_privilege(r.oid, current_database(), 'CREATE')
    UNION ALL SELECT 'may create in schema ' || n.nspname FROM pg_namespace n
    WHERE {_SYSTEM_SCHEMA} AND has_schema_privilege(r.oid, n.oid, 'CREATE')
    UNION ALL SELECT 'may set session_replication_role'
    WHERE has_parameter_privilege(r.oid, 'session_replication_role', 'SET')
       OR has_parameter_privilege(r.oid, 'session_replication_role', 'ALTER SYSTEM')
    UNION ALL SELECT 'may ALTER SYSTEM SET {pg.TENANCY_MARKER}'
    WHERE has_parameter_privilege(r.oid, '{pg.TENANCY_MARKER}', 'ALTER SYSTEM')
    UNION ALL SELECT 'is a member of ' || g.rolname FROM pg_roles g
    WHERE g.rolname IN ('pg_write_server_files', 'pg_execute_server_program')
      AND pg_has_role(r.oid, g.oid, 'MEMBER')
    UNION ALL SELECT o.why || CASE WHEN o.owner = r.oid THEN ''
                                   ELSE ' (as a member of ' || pg_get_userbyid(o.owner) || ')'
                              END
    FROM owned o WHERE pg_has_role(r.oid, o.owner, 'MEMBER')
) w(why)
ORDER BY 1, 2
"""

# The roles attach counts as its own: superusers, and a login role that itself owns
# loom.tenancy, loom.tenancy_mode or a guarded table (it can drop or disable the
# triggers, or lower the mode, whatever else it holds).
_OWN_ROLES = """
SELECT rolname FROM pg_roles WHERE rolcanlogin AND (rolsuper OR oid IN (
    SELECT c.relowner FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'loom' AND c.relkind IN ('r', 'p')
      AND (c.relname IN ('tenancy', 'tenancy_mode') OR c.relname = ANY($1::text[]))
)) OR rolname = current_user
ORDER BY 1
"""


@dataclass(frozen=True)
class Holders:
    """The login roles holding a right `_RIGHTS` counts: the ones attach counts as its own
    (`own`, named in its output) and every other with its reasons."""

    own: list[str]
    others: dict[str, list[str]]


async def privilege_holders(conn: Any) -> Holders:
    guarded = [t for t in await cell_triggers.planned_tables(conn) if t != "tenancy"]
    own = [str(r["rolname"]) for r in await conn.fetch(_OWN_ROLES, guarded)]
    others: dict[str, list[str]] = {}
    for r in await conn.fetch(_RIGHTS):
        if r["role"] not in own:
            others.setdefault(str(r["role"]), []).append(str(r["why"]))
    return Holders(own, others)


def privilege_refusals(holders: Holders, allowed: Iterable[str]) -> list[Refusal]:
    allow = set(allowed)
    return [
        Refusal(
            f"postgres role {role}",
            f"{'; '.join(why)}: it can make an org held or lower the mode outside attach; "
            "revoke it, or name the role with --pg-allow-role",
        )
        for role, why in sorted(holders.others.items())
        if role not in allow
    ]


# -- what passes an unchecked writer through the triggers -----------------------------

_ROLE_SETTINGS = f"""
SELECT coalesce(r.rolname, 'every role') AS role, coalesce(d.datname, 'every database') AS db,
       cfg
FROM pg_db_role_setting s
LEFT JOIN pg_roles r ON r.oid = s.setrole
LEFT JOIN pg_database d ON d.oid = s.setdatabase,
LATERAL unnest(s.setconfig) cfg
WHERE s.setdatabase IN (0, (SELECT oid FROM pg_database WHERE datname = current_database()))
  AND lower(split_part(cfg, '=', 1)) IN ('{pg.TENANCY_MARKER}', 'session_replication_role')
ORDER BY 1, 2, 3
"""

_SUBSCRIPTIONS = """
SELECT subname, subenabled FROM pg_subscription
WHERE subdbid = (SELECT oid FROM pg_database WHERE datname = current_database())
ORDER BY 1
"""


async def unmarked_refusals(conn: Any, session_passes: bool) -> list[Refusal]:
    """What lets a session that sends no marker pass the triggers: the session read
    (`session_passes`, doctor's unmarked session), an `ALTER ROLE … SET` or
    `ALTER DATABASE … SET` of the marker or `session_replication_role` for this database or
    every one, and any logical-replication subscription into this database (its apply
    worker runs as `replica`, firing no origin trigger)."""
    out = []
    if session_passes:
        out.append(
            Refusal(
                "postgres unmarked session",
                "a session that sends no marker is marked, or runs as replica (a role, "
                "database, ALTER SYSTEM or options setting): every unchecked writer passes",
            )
        )
    for r in await conn.fetch(_ROLE_SETTINGS):
        out.append(
            Refusal(
                f"postgres setting for {r['role']} in {r['db']}",
                f"{r['cfg']}: an unchecked writer under it passes the triggers; reset it",
            )
        )
    for r in await conn.fetch(_SUBSCRIPTIONS):
        state = "enabled" if r["subenabled"] else "disabled"
        out.append(
            Refusal(
                f"postgres subscription {r['subname']}",
                f"{state}: its apply worker fires no origin trigger; drop it",
            )
        )
    return out


async def session_passes() -> bool:
    """Whether a session that sends no marker passes the triggers all the same: marked,
    or in the replica role that fires no origin trigger."""
    async with pg.dedicated(statement_timeout_ms=5000, marked=False) as conn:
        value = await conn.fetchval("SELECT current_setting($1, true)", pg.TENANCY_MARKER)
        role = await conn.fetchval("SELECT current_setting('session_replication_role')")
    return bool(value == "on" or role == "replica")


# -- attach's own role ------------------------------------------------------------------

_OWN_GRANTS = """
SELECT need, held FROM (VALUES
  ('membership in pg_read_all_stats', pg_has_role(current_user, 'pg_read_all_stats', 'MEMBER')),
  ('membership in pg_signal_backend', pg_has_role(current_user, 'pg_signal_backend', 'MEMBER')),
  ('USAGE on schema loom', has_schema_privilege('loom', 'USAGE')),
  ('SELECT on loom.cell_first', has_table_privilege('loom.cell_first', 'SELECT')),
  ('SELECT on loom.tenancy_mode', has_table_privilege('loom.tenancy_mode', 'SELECT')),
  ('UPDATE on loom.tenancy_mode', has_table_privilege('loom.tenancy_mode', 'UPDATE')),
  ('SELECT on loom.cell_self', has_table_privilege('loom.cell_self', 'SELECT')),
  ('SELECT on loom.authority', has_table_privilege('loom.authority', 'SELECT')),
  ('INSERT on loom.tenancy', has_table_privilege('loom.tenancy', 'INSERT')),
  ('SELECT, INSERT, UPDATE and DELETE on loom.cell_probe',
   has_table_privilege('loom.cell_probe', 'SELECT, INSERT, UPDATE, DELETE')),
  ('TEMPORARY on the database', has_database_privilege(current_database(), 'TEMPORARY'))
) v(need, held)
WHERE NOT held
"""


async def own_role_refusals(conn: Any, tables: list[Any]) -> list[Refusal]:
    """What attach's own role lacks, `tables` being the rows `cell_triggers.ORG_TABLES`
    read: its grants, `SELECT` on every org table, and membership in the owner role of every
    table it gives triggers (or the trigger pass could not alter it); and a server that
    allows prepared transactions."""
    role = await conn.fetchval("SELECT current_user")
    lacks = [str(r["need"]) for r in await conn.fetch(_OWN_GRANTS)]
    for r in tables:
        if not await conn.fetchval(
            "SELECT has_table_privilege($1, 'SELECT')", f"loom.{r['relname']}"
        ):
            lacks.append(f"SELECT on loom.{r['relname']}")
    planned = set(cell_triggers.planned_names(tables))
    for r in tables:
        if r["relname"] in planned and not r["owned"]:
            lacks.append(f"membership in the owner role of loom.{r['relname']}")
    out = [Refusal(f"postgres role {role}", f"attach's role lacks {n}") for n in lacks]
    out += await prepared_refusals(conn)
    return out


async def prepared_refusals(conn: Any) -> list[Refusal]:
    setting = await conn.fetchval("SELECT current_setting('max_prepared_transactions')::int")
    if setting:
        return [
            Refusal(
                "postgres max_prepared_transactions",
                f"is {setting}: a prepared transaction outlives every session, so step 4 "
                "could not wait for it; set it to 0",
            )
        ]
    return []


# -- the triggers -----------------------------------------------------------------------

_MIGRATION_LOCK = "SELECT pg_advisory_lock($1)"
_MIGRATION_UNLOCK = "SELECT pg_advisory_unlock($1)"


async def ensure_triggers(conn: Any, wait: Wait, log: Callable[[str], None]) -> list[Refusal]:
    """Give every planned table its triggers (`cell_triggers.ensure_table` with
    `recording_first`), one try at a time under the migration lock, retrying a table whose
    lock is busy a second apart within `wait`; one autovacuum holding it is asked to yield
    by one try just past `deadlock_timeout`. A table this role cannot alter refuses at once.
    Then `missing`: any defect refuses."""
    deadlock_ms = int(
        await conn.fetchval(
            "SELECT setting::int * CASE unit WHEN 's' THEN 1000 ELSE 1 END "
            "FROM pg_settings WHERE name = 'deadlock_timeout'"
        )
    )
    with wait.counting():
        for table in await cell_triggers.planned_tables(conn):
            if refusal := await _ensure_table(conn, table, deadlock_ms, wait, log):
                return [refusal]
    return [Refusal("postgres triggers", d.text) for d in await cell_triggers.missing(conn)]


async def _ensure_table(
    conn: Any, table: str, deadlock_ms: int, wait: Wait, log: Callable[[str], None]
) -> Refusal | None:
    yielded = False
    while True:
        lock_ms = 100
        holders = await cell_triggers.holders(conn, table)
        if not yielded and any(h["backend_type"] == "autovacuum worker" for h in holders):
            lock_ms, yielded = deadlock_ms + 500, True
        lock_ms = wait.bound_ms(lock_ms)
        result = "busy" if lock_ms <= 0 else await _ensure_one(conn, table, lock_ms, wait)
        if result == "applied":
            return None
        if result == "denied":
            return Refusal(
                f"postgres loom.{table}",
                "attach's role cannot alter it: make the role a member of its owner role",
            )
        if not await wait.pause():
            return _busy_refusal(table, await cell_triggers.holders(conn, table))
        log(f"loom.{table}: its lock is busy; retrying")


async def _ensure_one(conn: Any, table: str, lock_ms: int, wait: Wait) -> str:
    """One try under the migration lock, whose own wait is bounded by what `wait` has left."""
    from server.core.schema.pg_migrate import _LOCK_KEY

    migration_ms = wait.bound_ms(_MIGRATION_LOCK_MS)
    if migration_ms <= 0:
        return "busy"
    await conn.execute(f"SET lock_timeout = '{migration_ms}ms'")
    try:
        await conn.execute(_MIGRATION_LOCK, _LOCK_KEY)
    except Exception as e:
        if getattr(e, "sqlstate", None) == "55P03":
            return "busy"  # a boot migrating: the next try
        raise
    finally:
        await conn.execute("RESET lock_timeout")
    try:
        return await cell_triggers.ensure_table(
            conn, table, recording_first=True, lock_timeout=f"{lock_ms}ms", attempts=1
        )
    finally:
        await conn.execute(_MIGRATION_UNLOCK, _LOCK_KEY)


def _busy_refusal(table: str, holders: list[Any]) -> Refusal:
    named = "; ".join(
        f"pid {h['pid']} ({'holds' if h['granted'] else 'waiting'}, {h['backend_type']}, "
        f"{h['application_name'] or 'no name'}, transaction since {h['xact_start']}"
        + (", an anti-wraparound vacuum, which never yields" if h["wraparound"] else "")
        + ")"
        for h in holders
    )
    return Refusal(
        f"postgres loom.{table}",
        f"its triggers' lock was not taken within --wait; held by {named or 'no one now'}",
    )


# -- step 3: the mode commit ------------------------------------------------------------

# The mode transaction in flight: the backend holding or waiting for the ExclusiveLock on
# loom.tenancy_mode in this database, which only attach takes.
_MODE_HOLDERS = """
SELECT l.pid, a.application_name, a.xact_start, r.rolsuper, l.granted
FROM pg_locks l
JOIN pg_stat_activity a ON a.pid = l.pid
LEFT JOIN pg_roles r ON r.oid = a.usesysid
WHERE l.locktype = 'relation' AND l.relation = 'loom.tenancy_mode'::regclass
  AND l.mode = 'ExclusiveLock'
  AND l.database = (SELECT oid FROM pg_database WHERE datname = current_database())
  AND l.pid <> pg_backend_pid()
"""

_TIMED_OUT = frozenset({"55P03", "57014"})


async def commit_enforced(
    conn: Any,
    confirm_lock: Callable[[], Any],
    ready: Callable[[], Any],
    wait: Wait,
    log: Callable[[str], None],
) -> None:
    """Raise the mode to `enforced` on `conn` (a session of its own, never the lock
    session): an EXCLUSIVE lock on loom.tenancy_mode a try, `_MODE_LOCK_MS` each (less
    when `wait` has less left), within `wait`; `confirm_lock` before each try and each
    termination (it refuses when another run holds the attach lock), and `ready` before
    each try (False skips the try while a process boots). After a failed try the in-flight
    mode transaction, found by its lock, is ended. Raises Refused when `wait` runs out;
    whether the commit landed is the durable read's to say."""
    with wait.counting():
        while True:
            await confirm_lock()
            lock_ms = wait.bound_ms(_MODE_LOCK_MS)
            if lock_ms > 0 and await ready():
                if await _mode_try(conn, lock_ms):
                    return
                log("the mode's lock is busy; retrying")
                await end_mode_transactions(conn, confirm_lock)
            if not await wait.pause():
                raise Refused(
                    [
                        Refusal(
                            "postgres loom.tenancy_mode",
                            "its lock was not taken within --wait: a long transaction reads "
                            "it FOR SHARE; rerun attach",
                        )
                    ]
                )


async def _mode_try(conn: Any, lock_ms: int) -> bool:
    """One try of the mode commit, its lock wait `lock_ms`; False when it timed out."""
    try:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL lock_timeout = '{lock_ms}ms'")
            await conn.execute(
                f"SET LOCAL statement_timeout = '{lock_ms + _MODE_STATEMENT_MARGIN_MS}ms'"
            )
            await conn.execute("LOCK TABLE loom.tenancy_mode IN EXCLUSIVE MODE")
            status = await conn.execute(
                "UPDATE loom.tenancy_mode SET mode = 'enforced', since = now()"
            )
            if status != "UPDATE 1":
                raise Refused(
                    [Refusal("postgres loom.tenancy_mode", f"the mode update read {status}")]
                )
        return True
    except Refused:
        raise
    except Exception as e:
        if getattr(e, "sqlstate", None) not in _TIMED_OUT:
            raise
        return False


async def end_mode_transactions(conn: Any, confirm_lock: Callable[[], Any]) -> None:
    """End an earlier run's mode transaction still in flight and wait until its lock
    clears; Refused when its role is a superuser and attach's is not."""
    for _ in range(30):
        rows = await conn.fetch(_MODE_HOLDERS)
        if not rows:
            return
        superuser = bool(
            await conn.fetchval("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
        )
        for r in rows:
            if r["rolsuper"] and not superuser:
                raise Refused(
                    [
                        Refusal(
                            f"postgres backend {r['pid']}",
                            f"an earlier mode transaction ({r['application_name'] or 'no name'}, "
                            f"since {r['xact_start']}) runs as a superuser: a superuser must "
                            "end it (pg_terminate_backend)",
                        )
                    ]
                )
            await confirm_lock()
            await conn.fetchval("SELECT pg_terminate_backend($1)", r["pid"])
        await _sleep(0.1)


# -- the durable read -------------------------------------------------------------------

_DURABLE_TRY_S = 10.0  # not a setting: one try's bound; a cancelled one is retried


@dataclass(frozen=True)
class Durable:
    """What the durable read saw: the mode, the `first` row, and the xid its logical
    message took."""

    mode: str
    first: tuple[bool, str] | None
    xid: int


async def durable_read() -> Durable:
    """The mode and the `first` row read in a transaction that emits a transactional
    logical message, whose commit waits for the synchronous standbys: once it returns, the
    mode read is on them. Refused when the message took no xid."""
    async with asyncio.timeout(_DURABLE_TRY_S), pg.durable_transaction() as conn:
        await conn.execute("SELECT pg_logical_emit_message(true, 'loom.attach', '')")
        xid = await conn.fetchval("SELECT pg_current_xact_id_if_assigned()::text")
        mode = await conn.fetchval("SELECT mode FROM loom.tenancy_mode")
        first = await conn.fetchrow("SELECT is_first, cell FROM loom.cell_first")
    if xid is None:
        raise Refused([Refusal("postgres durable read", "its logical message took no xid")])
    return Durable(
        mode=str(mode or ""),
        first=None if first is None else (bool(first["is_first"]), str(first["cell"])),
        xid=int(xid),
    )


async def snapshot_xmax(conn: Any) -> int:
    return int(await conn.fetchval("SELECT pg_snapshot_xmax(pg_current_snapshot())::text"))


# -- step 4: the transactions that could have read `recording` --------------------------

# Every backend but this statement's own: of the cell's database, one whose xid or (but
# an autovacuum's) snapshot is older than X; of another database, one whose xid is older
# (it holds every snapshot's horizon). Both compared through age() against one reference,
# which is exactly `< X`, wraparound-safe, and takes no xid.
_OPEN = """
WITH x AS (SELECT age($1::text::xid8::xid) AS a)
SELECT s.pid, s.backend_type, s.application_name, s.xact_start::text AS xact_start,
       s.datid = (SELECT oid FROM pg_database WHERE datname = current_database()) AS ours,
       coalesce(s.backend_xid IS NOT NULL AND age(s.backend_xid) > x.a, false) AS xid_old,
       coalesce(s.backend_xmin IS NOT NULL AND age(s.backend_xmin) > x.a
                AND s.backend_type <> 'autovacuum worker', false) AS xmin_old,
       coalesce(s.query = '<insufficient privilege>', false) AS hidden
FROM pg_stat_activity s, x
WHERE s.pid <> pg_backend_pid()
"""


def _named(r: Any) -> str:
    return (
        f"pid {r['pid']} ({r['backend_type']}, {r['application_name'] or 'no name'}, "
        f"transaction since {r['xact_start']})"
    )


async def wait_transactions(conn: Any, xmax: int, wait: Wait) -> None:
    """Wait until no backend of the cell's database holds an xid or a snapshot older than
    `xmax`; Refused at once on a prepared transaction or an unreadable row, and past
    `wait` naming each backend waited on and each other database's horizon holder."""
    with wait.counting():
        await _wait_transactions(conn, xmax, wait)


async def _wait_transactions(conn: Any, xmax: int, wait: Wait) -> None:
    while True:
        if refusals := await prepared_refusals(conn):
            raise Refused(refusals)
        if await conn.fetchval("SELECT count(*) FROM pg_prepared_xacts"):
            raise Refused(
                [Refusal("postgres prepared transaction", "one is open: commit or roll it back")]
            )
        rows = await conn.fetch(_OPEN, str(xmax))
        hidden = [r for r in rows if r["ours"] and r["hidden"]]
        if hidden:
            raise Refused(
                [
                    Refusal(
                        f"postgres backend {r['pid']}",
                        "its transaction state is hidden from attach's role: grant it "
                        "pg_read_all_stats",
                    )
                    for r in hidden
                ]
            )
        waited = [r for r in rows if r["ours"] and (r["xid_old"] or r["xmin_old"])]
        if not waited:
            return
        if not await wait.pause():
            horizon = [r for r in rows if not r["ours"] and r["xid_old"]]
            raise Refused(
                [
                    Refusal(f"postgres {_named(r)}", "may have read `recording`: still open")
                    for r in waited
                ]
                + [
                    Refusal(
                        f"postgres {_named(r)}",
                        "another database's transaction, holding the snapshot horizon",
                    )
                    for r in horizon
                ]
            )


# -- the tenancy rows -------------------------------------------------------------------


async def write_tenancy(orgs: Iterable[str]) -> int:
    """The tenancy rows of `orgs`, on a marked session, as one statement per thousand; the
    rows written."""
    batch = sorted(orgs)
    written = 0
    for i in range(0, len(batch), 1000):
        status = await pg.execute(
            "INSERT INTO loom.tenancy (org) SELECT unnest($1::text[]) ON CONFLICT DO NOTHING",
            batch[i : i + 1000],
        )
        written += int(status.split()[-1])
    return written


async def read_mode(conn: Any) -> tuple[str, tuple[bool, str] | None]:
    """The cell's state (`cell.read_state`) and its `first` row."""
    state = await cell.read_state(conn)
    first = await conn.fetchrow("SELECT is_first, cell FROM loom.cell_first")
    return state, None if first is None else (bool(first["is_first"]), str(first["cell"]))
