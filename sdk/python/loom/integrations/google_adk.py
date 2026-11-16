"""Loom as a Google ADK memory service.

``LoomMemoryService`` implements ``google.adk.memory.BaseMemoryService``, the
service a ``Runner`` takes as ``memory_service=``. ADK's memory tools read it:
``preload_memory`` searches with the user's message before each model call,
``load_memory`` lets the model search. A callback saves each turn::

    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.tools import preload_memory
    from loom import Agent
    from loom.integrations.google_adk import LoomMemoryService


    async def save_turn(ctx):
        turn = [e for e in ctx.session.events if e.invocation_id == ctx.invocation_id]
        await ctx.add_events_to_memory(events=turn)


    async with Agent(org="acme", namespace="support") as loom_agent:
        runner = Runner(
            app_name="support",
            agent=LlmAgent(
                name="assistant",
                model="gemini-flash-latest",
                tools=[preload_memory],
                after_agent_callback=save_turn,
            ),
            session_service=InMemorySessionService(),
            memory_service=LoomMemoryService(loom_agent, app_name="support"),
        )

ADK scopes memory by app and user. Loom's boundary is the namespace, so a
service serves one ADK app, the namespace of its Loom agent, and refuses a
call for another app; the ADK user is the Loom ``user_id``, and extraction
runs under a Loom session derived from the user and the ADK session. Install with
``pip install 'clickhouse-loom[google-adk]'``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

try:
    from google.adk.events import Event, get_active_events
    from google.adk.memory import BaseMemoryService
    from google.adk.memory.base_memory_service import SearchMemoryResponse
    from google.adk.memory.memory_entry import MemoryEntry
    from google.adk.sessions import Session
    from google.genai import types
except ImportError as e:  # the framework is an optional extra
    raise ImportError(
        "loom.integrations.google_adk needs google-adk: pip install 'clickhouse-loom[google-adk]'"
    ) from e

from loom.sync import SyncAgent

if TYPE_CHECKING:
    from loom.agent import Agent


class LoomMemoryService(BaseMemoryService):
    """ADK ``BaseMemoryService`` over a Loom ``Agent``, for one ADK app.

    Saved sessions and turns go to ``memory.set_from_messages``, ``add_memory`` to
    ``memory.set`` and ``search_memory`` to ``memory.search`` of the ADK user's
    memories; a returned memory's ``id`` is the key ``delete_memory`` deletes. ADK
    awaits its memory service, so the service takes a ``loom.Agent``. The mapping of
    each method is in ``docs/features/integrations/frameworks.md`` (Google ADK).
    """

    def __init__(
        self,
        agent: Agent,
        *,
        app_name: str,
        top_k: int = 10,
        honour_abstain: bool = True,
        defer: bool = True,
        key_prefix: str = "adk",
    ) -> None:
        if isinstance(agent, SyncAgent):
            raise TypeError("LoomMemoryService needs a loom.Agent: ADK awaits its memory service")
        if not app_name:
            raise ValueError("app_name must name the ADK app this service serves")
        if not 1 <= top_k <= 500:  # memory.search's bounds
            raise ValueError("top_k must be between 1 and 500")
        key_prefix = key_prefix.strip("/")
        if not key_prefix:
            raise ValueError("key_prefix must not be empty")
        self._agent = agent
        self._app_name = app_name
        self._top_k = top_k
        self._honour_abstain = honour_abstain
        self._defer = defer
        self._key_prefix = key_prefix

    async def add_session_to_memory(self, session: Session) -> None:
        self._check_scope(session.app_name, session.user_id)
        await self._extract(session.user_id, session.id, get_active_events(session.events))

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._check_scope(app_name, user_id)
        if not session_id:
            # Loom files a turn's verbatim chunks by session alone, not by user.
            raise ValueError("add_events_to_memory needs the ADK session_id")
        await self._extract(user_id, session_id, get_active_events(list(events)))

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._check_scope(app_name, user_id)
        for memory in memories:
            text = _text(memory.content)
            if not text:
                continue
            # The user is in the key: Loom files a long value's passages by key alone.
            digest = hashlib.sha256(f"{user_id}\0{text}".encode()).hexdigest()[:20]
            await self._agent.memory.set(
                f"{self._key_prefix}/{digest}",
                text,
                user_id=user_id,
                valid_at=_iso_time(memory.timestamp),
            )

    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        self._check_scope(app_name, user_id)
        if not query.strip():
            return SearchMemoryResponse()
        result = await self._agent.memory.search(
            query,
            top_k=self._top_k,
            user_id=user_id,
            # Read the namespace this service writes to, whatever the deployment's default.
            namespace_scope="this",
            # Pinned so every top_k gets full bodies and no per-hit explanation
            # (the server's defaults switch at 15).
            bodies="full",
            explain=False,
        )
        hits = list(result)
        if self._honour_abstain:
            if result.abstain:
                return SearchMemoryResponse()
            # A response that does not abstain can still carry rows below the floor.
            hits = [hit for hit in hits if not hit.below_threshold]
        return SearchMemoryResponse(
            memories=[
                MemoryEntry(
                    # The key memory.delete takes: a memory's, or for a passage of a long
                    # value the key it was written under; a conversation's chunks have none.
                    id=hit.memory_key if hit.source == "fact" else (hit.parent_memory_key or None),
                    content=types.Content(
                        role="user", parts=[types.Part(text=hit.content_excerpt)]
                    ),
                    custom_metadata={"similarity": hit.similarity, "source": hit.source},
                    timestamp=hit.valid_at or None,
                )
                # Derived rows may come back above the top_k slice: cut after filtering.
                for hit in hits[: self._top_k]
            ]
        )

    async def delete_memory(
        self, *, app_name: str, user_id: str, memory_id: str | None = None
    ) -> None:
        self._check_scope(app_name, user_id)
        if memory_id:  # an entry with no id has nothing a key deletes, as in ADK's own service
            await self._agent.memory.delete(memory_id, user_id=user_id)

    def _check_scope(self, app_name: str, user_id: str) -> None:
        # Loom merges a write into a matching memory of the namespace whichever agent
        # wrote it, so two apps in one namespace would share and lose memories.
        if app_name != self._app_name:
            raise ValueError(
                f"this LoomMemoryService serves the ADK app {self._app_name!r}, not "
                f"{app_name!r}: give each app its own Loom namespace and service"
            )
        # In Loom an empty or "-" user is no user: writes would land outside the user's reads.
        if user_id in ("", "-"):
            raise ValueError(f"an ADK user_id names a Loom user; {user_id!r} names none")

    async def _extract(self, user_id: str, session_id: str, events: Sequence[Event]) -> None:
        messages: list[dict[str, object]] = []
        for event in events:
            text = _text(event.content)
            if text:
                messages.append(
                    {
                        "role": "user" if event.author == "user" else "assistant",
                        "content": text,
                        "timestamp": datetime.fromtimestamp(event.timestamp, UTC).isoformat(),
                    }
                )
        if messages:
            await self._agent.memory.set_from_messages(
                messages,
                user_id=user_id,
                session_id=_loom_session(user_id, session_id),
                defer=self._defer,
            )


def _loom_session(user_id: str, session_id: str) -> str:
    """The Loom session of an ADK user's session: ADK makes a session id unique per
    user only, and Loom files a turn's verbatim chunks by session alone."""
    pair = json.dumps([user_id, session_id])
    return f"adk-{hashlib.sha256(pair.encode()).hexdigest()[:20]}"


def _iso_time(text: str | None) -> datetime | None:
    """An ISO 8601 timestamp; ADK allows any text, which is not a ``valid_at``."""
    try:
        return datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def _text(content: types.Content | None) -> str:
    """The text of a content, without the model's thoughts or tool calls."""
    if content is None or not content.parts:
        return ""
    return " ".join(p.text for p in content.parts if p.text and not p.thought).strip()
