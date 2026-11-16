"""LoomMemoryService against a running Loom: a turn an ADK Runner saves is
extracted, and preload_memory puts it in a later session's prompt for the same
user only; a memory, extracted or explicit, is deleted by the id a search
returned."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from tests.integration._eventually import eventually

# An SDK extra, outside the server's environment: install google-adk to run these.
pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import preload_memory
from google.genai import types
from loom.integrations.google_adk import LoomMemoryService
from pydantic import Field


class _EchoLlm(BaseLlm):
    """Answers "Noted." to every call and keeps the requests."""

    model: str = "echo"
    requests: list[LlmRequest] = Field(default_factory=list)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Noted.")]))


async def _save_turn(ctx: Any) -> None:
    turn = [e for e in ctx.session.events if e.invocation_id == ctx.invocation_id]
    await ctx.add_events_to_memory(events=turn)


async def _turn(service: LoomMemoryService, user: str, text: str) -> str:
    """One turn in a new session of the app "travel"; returns the model's prompt text."""
    llm = _EchoLlm()
    sessions = InMemorySessionService()
    runner = Runner(
        app_name="travel",
        agent=LlmAgent(
            name="assistant", model=llm, tools=[preload_memory], after_agent_callback=_save_turn
        ),
        session_service=sessions,
        memory_service=service,
    )
    session = await sessions.create_session(app_name="travel", user_id=user)
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for _ in runner.run_async(user_id=user, session_id=session.id, new_message=message):
        pass
    return " ".join(p.text or "" for c in llm.requests[0].contents for p in c.parts or [])


async def test_a_saved_turn_reaches_a_later_prompt(agent) -> None:
    # defer=False: extraction inline, since the test server runs no background workers
    service = LoomMemoryService(agent, app_name="travel", honour_abstain=False, defer=False)
    await _turn(service, "alice", "I fly to Lisbon every month.")

    async def recalled() -> bool:
        prompt = await _turn(service, "alice", "Where does Alice fly every month?")
        return "lisbon" in prompt.lower()

    # A write is searchable once its embedding lands; eventually raises on timeout.
    await eventually(recalled, description="'lisbon' in the next session's prompt")
    found = await service.search_memory(app_name="travel", user_id="bob", query="Lisbon")
    assert not found.memories
    found = await service.search_memory(app_name="travel", user_id="alice", query="Lisbon")
    facts = {m.id for m in found.memories if m.custom_metadata["source"] == "fact"}
    assert facts
    for memory_id in facts:
        await service.delete_memory(app_name="travel", user_id="alice", memory_id=memory_id)
    found = await service.search_memory(app_name="travel", user_id="alice", query="Lisbon")
    # The deleted facts do not come back.
    assert not {m.id for m in found.memories} & facts


async def test_add_memory_then_delete_by_id(agent) -> None:
    service = LoomMemoryService(agent, app_name="travel", honour_abstain=False)
    entry = MemoryEntry(
        content=types.Content(parts=[types.Part(text="Alice prefers aisle seats")]),
        timestamp="2026-03-01T09:00:00+00:00",
    )
    await service.add_memory(app_name="travel", user_id="alice", memories=[entry])

    async def ids(user: str) -> list[str | None]:
        found = await service.search_memory(app_name="travel", user_id=user, query="aisle seat")
        return [m.id for m in found.memories]

    await eventually(lambda: ids("alice"), description="the memory found")
    (memory_id,) = await ids("alice")
    assert memory_id is not None
    row = await agent.memory.get(memory_id, user_id="alice")
    assert row is not None
    fields = ("value", "agent", "user_id", "session_id", "memory_type", "source_class", "tags")
    assert [row[f] for f in fields] == [
        "Alice prefers aisle seats",
        agent.agent,
        "alice",
        "-",
        "semantic",
        "asserted",
        [],
    ]
    assert row["valid_at"].startswith("2026-03-01")
    await service.delete_memory(app_name="travel", user_id="bob", memory_id=memory_id)
    assert await ids("alice") == [memory_id]  # another user's delete reaches nothing
    await service.delete_memory(app_name="travel", user_id="alice", memory_id=memory_id)
    assert await ids("alice") == []


async def test_two_users_saving_the_same_text_each_find_their_own(agent) -> None:
    service = LoomMemoryService(agent, app_name="travel", honour_abstain=False)
    entry = MemoryEntry(content=types.Content(parts=[types.Part(text="Prefers window seats")]))

    async def ids(user: str) -> list[str | None]:
        found = await service.search_memory(app_name="travel", user_id=user, query="window seat")
        return [m.id for m in found.memories]

    await service.add_memory(app_name="travel", user_id="carol", memories=[entry])

    async def carols_vector() -> list[str]:
        hits = await agent.memory.search("window seats", memory_type="semantic", top_k=5)
        return [h.memory_key for h in hits]

    # Loom's write-time check reads vectors: dave saves once carol's is searchable.
    await eventually(carols_vector, description="carol's vector")
    await service.add_memory(app_name="travel", user_id="dave", memories=[entry])
    carol = await ids("carol")
    dave = await eventually(lambda: ids("dave"), description="dave's memory")
    assert len(carol) == len(dave) == 1 and carol != dave
