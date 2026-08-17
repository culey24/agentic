import asyncio

from experiments.multiobj.processors import (
    MultiObjectiveEvolver,
    ObservationCaptureProcessor,
    ToolSummarizerProcessor,
)
from harnessx.core.processor import Order
from harnessx.events import BeforeModelEvent, Message, MessageRole, ToolResultEvent
from harnessx.processors.registry import create, kinds


def test_kinds_registered():
    assert "tool_summarizer" in kinds()
    assert "observation_capture" in kinds()


def _run(gen):
    async def collect():
        return [ev async for ev in gen]

    return asyncio.run(collect())


def test_tool_summarizer_condenses():
    proc = ToolSummarizerProcessor(max_chars=10)
    event = ToolResultEvent(name="web", result={"text": "x" * 100}, is_error=False)
    events = _run(proc.process(event))
    assert events[0].result["text"] == "x" * 10 + "…"


def test_observation_capture_appends_notes():
    proc = ObservationCaptureProcessor(max_notes=2)
    messages = [
        Message(role=MessageRole.TOOL, content="first result", name="a"),
        Message(role=MessageRole.TOOL, content="second result", name="b"),
        Message(role=MessageRole.TOOL, content="third result", name="c"),
    ]
    event = BeforeModelEvent(messages=messages)
    events = _run(proc.process(event))
    appended = events[0].messages[-1]
    assert appended.role == MessageRole.SYSTEM
    assert "Observation notes:" in appended.content
    assert "third result" in appended.content


def test_builder_can_create_processors():
    assert create("tool_summarizer", {"max_chars": 100})._singleton_group == "tool_summarizer"
    assert create("observation_capture", {"max_notes": 5})._order == Order.POST


def test_evolver_surface_includes_new_kinds():
    assert "tool_summarizer" in MultiObjectiveEvolver.EXTRA_KINDS
    assert "observation_capture" in MultiObjectiveEvolver.EXTRA_KINDS