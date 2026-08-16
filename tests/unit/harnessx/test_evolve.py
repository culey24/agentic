import asyncio

from harnessx import HarnessConfig, ModelConfig
from harnessx.core.hooks import Hook
from harnessx.evolve import (
    Builder,
    Edit,
    EditOp,
    Ensemble,
    EvolutionLoop,
    Gate,
    pass_at_k,
)
from harnessx.evolve.manifest import ChangeManifest
from harnessx.tracing.journal import Journal
from tests.unit.harnessx.fakes import EchoProvider, Task


def test_pass_at_k() -> None:
    assert pass_at_k([True, False], 2) == 1.0
    assert pass_at_k([False, False], 2) == 0.0
    assert abs(pass_at_k([True, False], 1) - 0.5) < 1e-9


def test_builder_gate() -> None:
    async def main() -> None:
        base = HarnessConfig()
        manifest = ChangeManifest(
            id="c1",
            intended_effect="set a system prompt",
            edits=[
                Edit(
                    op=EditOp.INSERT,
                    hook="task_start",
                    group="system_prompt",
                    kind="system_prompt",
                    params={"prompt": "be concise"},
                )
            ],
        )
        gate = Gate(Builder())
        result = await gate.check(manifest, base)
        assert result.passed, result.reason
        assert result.config is not None
        procs = result.config.processors_for(Hook.TASK_START)
        assert any(p._singleton_group == "system_prompt" for p in procs)

    asyncio.run(main())


def test_gate_rejects_duplicate_group() -> None:
    async def main() -> None:
        base = HarnessConfig()
        manifest = ChangeManifest(
            id="c2",
            intended_effect="insert twice",
            edits=[
                Edit(op=EditOp.INSERT, hook="task_start", group="g", kind="system_prompt"),
                Edit(op=EditOp.INSERT, hook="task_start", group="g", kind="system_prompt"),
            ],
        )
        result = await Gate(Builder()).check(manifest, base)
        assert not result.passed
        assert result.stage in ("build", "normalize")

    asyncio.run(main())


def test_ensemble_routing() -> None:
    base = HarnessConfig()
    ens = Ensemble(base, max_variants=3)
    v0 = ens.variants[0]
    v0.record("t1", True)
    v0.record("t1", True)
    v1 = ens.fork(HarnessConfig())
    v1.record("t1", False)
    assert ens.route("t1") is v0
    assert ens.route("t2") is v0


def test_evolution_loop_end_to_end() -> None:
    async def verify(task: Task, final_output: str) -> bool:
        return final_output == "answer: 4"

    async def main() -> None:
        model = ModelConfig(main=EchoProvider("fake"))
        harness = model.agentic(HarnessConfig(max_steps=2))
        tasks = [Task(id=f"t{i}", description="x") for i in range(3)]
        journal = Journal("test_run")
        loop = EvolutionLoop(
            meta_provider=None,
            harness=harness,
            tasks=tasks,
            verifier=verify,
            journal=journal,
            n_rollouts=1,
            max_rounds=3,
            patience=3,
        )
        result = await loop.run()
        assert result["best"] == 1.0

    asyncio.run(main())
