"""Command-line entry point (``hx``).

Supports non-interactive invocation (``hx run`` / ``hx evolve``) and an
interactive mode when run with no subcommand. Defaults are loaded from
``~/.harnessx/config.json`` (created/updated via ``set``) and API keys from a
``.env`` file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from harnessx import HarnessConfig, ModelConfig
from harnessx.providers.anthropic_provider import AnthropicProvider
from harnessx.providers.openai_provider import OpenAIProvider
from harnessx.providers.openrouter_provider import OpenRouterProvider

PROVIDERS = ("anthropic", "openai", "openrouter")


def _load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _config_path() -> Path:
    return Path.home() / ".harnessx" / "config.json"


def _load_config() -> dict:
    try:
        return json.loads(_config_path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_config(cfg: dict) -> None:
    _config_path().parent.mkdir(parents=True, exist_ok=True)
    _config_path().write_text(json.dumps(cfg, indent=2))


@dataclass
class Settings:
    model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"
    meta_model: str = "claude-opus-4-6"
    benchmark: str = "gaia"
    data: str | None = None
    rounds: int = 15
    rollouts: int = 2
    extra: dict = field(default_factory=dict)


def _make_provider(name: str, model: str):
    if name == "anthropic":
        return AnthropicProvider(model)
    if name == "openai":
        return OpenAIProvider(model)
    if name == "openrouter":
        return OpenRouterProvider(model)
    raise ValueError(f"unknown provider {name!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hx", description="HarnessX CLI")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a single task")
    run.add_argument("task", help="task description")
    run.add_argument("-p", "--print-only", action="store_true")
    run.add_argument("--model", default=None)
    run.add_argument("--provider", choices=PROVIDERS, default=None)

    evolve = sub.add_parser("evolve", help="run harness evolution on a benchmark")
    evolve.add_argument("--benchmark", default=None)
    evolve.add_argument("--data", default=None, help="path to benchmark data (JSONL)")
    evolve.add_argument("--rounds", type=int, default=None)
    evolve.add_argument("--model", default=None)
    evolve.add_argument("--meta-model", default=None)
    evolve.add_argument("--provider", choices=PROVIDERS, default=None)
    evolve.add_argument("--rollouts", type=int, default=None)

    return parser


class _Task:
    def __init__(self, description: str) -> None:
        self.description = description

    @property
    def id(self) -> str:
        return "cli-task"


async def _run_task(settings: Settings, task: str) -> None:
    provider = _make_provider(settings.provider, settings.model)
    model = ModelConfig(main=provider)
    harness = model.agentic(HarnessConfig())
    result = await harness.run(_Task(task))
    print(result.final_output)


async def _evolve(settings: Settings) -> None:
    from harnessx.benchmarks.alfworld import ALFWorldAdapter
    from harnessx.benchmarks.gaia import GAIAAdapter
    from harnessx.benchmarks.swebench import SWEBenchAdapter, SWEHarness
    from harnessx.benchmarks.tau3.adapter import DialogueHarness, Tau3Adapter
    from harnessx.benchmarks.text_env import TextGameHarness
    from harnessx.benchmarks.webshop import WebShopAdapter
    from harnessx.evolve.loop import EvolutionLoop
    from harnessx.tracing.journal import Journal

    if not settings.data:
        print("no --data set; use `set data <path>` or pass --data")
        raise SystemExit(1)

    task_provider = _make_provider(settings.provider, settings.model)
    meta_provider = _make_provider(settings.provider, settings.meta_model)
    model = ModelConfig(main=task_provider, meta=meta_provider)

    benchmark = settings.benchmark
    if benchmark == "gaia":
        adapter = GAIAAdapter(data_path=settings.data)
        harness = model.agentic(adapter.default_harness_config())
    elif benchmark == "tau3":
        adapter = Tau3Adapter(data_path=settings.data)
        harness = DialogueHarness(model, adapter.default_harness_config())
    elif benchmark == "alfworld":
        adapter = ALFWorldAdapter(data_path=settings.data)
        harness = TextGameHarness(model, adapter.default_harness_config())
    elif benchmark == "webshop":
        adapter = WebShopAdapter(data_path=settings.data)
        harness = TextGameHarness(model, adapter.default_harness_config())
    elif benchmark == "swebench":
        adapter = SWEBenchAdapter(data_path=settings.data)
        harness = SWEHarness(model, adapter.default_harness_config())
    else:
        print(f"unknown benchmark {benchmark!r}")
        raise SystemExit(1)

    tasks = adapter.load_tasks()
    verifier = adapter.verifier()

    journal = Journal(f"{benchmark}_evolution")
    loop = EvolutionLoop(
        meta_provider=meta_provider,
        harness=harness,
        tasks=tasks,
        verifier=verifier,
        journal=journal,
        n_rollouts=settings.rollouts,
        max_rounds=settings.rounds,
    )
    result = await loop.run()
    print(json.dumps(result, indent=2))


_SETTABLE = ("model", "provider", "meta_model", "benchmark", "data", "rounds", "rollouts")


def _settings_from_args(args: argparse.Namespace, config: dict) -> Settings:
    settings = Settings(**{k: v for k, v in config.items() if k in Settings.__dataclass_fields__})
    if args.command == "run":
        for key in ("model", "provider"):
            value = getattr(args, key)
            if value is not None:
                setattr(settings, key, value)
    elif args.command == "evolve":
        for key in ("benchmark", "data", "rounds", "model", "meta_model", "provider", "rollouts"):
            value = getattr(args, key)
            if value is not None:
                setattr(settings, key, value)
    return settings


_REQUIREMENTS = (
    ("ANTHROPIC_API_KEY", "anthropic provider"),
    ("OPENAI_API_KEY", "openai provider"),
    ("OPENROUTER_API_KEY", "openrouter provider"),
    ("BRAVE_API_KEY", "search tool"),
    ("TAVILY_API_KEY", "search tool"),
)


def _set(settings: Settings, key: str, value: str) -> None:
    if key not in _SETTABLE:
        print(f"unknown key {key!r}; settable: {', '.join(_SETTABLE)}")
        return
    if key in ("rounds", "rollouts"):
        setattr(settings, key, int(value))
    else:
        setattr(settings, key, value)
    _save_config(asdict(settings))


def _render_dashboard(settings: Settings) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    console.clear()

    header = Text("HARNESS X", style="bold white on blue")
    header.append("  Compose · Adapt · Evolve", style="bold")
    console.print(Panel(header, border_style="blue"))

    req = Table(title="Requirements", title_style="bold", box=None)
    req.add_column("Variable", style="cyan")
    req.add_column("Status", style="bold")
    req.add_column("Used by", style="dim")
    for var, use in _REQUIREMENTS:
        if os.environ.get(var):
            status = "[green]✓ set[/]"
        else:
            status = "[yellow]? optional[/]" if var in ("BRAVE_API_KEY", "TAVILY_API_KEY") else "[red]✗ missing[/]"
        req.add_row(var, status, use)
    console.print(req)

    st = Table(title="Settings", title_style="bold", box=None)
    st.add_column("Key", style="cyan")
    st.add_column("Value", style="bold")
    for key in _SETTABLE:
        st.add_row(key, str(getattr(settings, key)))
    if settings.data:
        data_ok = Path(settings.data).exists()
        st.add_row("data status", "[green]exists[/]" if data_ok else "[red]missing[/]")
    console.print(st)

    commands = (
        "run <task>        run a single task\n"
        "evolve [bench]    run harness evolution (needs `data`)\n"
        "set <key> <val>   change a setting (saved to ~/.harnessx/config.json)\n"
        "edit              open the config file in $EDITOR\n"
        "refresh           reload .env and config\n"
        "help / quit       this help / exit\n"
        "\nAny other input is treated as a task to run."
    )
    console.print(Panel(commands, title="Commands", border_style="green"))
    console.print("[dim]Type a command above, then press Enter.[/]")


def _interactive(settings: Settings) -> None:
    _render_dashboard(settings)
    while True:
        try:
            line = input("hx> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()
        if cmd in ("q", "quit", "exit"):
            break
        if cmd == "help":
            _render_dashboard(settings)
        elif cmd in ("show", "refresh", "r"):
            _load_dotenv()
            config = _load_config()
            for key in _SETTABLE:
                if key in config:
                    setattr(settings, key, config[key])
            _render_dashboard(settings)
        elif cmd == "edit":
            _edit_config()
            _render_dashboard(settings)
        elif cmd == "set" and len(parts) >= 3:
            _set(settings, parts[1], " ".join(parts[2:]))
            _render_dashboard(settings)
        elif cmd == "evolve":
            if len(parts) > 1:
                settings.benchmark = parts[1]
            _save_config(asdict(settings))
            asyncio.run(_evolve(settings))
            _render_dashboard(settings)
        else:
            task = line[len("run"):].strip() if cmd == "run" else line
            if task:
                asyncio.run(_run_task(settings, task))
                _render_dashboard(settings)


def _edit_config() -> None:
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL"))
    path = _config_path()
    if editor:
        os.system(f"{editor} {path}")
    else:
        print(f"set $EDITOR to edit the config; path: {path}")


def main() -> None:
    _load_dotenv()
    config = _load_config()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in ("run", "evolve"):
        settings = _settings_from_args(args, config)
        if args.command == "run":
            asyncio.run(_run_task(settings, args.task))
        else:
            asyncio.run(_evolve(settings))
    else:
        _interactive(Settings(**{k: v for k, v in config.items() if k in Settings.__dataclass_fields__}))


if __name__ == "__main__":
    main()
