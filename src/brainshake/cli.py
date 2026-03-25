"""Central CLI that discovers the Brainshake workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Mapping


@dataclass(frozen=True)
class CommandDef:
    """Describe a workflow that can be invoked via ``python -m`` or ``sbatch``."""

    name: str
    target: str | Path
    description: str
    category: str
    run_type: str = "python"


COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="analyze-data",
        target="brainshake.data_analyze.analyze_data",
        description="Run the batch analysis over CHB-MIT EEG windows.",
        category="data analysis",
    ),
    CommandDef(
        name="visualize-data",
        target="brainshake.data_analyze.visualize_data",
        description="Create the summary visualizations after analysis.",
        category="data visualization",
    ),
    CommandDef(
        name="train-cnn",
        target="brainshake.models.cnn.model",
        description="Train the convolutional classifier or run its training command set.",
        category="model training",
    ),
    CommandDef(
        name="evaluate-cnn",
        target="brainshake.models.cnn.evaluate",
        description="Patient-wise evaluation for the CNN model.",
        category="model evaluation",
    ),
    CommandDef(
        name="evaluate-randomforest",
        target="brainshake.models.randomforest.evaluate",
        description="Patient-wise evaluation for the random forest classifier.",
        category="model evaluation",
    ),
    CommandDef(
        name="evaluate-threshold",
        target="brainshake.models.threshold.evaluate",
        description="Evaluate the lightweight threshold baseline.",
        category="model evaluation",
    ),
    CommandDef(
        name="plot-benchmarks",
        target="brainshake.plotting.plots",
        description="Generate benchmark plots from saved metrics.",
        category="graph plotting",
    ),
)

COMMAND_LOOKUP: Mapping[str, CommandDef] = {cmd.name: cmd for cmd in COMMANDS}


def _group_by_category() -> dict[str, list[CommandDef]]:
    grouped: dict[str, list[CommandDef]] = {}
    for command in COMMANDS:
        grouped.setdefault(command.category, []).append(command)
    return grouped


def list_commands(category: str | None = None) -> None:
    grouped = _group_by_category()
    categories = [category] if category else sorted(grouped)
    for cat in categories:
        if cat not in grouped:
            continue
        print(f"{cat.title()}")
        for command in grouped[cat]:
            print(f"  {command.name.ljust(30)}{command.description}")


def run_command(command: CommandDef, extra_args: Sequence[str]) -> int:
    extra = list(extra_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    args = [sys.executable, "-m", str(command.target), *extra]
    result = subprocess.run(args)
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and run Brainshake workflows from data analysis to plotting."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    list_parser = subparsers.add_parser("list", help="List available workflows.")
    list_parser.add_argument(
        "--category",
        choices=sorted({cmd.category for cmd in COMMANDS}),
        help="Limit the listing to a single category.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Invoke a workflow by name and forward any additional arguments.",
    )
    run_parser.add_argument(
        "command",
        choices=list(COMMAND_LOOKUP),
        help="Command to execute.",
    )
    run_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected workflow.",
    )

    subparsers.add_parser(
        "compile", help="Run the entire pipeline from analysis to plotting."
    )

    return parser.parse_args()


def compile_pipeline() -> int:
    outcome = 0
    print("\n=== Running full Brainshake pipeline ===")
    for name in (
        "analyze-data",
        "visualize-data",
        "train-cnn",
        "evaluate-cnn",
        "evaluate-randomforest",
        "evaluate-threshold",
        "plot-benchmarks",
    ):
        command = COMMAND_LOOKUP[name]
        print(f"\n==> {name} ({command.run_type})")
        status = run_command(command, [])
        if status != 0:
            print(f"command {name} failed with exit code {status}")
            outcome = status
            break
    print("=== Pipeline complete ===")
    return outcome


def main() -> int:
    args = parse_args()
    if args.subcommand == "list":
        list_commands(category=args.category)
        return 0

    if args.subcommand == "compile":
        return compile_pipeline()

    assert args.subcommand == "run"
    command = COMMAND_LOOKUP[args.command]
    extra_args = args.args or []
    return run_command(command, extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
