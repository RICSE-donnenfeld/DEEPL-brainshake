"""Central CLI that discovers the Brainshake workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence, Mapping


@dataclass(frozen=True)
class CommandDef:
    """Describe a workflow that can be invoked via ``python -m module``."""

    name: str
    module: str
    description: str
    category: str


COMMANDS: tuple[CommandDef, ...] = (
    CommandDef(
        name="analyze-data",
        module="brainshake.data_analyze.analyze_data",
        description="Run the batch analysis over CHB-MIT EEG windows.",
        category="data analysis",
    ),
    CommandDef(
        name="visualize-data",
        module="brainshake.data_analyze.visualize_data",
        description="Create the summary visualizations after analysis.",
        category="data visualization",
    ),
    CommandDef(
        name="train-cnn",
        module="brainshake.models.cnn.model",
        description="Train the convolutional classifier or run its training command set.",
        category="model training",
    ),
    CommandDef(
        name="evaluate-cnn",
        module="brainshake.models.cnn.evaluate",
        description="Patient-wise evaluation for the CNN model.",
        category="model evaluation",
    ),
    CommandDef(
        name="evaluate-randomforest",
        module="brainshake.models.randomforest.evaluate",
        description="Patient-wise evaluation for the random forest classifier.",
        category="model evaluation",
    ),
    CommandDef(
        name="evaluate-threshold",
        module="brainshake.models.threshold.evaluate",
        description="Evaluate the lightweight threshold baseline.",
        category="model evaluation",
    ),
    CommandDef(
        name="plot-benchmarks",
        module="brainshake.plotting.plots",
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
            print(f"  {command.name.ljust(25)}{command.description}")


def run_command(command: CommandDef, extra_args: Sequence[str]) -> int:
    args = [sys.executable, "-m", command.module, *extra_args]
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

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.subcommand == "list":
        list_commands(category=args.category)
        return 0

    assert args.subcommand == "run"
    command = COMMAND_LOOKUP[args.command]
    extra_args = list(args.args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    return run_command(command, extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
