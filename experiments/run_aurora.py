"""Run a deterministic AURORA smoke experiment and emit JSON metrics."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Permit ``python experiments/run_aurora.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.aurora.controller import AURORAController
from backend.aurora.scenario import Scenario, ScenarioConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = AURORAController(
        Scenario.generate(ScenarioConfig(seed=args.seed)), seed=args.seed
    ).optimize()
    payload = {"seed": args.seed, "result": result.__dict__}
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
