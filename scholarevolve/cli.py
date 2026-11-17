"""Command line interface.

    python -m scholarevolve evolve   --config CONFIG --out RUN_DIR
    python -m scholarevolve evaluate --config CONFIG --spec SPEC --split test --out EVAL_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scholarevolve.config import RunConfig
from scholarevolve.envs import get_environment
from scholarevolve.search.evolve import Search
from scholarevolve.utils import load_json, safe_name


def _log(message: str) -> None:
    print(message, flush=True)


def _environment(config: RunConfig, out_dir: Path):
    settings = dict(config.environment)
    name = settings.pop("name")
    settings.setdefault("experiment_prefix", safe_name(out_dir.resolve().name))
    return get_environment(name, settings, backbone=config.backbone, work_dir=out_dir)


def cmd_evolve(args: argparse.Namespace) -> int:
    config = RunConfig.load(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    search = Search(environment=_environment(config, out_dir), research_llm=config.research_model,
                    coding=config.coding, settings=config.search, out_dir=out_dir, on_log=_log)
    state = search.run()
    print(json.dumps({"champion": state["champion"]["name"], "history": state["history"]},
                     indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = RunConfig.load(args.config)
    out_dir = Path(args.out)
    environment = _environment(config, out_dir)
    spec = load_json(args.spec)
    evaluation = environment.evaluate(spec, split=args.split, trials=args.trials,
                                      label=f"{safe_name(Path(args.spec).stem)}-{args.split}",
                                      out_dir=out_dir)
    print(json.dumps({"split": args.split, "trials": evaluation.trials,
                      "mean": evaluation.mean, "trial_means": evaluation.trial_means}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scholarevolve",
                                     description="Literature-guided agent harness evolution")
    sub = parser.add_subparsers(dest="command", required=True)
    evolve = sub.add_parser("evolve", help="run the generational harness search")
    evolve.add_argument("--config", required=True)
    evolve.add_argument("--out", required=True, help="run directory (resumable)")
    evolve.set_defaults(handler=cmd_evolve)
    evaluate = sub.add_parser("evaluate", help="evaluate a harness specification on a split")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--spec", required=True, help="harness spec, e.g. RUN_DIR/champion/spec.json")
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--trials", type=int, default=3)
    evaluate.add_argument("--out", required=True)
    evaluate.set_defaults(handler=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
