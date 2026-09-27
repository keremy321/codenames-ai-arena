"""python -m codenames_ai.evaluation [--case NAME ...] [--repeat N] [--json PATH]"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from codenames_ai.config import Settings
from codenames_ai.llm.ollama import OllamaClient

from .cases import OPERATIVE_CASES, SPYMASTER_CASES
from .harness import CaseResult, run_operative_case, run_spymaster_case


async def evaluate(args: argparse.Namespace, settings: Settings) -> list[CaseResult]:
    spy = [c for c in SPYMASTER_CASES if not args.case or c.name in args.case]
    ops = [c for c in OPERATIVE_CASES if not args.case or c.name in args.case]
    if args.kind == "spymaster":
        ops = []
    elif args.kind == "operative":
        spy = []
    results: list[CaseResult] = []
    async with OllamaClient(
        settings.ollama_base_url, settings.ollama_model, timeout=settings.ollama_timeout
    ) as llm:
        for run in range(args.repeat):
            for case in [*spy, *ops]:
                runner = run_spymaster_case if case in spy else run_operative_case
                result = await runner(case, llm)  # type: ignore[arg-type]
                results.append(result)
                header = f"== {case.name} ({result.kind})" + (
                    f" run {run + 1}" if args.repeat > 1 else ""
                )
                print(f"\n{header}: {case.purpose}")
                for line in result.lines:
                    print(f"  {line}")
                if result.error:
                    print(f"  ERROR: {result.error}")
                for check, ok in result.checks.items():
                    print(f"  [{'PASS' if ok else 'FAIL'}] {check}")
    return results


def summarize(results: list[CaseResult]) -> None:
    print("\n== summary")
    print(f"  {'case':28} {'kind':10} {'result':6} {'calls':>5} {'seconds':>8}")
    for r in results:
        print(
            f"  {r.name:28} {r.kind:10} {'pass' if r.passed else 'FAIL':6} "
            f"{len(r.calls):5d} {r.seconds:8.1f}"
        )
    for kind in ("spymaster", "operative"):
        chosen = [r for r in results if r.kind == kind]
        if chosen:
            passed = sum(r.passed for r in chosen)
            print(f"  {kind}: {passed}/{len(chosen)} cases passed every check")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Codenames decision evaluation")
    parser.add_argument("--case", action="append", help="Run only this case (repeatable)")
    parser.add_argument("--kind", choices=["spymaster", "operative"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json", type=Path, help="Write machine-readable results here")
    parser.add_argument("--list", action="store_true", help="List cases and exit")
    parser.add_argument("--verbose", action="store_true", help="Show agent INFO logs")
    args = parser.parse_args()
    if args.list:
        for case in [*SPYMASTER_CASES, *OPERATIVE_CASES]:
            print(f"{case.name:28} {case.purpose}")
        return 0
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    results = asyncio.run(evaluate(args, Settings.from_env()))
    summarize(results)
    if args.json:
        args.json.write_text(json.dumps([r.as_dict() for r in results], indent=2))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
