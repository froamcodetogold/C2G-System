from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from c2g.report import create_report_bundle
from c2g.suite import run_canonical_suite, save_suite_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every canonical frozen C2G V1 backtest and build the report bundle."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "latest")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    results = run_canonical_suite(
        arguments.project_root,
        bootstrap_samples=arguments.bootstrap_samples,
    )
    save_suite_results(results, arguments.output_dir)
    files = create_report_bundle(results, arguments.output_dir)

    summary = results["summary"]
    selected = summary[
        (summary["scope"] == "ASSET_COMMON_PERIOD") & (summary["scenario"] == "ASSET_COMMON_COST")
    ][
        [
            "asset",
            "trades",
            "win_rate_pct",
            "profit_factor",
            "expectancy_pct",
            "simple_return_pct",
            "max_drawdown_pct",
            "classification",
        ]
    ]
    print()
    print("C2G V1.18 · CANONICAL BACKTEST SUITE")
    print("Frozen rules · common-period cross-asset · after costs")
    print(selected.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(f"CSV results: {arguments.output_dir}")
    print(f"Dashboard:   {files['dashboard']}")
    print(f"HTML report: {files['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
