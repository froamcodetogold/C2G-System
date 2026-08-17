from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from c2g import __version__
from c2g.config import FrozenV1Config
from c2g.data import load_ohlcv
from c2g.forward import (
    build_forward_snapshot,
    load_ledger,
    merge_forward_ledger,
    save_forward_state,
)
from c2g.strategy import prepare_signals

PROFILES = {
    "v114": {
        "freeze": "2026-08-15T19:00:00Z",
        "sources": {
            ("BTC", "BINANCE_SPOT"): "btc_data_binance_full_1h.csv",
            ("BTC", "BYBIT_PERPETUAL"): "btc_data_bybit_perp_1h.csv",
            ("BTC", "OKX_PERPETUAL"): "btc_data_okx_perp_1h.csv",
        },
    },
    "v117": {
        "freeze": "2026-08-15T20:00:00Z",
        "sources": {
            ("BTC", "BYBIT_PERPETUAL"): "btc_data_bybit_perp_1h.csv",
            ("ETH", "BYBIT_PERPETUAL"): "eth_data_bybit_perp_1h.csv",
            ("SOL", "BYBIT_PERPETUAL"): "sol_data_bybit_perp_1h.csv",
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update an append-only C2G forward-paper ledger.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v117")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    profile = PROFILES[arguments.profile]
    config = FrozenV1Config()
    observed_at = pd.Timestamp(datetime.now(UTC))
    snapshots: list[pd.DataFrame] = []
    quality: list[dict[str, object]] = []

    for (asset, market), filename in profile["sources"].items():
        frame, audit = load_ohlcv(arguments.project_root / filename)
        prepared = prepare_signals(frame, config)
        snapshots.append(
            build_forward_snapshot(
                prepared,
                asset=asset,
                market=market,
                freeze_time=profile["freeze"],
                config=config,
                observed_at=observed_at,
            )
        )
        quality.append({"asset": asset, "market": market, **audit.as_dict(), "path": filename})

    snapshot = pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()
    prefix = f"c2g_{arguments.profile}_forward"
    ledger_path = arguments.project_root / f"{prefix}_ledger.csv"
    summary_path = arguments.project_root / f"{prefix}_summary.csv"
    manifest_path = arguments.project_root / f"{prefix}_manifest.json"
    ledger = merge_forward_ledger(load_ledger(ledger_path), snapshot)
    manifest = {
        "engine_version": __version__,
        "profile": arguments.profile,
        "freeze_utc": profile["freeze"],
        "observed_at_utc": observed_at.isoformat(),
        "config": config.as_dict(),
        "config_hash": config.fingerprint,
        "data": quality,
        "paper_only": True,
        "live_orders": False,
        "funding_included": config.funding_pct_per_trade != 0,
    }
    save_forward_state(
        ledger,
        ledger_path=ledger_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        manifest=manifest,
        expected_pairs=list(profile["sources"].keys()),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Ledger rows: {len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
