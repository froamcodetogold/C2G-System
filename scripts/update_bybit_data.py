from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from c2g.market_data import update_bybit_file

FILES = {
    "BTCUSDT": "btc_data_bybit_perp_1h.csv",
    "ETHUSDT": "eth_data_bybit_perp_1h.csv",
    "SOLUSDT": "sol_data_bybit_perp_1h.csv",
    "XRPUSDT": "xrp_data_bybit_perp_1h.csv",
    "BNBUSDT": "bnb_data_bybit_perp_1h.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally refresh C2G Bybit 1H data.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--forward-only", action="store_true", help="Update only BTC, ETH and SOL.")
    arguments = parser.parse_args()
    selected = {
        key: value
        for key, value in FILES.items()
        if not arguments.forward_only or key in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    }
    results = [
        update_bybit_file(arguments.project_root / filename, symbol=symbol)
        for symbol, filename in selected.items()
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
