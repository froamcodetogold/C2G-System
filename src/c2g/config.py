from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class FrozenV1Config:
    """Parameters frozen before the V1.14/V1.17 forward observation."""

    supertrend_length: int = 10
    supertrend_multiplier: float = 3.0
    adx_length: int = 14
    adx_min: float = 25.0
    ema_length: int = 200
    ema_slope_lookback: int = 50
    time_exit_bars: int = 24
    fee_pct_per_side: float = 0.055
    slippage_pct_per_side: float = 0.020
    funding_pct_per_trade: float = 0.0
    timeframe_hours: int = 1
    strategy_version: str = "C2G-V1-FROZEN-BUY24H"

    @property
    def round_trip_cost_pct(self) -> float:
        return (
            2.0 * self.fee_pct_per_side
            + 2.0 * self.slippage_pct_per_side
            + self.funding_pct_per_trade
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


BYBIT_ASSETS = {
    "BTC": "btc_data_bybit_perp_1h.csv",
    "ETH": "eth_data_bybit_perp_1h.csv",
    "SOL": "sol_data_bybit_perp_1h.csv",
    "XRP": "xrp_data_bybit_perp_1h.csv",
    "BNB": "bnb_data_bybit_perp_1h.csv",
}

BTC_MARKETS = {
    "BINANCE_SPOT": "btc_data_binance_full_1h.csv",
    "BYBIT_PERPETUAL": "btc_data_bybit_perp_1h.csv",
    "OKX_PERPETUAL": "btc_data_okx_perp_1h.csv",
}
