From Code to Gold — C2G System

Build. Test. Fail. Learn. Improve.

From Code to Gold (C2G) is an experimental algorithmic trading research project focused on designing, validating, and forward-testing systematic trading logic for cryptocurrency markets.

The project is developed as a software engineering research initiative, with an emphasis on reproducibility, robustness, cross-market validation, transparent failure analysis, and disciplined separation between research and forward testing.

The current research stack is primarily implemented in Python. A future production layer is planned for MetaTrader 5 / MQL5, where validated research components may be translated into a custom indicator and, later, an Expert Advisor.

Table of Contents

Project Status

Project Goals

Research Philosophy

Current Frozen Strategy

Why These Components Work Together

Research Evolution

Cross-Market Validation

Cross-Asset Validation

Current Findings

Repository Structure

Installation

Data Loaders

Running the Forward Tests

Research Methodology

Metrics

Forward Testing

Planned MQL5 Architecture

Limitations

Roadmap

Versioning Policy

Disclaimer

License

Project Status

Current Stage

Active research status: Forward testing

The first generation of the strategy has completed multiple historical research phases and is currently frozen for forward observation.

Active forward versions

Version

Purpose

Assets / Markets

Status

V1.14

BTC true forward paper logger

BTC across previously validated research workflow

Active

V1.17

Multi-asset true forward logger

BTC, ETH, SOL — Bybit Perpetual 1H

Active

Historical experiments are preserved under archive/.

Important

The currently frozen strategy must not be modified while forward data is being collected.

Any change to:

indicators;

thresholds;

timeframe;

entry conditions;

exit logic;

supported assets;

must create a new research branch/version with a new freeze timestamp.

Project Goals

C2G is designed around four primary goals.

1. Research

Convert discretionary technical-analysis ideas into deterministic, testable rules.

2. Engineering

Build a reproducible software pipeline for:

historical market data;

indicator computation;

backtesting;

diagnostics;

robustness testing;

multi-exchange comparison;

multi-asset comparison;

forward logging.

3. Validation

Avoid trusting a strategy only because it performs well on one dataset.

The project intentionally tests hypotheses across:

different time periods;

different exchanges;

different crypto assets;

realistic trading-cost assumptions;

bootstrap resampling;

leave-one-year-out analysis;

outlier dependency.

4. Product Development

Translate sufficiently validated research into:

a MetaTrader 5 indicator;

alerting infrastructure;

later, an Expert Advisor;

transparent technical documentation.

Research Philosophy

The central rule of the project is:

A better backtest is not automatically better evidence.

The research process prioritizes:

Hypothesis
    ↓
Implementation
    ↓
Backtest
    ↓
Failure analysis
    ↓
Robustness checks
    ↓
Cross-market validation
    ↓
Cross-asset validation
    ↓
Frozen rules
    ↓
Forward testing

The project explicitly avoids continuously changing parameters until historical performance looks attractive.

This distinction is critical:

Optimization goal:
"Find the highest historical Profit Factor."

C2G goal:
"Find evidence that survives multiple independent challenges."

Current Frozen Strategy

Timeframe

1 Hour

Direction

BUY ONLY

SELL research was suspended after directional diagnostics consistently showed weaker or negative post-entry behavior.

Entry Stack

The current frozen BUY signal requires all of the following conditions:

1. Bullish Regime

Close > EMA200
AND
EMA200[current] > EMA200[50 bars ago]

This attempts to confirm that price is above a long-term trend reference and that the reference itself is rising.

2. Supertrend Flip

Supertrend Length = 10
Supertrend Multiplier = 3.0

Previous Direction = bearish
Current Direction  = bullish

The Supertrend is used as the structural event trigger.

3. ADX Strength

ADX Length = 14
ADX > 25

The system requires sufficient trend strength.

4. ADX Acceleration

ADX[current] > ADX[previous]

The strategy does not only require strong trend conditions; it requires trend strength to be increasing at the signal.

Execution

The signal is confirmed on a closed candle.

Entry occurs at:

OPEN of the next 1H candle

This prevents look-ahead bias caused by assuming execution at a price that was not available until the signal candle had already closed.

Exit

The current frozen exit is:

CLOSE after 24 x 1H candles

No fixed Stop Loss or Take Profit is used in the current research candidate.

Current Frozen Logic

IF
    Close > EMA200
AND EMA200 > EMA200[50]
AND Supertrend flips bearish → bullish
AND ADX(14) > 25
AND ADX is rising

THEN
    Enter BUY at next candle OPEN
    Hold for 24 x 1H candles
    Exit at the final candle CLOSE

Why These Components Work Together

Each filter is assigned a different responsibility.

Component

Role

Question

EMA200

Market regime

Is the broader environment bullish?

EMA200 slope

Structural confirmation

Is the long-term trend itself rising?

Supertrend

Trigger

Has direction recently changed bullish?

ADX > 25

Trend strength

Is the move strong enough to consider?

ADX Rising

Trend acceleration

Is trend strength increasing?

24H Time Exit

Trade management

Does the directional thesis develop over time?

The system is intentionally not designed as:

Indicator A says BUY
→ Buy immediately

Instead:

Regime
  +
Trigger
  +
Strength
  +
Acceleration
  +
Causal execution
  +
Consistent holding horizon

The components are intended to be complementary rather than redundant.

Research Evolution

C2G did not begin with a profitable strategy.

The development history contains multiple failed hypotheses, which are preserved because they are part of the research process.

Baseline

The early baseline used:

Supertrend 10 / 3
ADX > 20
BUY + SELL
SL = 1.5 ATR
TP = 3.0 ATR

Approximate historical result:

497 trades
Win Rate: 28.97%
Profit Factor: 0.803
PnL: -74.38%
Max Drawdown: approximately -60.84%

The baseline clearly did not demonstrate a usable edge.

ADX Threshold Experiments

The research tested increasing trend-strength requirements:

ADX > 20
ADX > 25
ADX > 30

Higher thresholds reduced trade frequency but did not automatically improve strategy quality.

Lesson

Fewer trades do not necessarily mean better trades.

ADX Rising

A more useful condition was:

ADX[current] > ADX[previous]

The purpose was to distinguish:

Strong but weakening trend

from:

Strong and strengthening trend

This materially improved several historical diagnostics.

Lesson

Trend-strength direction was more informative than simply increasing the ADX threshold.

+DI / -DI

Directional Index filters were tested as an additional confirmation layer.

They did not provide enough incremental improvement to justify becoming part of the current frozen core.

Lesson

A theoretically logical indicator must still earn its place through evidence.

Long-History Validation

Expanding the historical sample changed the interpretation of BUY versus SELL behavior.

In one long-history diagnostic:

BUY PF  ≈ 1.183
SELL PF ≈ 0.788

This contradicted conclusions from shorter recent samples.

Lesson

Strategy behavior can be regime-dependent, and conclusions from one historical window can be misleading.

Regime Discovery

A major research step was the introduction of explicit market regimes.

The regime logic used:

BULL:
Close > EMA200
AND EMA200 > EMA200[50]

BEAR:
Close < EMA200
AND EMA200 < EMA200[50]

NEUTRAL:
everything else

Historical diagnostics revealed strong directional asymmetry.

Examples from the research phase included approximately:

BUY in BULL regime  → PF ~1.77
SELL in BEAR regime → PF ~1.42
SELL in BULL regime → PF ~0.52

Lesson

Direction and market context cannot be treated independently.

Overfitting Warning from ATR Filtering

A later version introduced ATR Ratio filtering and produced visually impressive Binance results.

Examples:

Regime + ATR band
PF > 2.0 in historical Binance research

However, these filters were discovered using the same dataset on which performance was measured.

The strategy was therefore treated as an in-sample hypothesis, not a validated system.

That caution proved important.

Bybit Validation Failure

The same frozen rules were tested on Bybit BTCUSDT Perpetual.

Performance deteriorated significantly.

Example:

Historical Binance candidate:
PF > 1

Bybit:
PF < 1

One ATR-filtered candidate that looked especially strong on Binance dropped to roughly:

PF ~0.49 on Bybit before costs

Lesson

A strong result on a single feed is not sufficient evidence.

Instead of retuning the strategy immediately, C2G investigated the source of the discrepancy.

Cross-Market Signal Audit

Binance and Bybit were compared over the same BTC period.

Important finding:

Regime agreement ≈ 99.3%

The broader regime filter was highly consistent.

However, exact signal agreement was materially lower.

Some signal families showed only approximately:

45%–60% exact overlap

depending on direction and filter set.

Interpretation

The regime logic was relatively robust, while exact trigger conditions were sensitive to small differences in:

OHLC values;

indicator calculations;

threshold crossings.

Threshold Cliffs

The project identified a common systematic-trading problem:

ADX = 24.99 → signal rejected
ADX = 25.01 → signal accepted

Even though the economic difference is minimal, a hard threshold produces a binary decision.

This issue appeared in:

ADX thresholding;

one-bar ADX Rising;

ATR Ratio bands.

Lesson

Hard thresholds can make strategies feed-sensitive.

Entry Edge Diagnostic

A major turning point occurred when the project stopped asking only:

"Does the full trade make money?"

and instead asked:

"What happens after the signal?"

Post-signal returns were measured at:

6H
12H
24H
48H
72H

along with:

MFE — Maximum Favorable Excursion;

MAE — Maximum Adverse Excursion.

BUY Behavior

BUY signals demonstrated materially better forward directional behavior than SELL signals.

Examples from the robust BUY candidate included positive average forward returns across meaningful horizons on both Binance and Bybit.

SELL Behavior

SELL signals remained weak or negative in several forward-return diagnostics.

Decision

SELL = disabled

for the current frozen generation.

Exit Architecture Research

The previous execution model used:

SL = 1.5 ATR
TP = 3.0 ATR

However, MFE/MAE diagnostics showed many BUY events experiencing adverse movement before later moving strongly in the expected direction.

In one diagnostic, median BUY excursions were approximately:

MFE > 4 ATR
MAE ~2.5 ATR

This suggested that the fixed 1.5 ATR stop could terminate trades before the directional thesis had time to develop.

The project then compared a small, pre-defined exit set:

SL 1.5 / TP 3.0
SL 2.0 / TP 3.0
SL 2.0 / TP 2.0
SL 2.5 / TP 3.0
Time Exit 24H
Time Exit 48H

The 24H exit was the most robust cross-market candidate in the tested set.

This was the basis for the current frozen strategy.

Cross-Market Validation

The frozen BUY + 24H architecture was tested on BTC across:

Binance;

Bybit;

OKX.

Historical Research Result

Approximate same-period metrics:

Market

Gross PF

Cost-Stressed PF

Binance

2.002

1.707

Bybit

1.706

1.459

OKX

1.621

1.401

All three venues remained historically positive under the same frozen rule and the same cost assumptions.

Important limitation

These are different venues observing the same underlying BTC market over overlapping periods.

Therefore:

Cross-venue consistency is robustness evidence, but not three statistically independent out-of-sample tests.

Cross-Asset Validation

The exact same frozen rule was tested on Bybit perpetual markets without retuning:

BTC
ETH
SOL
XRP
BNB

Common-Period Cost-Stressed Results

Asset

Trades

PF

Expectancy

Result

BTC

20

1.412

+0.3366%

Positive

ETH

14

1.953

+1.2563%

Positive

SOL

13

1.553

+0.6291%

Positive

XRP

12

0.963

-0.1435%

Failed

BNB

12

0.382

-0.9944%

Failed

The strategy generalized to 3 of 5 tested crypto assets without asset-specific parameter changes.

Current Findings

BTC

Historically positive, but the recent common-period sample showed meaningful dependence on large winning trades.

Example robustness diagnostic:

Original PF: ~1.41
Remove top winner: PF falls to ~0.97

BTC remains an active candidate because it has also been studied across three exchanges.

ETH

ETH produced the strongest historical robustness among the tested assets.

Approximate common-period cost-stressed result:

Trades: 14
PF: 1.953
Expectancy: +1.2563%
Median trade: +1.0924%

Top-winner dependency remained better than BTC and SOL.

Leave-one-year-out testing also remained historically positive for every removed year.

Interpretation

ETH is currently the strongest cross-asset research candidate, but the sample is still small.

SOL

SOL remained historically profitable under the frozen rule:

PF: ~1.55
Expectancy: positive

However:

median trade was negative;

outlier dependency was stronger than ETH;

removing the largest winners materially weakened performance.

Interpretation

SOL behaves more like a trend-following distribution where a small number of strong moves may drive much of the edge.

XRP

XRP was approximately breakeven before costs and negative after costs.

Lesson

A strategy near zero gross expectancy may become clearly unprofitable after realistic execution assumptions.

BNB

BNB failed materially under the same frozen rule.

Lesson

The strategy should not be described as universally applicable to every crypto asset.

Repository Structure

Recommended current structure:

C2G System/
│
├── .git/
├── venv/
├── archive/
│   ├── code_history/
│   ├── results_history/
│   ├── old_loaders/
│   └── misc/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── btc_data_binance_full_1h.csv
├── btc_data_bybit_perp_1h.csv
├── btc_data_okx_perp_1h.csv
├── eth_data_bybit_perp_1h.csv
├── sol_data_bybit_perp_1h.csv
├── xrp_data_bybit_perp_1h.csv
├── bnb_data_bybit_perp_1h.csv
│
├── data_loader_binance_full_history.py
├── data_loader_bybit_btcusdt_perp_1h.py
├── data_loader_okx_btcusdt_swap_1h.py
├── data_loader_bybit_multiasset_perp_1h.py
│
├── c2g_engine_v114_true_forward_paper_logger.py
├── c2g_v114_forward_ledger.csv
├── c2g_v114_forward_summary.csv
├── c2g_v114_forward_manifest.txt
│
├── c2g_engine_v117_multiasset_true_forward_logger.py
├── c2g_v117_forward_ledger.csv
├── c2g_v117_forward_summary.csv
└── c2g_v117_forward_manifest.txt

Historical research code is intentionally retained rather than deleted.

The archive represents the development history of the project.

Installation

Requirements

Recommended:

Python 3.11+
Windows 10 / 11
PowerShell

Create a virtual environment:

python -m venv venv

Activate it:

.\venv\Scripts\Activate.ps1

Install core dependencies:

pip install pandas numpy pandas-ta ccxt requests

Depending on the local Python environment, the exact compatible pandas-ta package/version may vary.

Data Loaders

Binance Full BTC History

.\venv\Scripts\python.exe data_loader_binance_full_history.py

Expected output:

btc_data_binance_full_1h.csv

Bybit BTC Perpetual

.\venv\Scripts\python.exe data_loader_bybit_btcusdt_perp_1h.py

Expected output:

btc_data_bybit_perp_1h.csv

OKX BTC Perpetual

.\venv\Scripts\python.exe data_loader_okx_btcusdt_swap_1h.py

Expected output:

btc_data_okx_perp_1h.csv

Bybit Multi-Asset

.\venv\Scripts\python.exe data_loader_bybit_multiasset_perp_1h.py

Expected outputs include:

eth_data_bybit_perp_1h.csv
sol_data_bybit_perp_1h.csv
xrp_data_bybit_perp_1h.csv
bnb_data_bybit_perp_1h.csv

Running the Forward Tests

V1.14 — BTC Forward Logger

.\venv\Scripts\python.exe c2g_engine_v114_true_forward_paper_logger.py

This script does not send exchange orders.

It reconstructs only valid signals occurring after the configured freeze timestamp.

V1.17 — Multi-Asset Forward Logger

.\venv\Scripts\python.exe c2g_engine_v117_multiasset_true_forward_logger.py

Assets:

BTC
ETH
SOL

This version records:

post-freeze signals;

entry timestamp;

entry price;

open / closed status;

scheduled exit;

gross PnL;

cost-stressed PnL;

MFE;

MAE;

ADX at signal.

Research Methodology

C2G uses several robustness tools.

Same-Period Comparison

Different markets are compared over the same historical range whenever possible.

This avoids comparing one venue with a favorable bull period against another venue with a completely different time window.

Cost Stress

Current research cost assumptions:

Fee:     0.055% per side
Slippage: 0.020% per side

Approximate round-trip deduction:

0.15 percentage points per trade

Funding is not currently included.

Bootstrap

Trade returns are resampled with replacement to estimate uncertainty around:

expectancy;

Profit Factor;

probability expectancy > 0;

probability PF > 1.

Current research scripts have used:

20,000 bootstrap samples

Bootstrap results are treated as diagnostics, not guarantees.

Top-Winner Dependency

The strategy is re-evaluated after removing:

Top 1 winner
Top 2 winners
Top 3 winners

This tests whether performance is driven almost entirely by a few outliers.

This is especially important for trend-following systems.

Leave-One-Year-Out

Each calendar year is removed one at a time.

Example:

Remove 2022
Recalculate performance

Remove 2023
Recalculate performance
...

This checks whether the strategy depends disproportionately on a single favorable regime/year.

Cross-Exchange Signal Agreement

Signals generated by Binance, Bybit, and OKX are compared using exact or time-tolerant matching.

This allows the project to detect:

feed sensitivity;

threshold sensitivity;

indicator instability.

Metrics

The project tracks more than win rate.

Win Rate

Winning Trades / Total Trades

Useful but insufficient alone.

Profit Factor

Gross Profit / Gross Loss

General interpretation:

PF < 1.0  → historical losing system
PF = 1.0  → historical breakeven
PF > 1.0  → historical positive

Higher PF with extremely low sample size should be treated cautiously.

Expectancy

Average percentage return per trade.

Expectancy = mean(trade returns)

Drawdown

Research scripts build a compounded equity series beginning at 100 and calculate peak-to-trough decline.

Important:

Current research drawdown is not the same as a fully realistic leveraged portfolio drawdown.

MFE

Maximum Favorable Excursion

Maximum movement in the trade direction while the trade is active.

MAE

Maximum Adverse Excursion

Maximum movement against the trade while the trade is active.

MFE/MAE analysis was central to identifying that fixed ATR stop logic could conflict with the directional behavior of BUY signals.

Forward Testing

Forward testing is the current priority.

The central rule is:

Do not change the frozen strategy because of early forward results.

If the first forward trades lose:

Do not retune.

If the first forward trades win:

Do not retune.

The purpose is to build data that did not exist when the strategy was selected.

Forward evidence is more valuable than another optimized historical backtest.

Planned MQL5 Architecture

The current research engine is Python-based.

The future MetaTrader 5 implementation is expected to separate:

Signal Engine
    ↓
Indicator Layer
    ↓
Alert Manager
    ↓
Risk Engine
    ↓
Execution Engine
    ↓
Position Manager

Planned Indicator Components

Future research may evaluate:

EMA 9 / 21 / 50;

ADX + DI;

MACD;

RSI;

ATR;

optional volume.

These are not currently considered validated parts of the frozen C2G V1 core.

Any future component must prove incremental value before being promoted into the production indicator.

Planned MQL5 Implementation Concepts

Expected technical components include:

OnInit()
OnCalculate()
OnDeinit()
native indicator handles
CopyBuffer()
indicator buffers
plot buffers
alert state management
new-bar / closed-bar confirmation
duplicate-alert protection

Potential native handles:

iMA()
iADX()
iMACD()
iRSI()
iATR()

The production indicator should avoid recreating handles during every calculation cycle.

Future Indicator Design Principle

The final product should not become a collection of unrelated indicators.

Each component must have a defined role.

Possible future architecture:

Layer

Candidate

Structural regime

EMA200

Local trend

EMA 9 / 21 / 50

Trend trigger

Supertrend

Strength

ADX

Momentum

MACD

Exhaustion

RSI

Volatility

ATR

Participation

Volume

But:

No component should be added only because it improves one historical backtest.

Limitations

C2G has important limitations.

Small Samples

Current strategy frequency is low.

Several historical evaluations contain fewer than 30 trades per asset.

Correlated Assets

BTC, ETH, and SOL are not statistically independent markets.

A pooled sample across these assets should not be interpreted as fully independent evidence.

Historical Selection Bias

BTC, ETH, and SOL were selected for multi-asset forward observation after historical research.

Only future post-freeze signals should be treated as new forward evidence for this selected set.

No Funding Model

Perpetual funding is not currently included.

Simplified Execution

Research assumes deterministic execution based on OHLC data.

Live execution may differ due to:

spread;

order-book depth;

latency;

partial fills;

exchange outages;

price gaps;

execution priority.

No Position-Sizing Engine Yet

Historical return metrics are based primarily on trade percentage returns.

A complete live portfolio model still requires:

risk per trade;

capital allocation;

simultaneous-position handling;

leverage constraints;

correlated exposure limits.

No Live Trading Yet

The current forward loggers are:

PAPER / RESEARCH ONLY

They do not send live orders.

Roadmap

Phase 1 — Historical Research

Status: Completed for C2G V1 core

Includes:

baseline;

ADX experiments;

regime analysis;

cross-market validation;

signal timing diagnostics;

entry-edge diagnostics;

exit architecture;

multi-exchange validation;

multi-asset validation.

Phase 2 — Forward Testing

Status: Active

Targets:

BTC V1.14;

BTC / ETH / SOL V1.17.

Phase 3 — C2G V2 Research

Future research must remain separated from the frozen V1 forward branch.

Potential research questions:

ADX threshold sensitivity
multi-timeframe context
EMA 9/21/50 incremental value
MACD incremental value
RSI exhaustion filtering
volatility-state modeling
dynamic exit research
portfolio exposure management

Each experiment should modify as few variables as possible.

Phase 4 — MQL5 Indicator

Planned:

production signal engine;

visual BUY / SELL markers;

regime state;

alert system;

configurable parameters;

documentation.

Phase 5 — Expert Advisor

Only after signal and risk logic are sufficiently validated.

Planned components:

execution engine;

position sizing;

portfolio limits;

spread/slippage protection;

logging;

fail-safe handling.

Versioning Policy

Historical versions are never silently overwritten.

Example research lineage:

V1.1  → corrected baseline
V1.2  → ADX experiments
V1.3  → long-history validation
V1.4  → regime diagnostics
V1.5  → regime-filter hypotheses
V1.6  → Bybit validation
V1.7  → cross-market audit
V1.8  → timing / threshold audit
V1.9  → robust trigger research
V1.10 → entry-edge diagnostic
V1.11 → exit-path diagnostic
V1.12 → frozen BUY24H robustness
V1.13 → third-venue validation
V1.14 → BTC true forward logger
V1.15 → cross-asset validation
V1.16 → cross-asset robustness
V1.17 → BTC / ETH / SOL forward logger

Completed historical code is moved to:

archive/code_history/

Historical results are moved to:

archive/results_history/

The archive is part of the project's research history.

Development Principles

C2G follows several rules:

1. Do not optimize away a failed validation.
2. Do not remove bad years after seeing the results.
3. Do not select only the highest PF.
4. Do not confuse historical fit with forward evidence.
5. Do not add indicators without incremental justification.
6. Preserve failed experiments.
7. Include realistic costs.
8. Prefer reproducible rules over subjective chart interpretation.
9. Separate research versions from frozen forward versions.
10. Treat small samples with caution.

Build in Public

From Code to Gold is also a public learning project.

The development narrative is intentionally transparent:

Build
Test
Fail
Learn
Improve

Failures are not removed from the story.

A failed external validation, fragile threshold, negative asset, or weak strategy version is treated as useful information.

The goal is not to present a perfect historical equity curve.

The goal is to document how a systematic trading tool is engineered, tested, challenged, and improved.

Disclaimer

This repository is for software engineering, educational, and research purposes only.

Nothing in this repository constitutes:

financial advice;

investment advice;

a recommendation to buy or sell any asset;

a promise of future profitability.

Historical backtests do not guarantee future results.

Cryptocurrency and leveraged derivatives involve substantial risk.

The authors and contributors are not responsible for financial losses resulting from the use of this software or research.

License

See:

LICENSE

for the repository's licensing terms.

Author

From Code to Gold

Software engineering applied to systematic trading research.

Build. Test. Fail. Learn. Improve.# C2G-System
