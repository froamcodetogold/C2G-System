import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.2 EXPERIMENTOS
#
# Testa automaticamente:
# 1) ADX > 20
# 2) ADX > 25
# 3) ADX > 30
# 4) ADX > 25 + ADX subindo
# 5) ADX > 25 + ADX subindo + confirmação +DI/-DI
#
# Mantém fixos:
# Supertrend 10/3
# Stop = 1.5 ATR
# Take Profit = 3.0 ATR
#
# NÃO usa EMA200 nesta rodada.
# ============================================================


# ============================================================
# AUXILIARES
# ============================================================

def find_column(columns, prefix):
    matches = [col for col in columns if str(col).startswith(prefix)]
    if not matches:
        raise KeyError(
            f"Nenhuma coluna começando com '{prefix}' foi encontrada.\n"
            f"Colunas disponíveis: {list(columns)}"
        )
    return matches[0]


def calculate_pnl(side, entry_price, exit_price, fee_pct=0.0, slippage_pct=0.0):
    if side == "BUY":
        gross_pct = ((exit_price - entry_price) / entry_price) * 100.0
    else:
        gross_pct = ((entry_price - exit_price) / entry_price) * 100.0

    total_cost_pct = (fee_pct * 2.0) + (slippage_pct * 2.0)
    return gross_pct - total_cost_pct


def max_consecutive_losses(pnl_values):
    best = 0
    current = 0

    for pnl in pnl_values:
        if pnl < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def build_equity_curve(pnl_values, initial_equity=100.0):
    equity = [initial_equity]

    for pnl_pct in pnl_values:
        equity.append(
            equity[-1] * (1.0 + pnl_pct / 100.0)
        )

    return pd.Series(equity, dtype=float)


def calculate_max_drawdown(equity_curve):
    if equity_curve.empty:
        return 0.0

    peak = equity_curve.cummax()
    dd = ((equity_curve / peak) - 1.0) * 100.0

    return float(dd.min())


def calculate_metrics(trades_df):
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_loss_streak": 0,
            "max_drawdown": 0.0,
            "ending_equity": 100.0,
        }

    winners = trades_df[trades_df["pnl_pct"] > 0]
    losers = trades_df[trades_df["pnl_pct"] < 0]

    trades = len(trades_df)
    wins = len(winners)
    losses = len(losers)

    win_rate = (wins / trades) * 100.0
    pnl = float(trades_df["pnl_pct"].sum())
    expectancy = float(trades_df["pnl_pct"].mean())

    avg_win = float(winners["pnl_pct"].mean()) if wins else 0.0
    avg_loss = float(losers["pnl_pct"].mean()) if losses else 0.0

    gross_profit = float(winners["pnl_pct"].sum()) if wins else 0.0
    gross_loss = abs(float(losers["pnl_pct"].sum())) if losses else 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    max_loss_streak = max_consecutive_losses(
        trades_df["pnl_pct"].tolist()
    )

    equity = build_equity_curve(
        trades_df["pnl_pct"].tolist(),
        initial_equity=100.0
    )

    max_drawdown = calculate_max_drawdown(equity)
    ending_equity = float(equity.iloc[-1])

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "pnl": pnl,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_loss_streak": max_loss_streak,
        "max_drawdown": max_drawdown,
        "ending_equity": ending_equity,
    }


# ============================================================
# CARREGAR DADOS E CALCULAR INDICADORES UMA ÚNICA VEZ
# ============================================================

def prepare_data(
    file_path="btc_data.csv",
    supertrend_length=10,
    supertrend_multiplier=3.0,
    adx_length=14,
    atr_length=14,
):
    print("\nCarregando dados...")

    df = pd.read_csv(
        file_path,
        index_col="timestamp",
        parse_dates=True
    )

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    required = ["open", "high", "low", "close"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        )

    df = df.dropna(subset=required)

    print(f"Candles carregados: {len(df)}")

    # ---------------- SUPERTREND ----------------
    st = ta.supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=supertrend_length,
        multiplier=supertrend_multiplier
    )

    if st is None:
        raise RuntimeError("Falha ao calcular Supertrend.")

    st_dir_col = find_column(st.columns, "SUPERTd_")
    st_line_col = find_column(st.columns, "SUPERT_")

    df["Trend_Direction"] = st[st_dir_col]
    df["Trend_Line"] = st[st_line_col]
    df["Signal"] = df["Trend_Direction"].diff()

    # ---------------- ADX / DI ----------------
    adx = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=adx_length
    )

    if adx is None:
        raise RuntimeError("Falha ao calcular ADX.")

    adx_col = find_column(adx.columns, "ADX_")
    dmp_col = find_column(adx.columns, "DMP_")
    dmn_col = find_column(adx.columns, "DMN_")

    df["ADX"] = adx[adx_col]
    df["PLUS_DI"] = adx[dmp_col]
    df["MINUS_DI"] = adx[dmn_col]

    # ADX atual maior que o ADX do candle anterior.
    df["ADX_Rising"] = (
        df["ADX"] > df["ADX"].shift(1)
    )

    # ---------------- ATR ----------------
    df["ATR"] = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=atr_length
    )

    return df


# ============================================================
# GERAR SINAIS PARA UM EXPERIMENTO
# ============================================================

def build_signals(
    df,
    adx_min=20.0,
    use_adx_rising=False,
    use_di_filter=False,
):
    work = df.copy()

    buy = (
        (work["Signal"] == 2)
        &
        (work["ADX"] > adx_min)
    )

    sell = (
        (work["Signal"] == -2)
        &
        (work["ADX"] > adx_min)
    )

    if use_adx_rising:
        buy = buy & work["ADX_Rising"]
        sell = sell & work["ADX_Rising"]

    if use_di_filter:
        buy = (
            buy
            &
            (work["PLUS_DI"] > work["MINUS_DI"])
        )

        sell = (
            sell
            &
            (work["MINUS_DI"] > work["PLUS_DI"])
        )

    work["Buy_Signal"] = buy.astype(int)
    work["Sell_Signal"] = sell.astype(int)

    return work


# ============================================================
# MOTOR DE BACKTEST
# ============================================================

def run_backtest(
    df,
    stop_atr=1.5,
    target_atr=3.0,
    allow_shorts=True,
    fee_pct=0.0,
    slippage_pct=0.0,
):
    trades = []

    position = None
    entry_price = None
    entry_time = None
    stop_loss = None
    take_profit = None

    for i in range(1, len(df)):
        previous = df.iloc[i - 1]
        current = df.iloc[i]

        current_open = float(current["open"])
        current_high = float(current["high"])
        current_low = float(current["low"])
        current_close = float(current["close"])

        # ----------------------------------------------------
        # 1. ENTRADA
        # sinal confirmado no candle anterior;
        # entrada no OPEN do candle atual.
        # ----------------------------------------------------
        if position is None:
            previous_atr = previous["ATR"]

            if pd.notna(previous_atr) and previous_atr > 0:

                if previous["Buy_Signal"] == 1:
                    position = "BUY"
                    entry_time = df.index[i]
                    entry_price = current_open

                    stop_loss = (
                        entry_price
                        -
                        (stop_atr * previous_atr)
                    )

                    take_profit = (
                        entry_price
                        +
                        (target_atr * previous_atr)
                    )

                elif (
                    allow_shorts
                    and previous["Sell_Signal"] == 1
                ):
                    position = "SELL"
                    entry_time = df.index[i]
                    entry_price = current_open

                    stop_loss = (
                        entry_price
                        +
                        (stop_atr * previous_atr)
                    )

                    take_profit = (
                        entry_price
                        -
                        (target_atr * previous_atr)
                    )

        # ----------------------------------------------------
        # 2. GERENCIAR BUY
        # inclusive no candle da entrada
        # ----------------------------------------------------
        if position == "BUY":
            exit_price = None
            reason = None

            stop_hit = current_low <= stop_loss
            target_hit = current_high >= take_profit

            # Sem dados intrabar, usa hipótese conservadora:
            # se SL e TP ocorrerem no mesmo candle, STOP primeiro.
            if stop_hit:
                exit_price = stop_loss
                reason = "STOP"

            elif target_hit:
                exit_price = take_profit
                reason = "TAKE_PROFIT"

            elif current["Trend_Direction"] == -1:
                exit_price = current_close
                reason = "TREND_REVERSAL"

            if exit_price is not None:
                pnl = calculate_pnl(
                    "BUY",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct
                )

                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": pnl
                })

                position = None

        # ----------------------------------------------------
        # 3. GERENCIAR SELL
        # ----------------------------------------------------
        elif position == "SELL":
            exit_price = None
            reason = None

            stop_hit = current_high >= stop_loss
            target_hit = current_low <= take_profit

            if stop_hit:
                exit_price = stop_loss
                reason = "STOP"

            elif target_hit:
                exit_price = take_profit
                reason = "TAKE_PROFIT"

            elif current["Trend_Direction"] == 1:
                exit_price = current_close
                reason = "TREND_REVERSAL"

            if exit_price is not None:
                pnl = calculate_pnl(
                    "SELL",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct
                )

                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": pnl
                })

                position = None

    # Fecha posição restante no fim da amostra.
    if position is not None:
        final_price = float(df["close"].iloc[-1])

        pnl = calculate_pnl(
            position,
            entry_price,
            final_price,
            fee_pct,
            slippage_pct
        )

        trades.append({
            "side": position,
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "entry_price": entry_price,
            "exit_price": final_price,
            "reason": "END_OF_DATA",
            "pnl_pct": pnl
        })

    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(trades_df)

    return trades_df, metrics


# ============================================================
# RELATÓRIO DETALHADO
# ============================================================

def print_detailed_report(name, trades_df, metrics):
    pf_text = (
        "inf"
        if np.isinf(metrics["profit_factor"])
        else f"{metrics['profit_factor']:.3f}"
    )

    print("\n" + "=" * 82)
    print(name)
    print("=" * 82)
    print(f"Trades:                    {metrics['trades']}")
    print(f"Vitórias:                  {metrics['wins']}")
    print(f"Derrotas:                  {metrics['losses']}")
    print(f"Win Rate:                  {metrics['win_rate']:.2f}%")
    print(f"PnL simples:               {metrics['pnl']:.2f}%")
    print(f"Profit Factor:             {pf_text}")
    print(f"Expectancy / trade:        {metrics['expectancy']:.4f}%")
    print(f"Média ganho:               {metrics['avg_win']:.4f}%")
    print(f"Média perda:               {metrics['avg_loss']:.4f}%")
    print(f"Max Drawdown comparativo:  {metrics['max_drawdown']:.2f}%")
    print(f"Maior sequência perdas:    {metrics['max_loss_streak']}")
    print(f"Banca-base 100 final:      {metrics['ending_equity']:.2f}")

    if not trades_df.empty:
        for side in ["BUY", "SELL"]:
            group = trades_df[trades_df["side"] == side]

            if len(group):
                m = calculate_metrics(group)

                side_pf = (
                    "inf"
                    if np.isinf(m["profit_factor"])
                    else f"{m['profit_factor']:.3f}"
                )

                print(
                    f"{side:>4}: "
                    f"Trades={m['trades']} | "
                    f"WR={m['win_rate']:.2f}% | "
                    f"PnL={m['pnl']:.2f}% | "
                    f"PF={side_pf}"
                )


# ============================================================
# GRÁFICO DA MELHOR VERSÃO
# ============================================================

def plot_best(df, trades_df, metrics, strategy_name, bars=200):
    plot_df = df.tail(bars).copy()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }

    if "volume" in plot_df.columns:
        rename_map["volume"] = "Volume"

    plot_df = plot_df.rename(columns=rename_map)

    apds = [
        mpf.make_addplot(
            plot_df["Trend_Line"],
            color="cyan",
            width=1.8,
            secondary_y=False
        )
    ]

    buy_points = np.where(
        plot_df["Buy_Signal"] == 1,
        plot_df["Low"] * 0.998,
        np.nan
    )

    sell_points = np.where(
        plot_df["Sell_Signal"] == 1,
        plot_df["High"] * 1.002,
        np.nan
    )

    if not np.isnan(buy_points).all():
        apds.append(
            mpf.make_addplot(
                buy_points,
                type="scatter",
                markersize=100,
                marker="^",
                color="dodgerblue"
            )
        )

    if not np.isnan(sell_points).all():
        apds.append(
            mpf.make_addplot(
                sell_points,
                type="scatter",
                markersize=100,
                marker="v",
                color="gold"
            )
        )

    mc = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        wick="inherit",
        edge="inherit"
    )

    style = mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style="nightclouds"
    )

    has_volume = "Volume" in plot_df.columns

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        addplot=apds,
        title=f"\nC2G System Pro - Melhor V1.2 - {strategy_name}",
        ylabel="Price (USDT)",
        volume=has_volume,
        figratio=(16, 9),
        figscale=1.1,
        returnfig=True
    )

    pf_text = (
        "inf"
        if np.isinf(metrics["profit_factor"])
        else f"{metrics['profit_factor']:.2f}"
    )

    info = (
        f"WR: {metrics['win_rate']:.2f}% | "
        f"Trades: {metrics['trades']} | "
        f"PF: {pf_text} | "
        f"PnL: {metrics['pnl']:.2f}% | "
        f"DD: {metrics['max_drawdown']:.2f}%"
    )

    fig.text(
        0.13,
        0.93,
        info,
        color="white",
        fontsize=10,
        bbox=dict(
            facecolor="black",
            alpha=0.7,
            edgecolor="white",
            boxstyle="round,pad=0.4"
        )
    )

    mpf.show()


# ============================================================
# EXECUÇÃO DOS 5 EXPERIMENTOS
# ============================================================

if __name__ == "__main__":

    BASE_DF = prepare_data(
        file_path="btc_data.csv",
        supertrend_length=10,
        supertrend_multiplier=3.0,
        adx_length=14,
        atr_length=14
    )

    EXPERIMENTS = [
        {
            "name": "V1.0 - ADX20",
            "adx_min": 20.0,
            "use_adx_rising": False,
            "use_di_filter": False,
        },
        {
            "name": "V1.2A - ADX25",
            "adx_min": 25.0,
            "use_adx_rising": False,
            "use_di_filter": False,
        },
        {
            "name": "V1.2B - ADX30",
            "adx_min": 30.0,
            "use_adx_rising": False,
            "use_di_filter": False,
        },
        {
            "name": "V1.2C - ADX25 + Rising",
            "adx_min": 25.0,
            "use_adx_rising": True,
            "use_di_filter": False,
        },
        {
            "name": "V1.2D - ADX25 + Rising + DI",
            "adx_min": 25.0,
            "use_adx_rising": True,
            "use_di_filter": True,
        },
    ]

    results = []
    saved_runs = {}

    for exp in EXPERIMENTS:
        test_df = build_signals(
            BASE_DF,
            adx_min=exp["adx_min"],
            use_adx_rising=exp["use_adx_rising"],
            use_di_filter=exp["use_di_filter"],
        )

        trades_df, metrics = run_backtest(
            test_df,

            # Mantemos o mesmo risco da V1.0
            stop_atr=1.5,
            target_atr=3.0,

            allow_shorts=True,

            # Custos ainda em zero para comparar somente a lógica.
            fee_pct=0.0,
            slippage_pct=0.0,
        )

        print_detailed_report(
            exp["name"],
            trades_df,
            metrics
        )

        row = {
            "strategy": exp["name"],
            "adx_min": exp["adx_min"],
            "adx_rising": exp["use_adx_rising"],
            "di_filter": exp["use_di_filter"],
            **metrics
        }

        results.append(row)

        saved_runs[exp["name"]] = {
            "df": test_df,
            "trades": trades_df,
            "metrics": metrics
        }

    # ========================================================
    # RANKING
    # ========================================================

    ranking = pd.DataFrame(results)

    ranking = ranking.sort_values(
        by=[
            "profit_factor",
            "expectancy",
        ],
        ascending=[
            False,
            False,
        ]
    ).reset_index(drop=True)

    print("\n")
    print("=" * 116)
    print("RANKING FINAL C2G SYSTEM V1.2 - ORDENADO POR PROFIT FACTOR")
    print("=" * 116)

    display_cols = [
        "strategy",
        "trades",
        "win_rate",
        "pnl",
        "profit_factor",
        "expectancy",
        "max_drawdown",
        "max_loss_streak",
    ]

    print(
        ranking[
            display_cols
        ].to_string(
            index=False,
            formatters={
                "win_rate": "{:.2f}%".format,
                "pnl": "{:.2f}%".format,
                "profit_factor": "{:.3f}".format,
                "expectancy": "{:.4f}%".format,
                "max_drawdown": "{:.2f}%".format,
            }
        )
    )

    print("=" * 116)

    # Salva ranking.
    ranking.to_csv(
        "c2g_v12_ranking.csv",
        index=False
    )

    # Melhor estratégia.
    best_name = ranking.iloc[0]["strategy"]

    best_run = saved_runs[best_name]

    best_run["trades"].to_csv(
        "c2g_v12_best_trades.csv",
        index=False
    )

    print(f"\nMelhor estratégia por Profit Factor: {best_name}")
    print("Ranking salvo em: c2g_v12_ranking.csv")
    print("Trades da melhor versão salvos em: c2g_v12_best_trades.csv")

    # Gráfico da campeã.
    plot_best(
        best_run["df"],
        best_run["trades"],
        best_run["metrics"],
        best_name,
        bars=200
    )
    