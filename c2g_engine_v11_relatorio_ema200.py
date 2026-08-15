import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import numpy as np


# ============================================================
# C2G SYSTEM PRO - BACKTEST V1.1
#
# Novidades:
# 1) Relatório completo:
#    - Win Rate
#    - Profit Factor
#    - Expectancy
#    - Média dos ganhos/perdas
#    - Maior sequência de perdas
#    - Drawdown máximo
#    - BUY x SELL
#    - Resultado por ano
#    - Resultado por motivo de saída
#
# 2) Primeiro filtro experimental:
#    - EMA 200 como filtro de tendência maior
#
# 3) Backtest mais rigoroso:
#    - Sinal no fechamento anterior
#    - Entrada no OPEN seguinte
#    - SL/TP já podem ser atingidos no candle de entrada
#    - Se SL e TP forem atingidos no mesmo candle, assume STOP primeiro
#
# IMPORTANTE:
# O PnL deste backtest é retorno percentual da posição sem modelar
# alavancagem/tamanho da banca. Use-o para comparar versões.
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


def calculate_pnl(
    side,
    entry_price,
    exit_price,
    fee_pct=0.0,
    slippage_pct=0.0
):
    if side == "BUY":
        gross_pct = ((exit_price - entry_price) / entry_price) * 100.0
    else:
        gross_pct = ((entry_price - exit_price) / entry_price) * 100.0

    # taxa + slippage na entrada e na saída
    total_cost_pct = (fee_pct * 2.0) + (slippage_pct * 2.0)

    return gross_pct - total_cost_pct


def max_consecutive_losses(pnl_series):
    max_streak = 0
    current_streak = 0

    for pnl in pnl_series:
        if pnl < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def build_equity_curve(pnl_series, initial_equity=100.0):
    """
    Curva comparativa:
    aplica cada retorno percentual sequencialmente sobre uma banca-base 100.

    Não representa ainda position sizing por risco fixo.
    Serve para comparar versões da estratégia.
    """
    equity = [initial_equity]

    for pnl_pct in pnl_series:
        next_equity = equity[-1] * (1.0 + pnl_pct / 100.0)
        equity.append(next_equity)

    return pd.Series(equity, dtype=float)


def calculate_max_drawdown(equity_curve):
    if equity_curve.empty:
        return 0.0

    running_peak = equity_curve.cummax()
    drawdown = ((equity_curve / running_peak) - 1.0) * 100.0

    return float(drawdown.min())


def calculate_metrics(trades_df):
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_loss_streak": 0,
            "max_drawdown": 0.0,
            "ending_equity": 100.0,
        }

    winners = trades_df[trades_df["pnl_pct"] > 0]
    losers = trades_df[trades_df["pnl_pct"] < 0]

    total_trades = len(trades_df)
    wins = len(winners)
    losses = len(losers)

    win_rate = (wins / total_trades) * 100.0

    total_pnl = float(trades_df["pnl_pct"].sum())
    avg_trade = float(trades_df["pnl_pct"].mean())

    avg_win = float(winners["pnl_pct"].mean()) if wins else 0.0
    avg_loss = float(losers["pnl_pct"].mean()) if losses else 0.0

    gross_profit = float(winners["pnl_pct"].sum()) if wins else 0.0
    gross_loss = abs(float(losers["pnl_pct"].sum())) if losses else 0.0

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    # expectancy percentual médio por trade
    expectancy = avg_trade

    max_loss_streak = max_consecutive_losses(
        trades_df["pnl_pct"].tolist()
    )

    equity_curve = build_equity_curve(
        trades_df["pnl_pct"].tolist(),
        initial_equity=100.0
    )

    max_drawdown = calculate_max_drawdown(equity_curve)
    ending_equity = float(equity_curve.iloc[-1])

    return {
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_trade": avg_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_loss_streak": max_loss_streak,
        "max_drawdown": max_drawdown,
        "ending_equity": ending_equity,
    }


def print_group_report(trades_df, group_col, title):
    print("\n" + "-" * 76)
    print(title)
    print("-" * 76)

    if trades_df.empty:
        print("Sem trades.")
        return

    for group_name, group in trades_df.groupby(group_col):
        metrics = calculate_metrics(group)

        pf_text = (
            "inf"
            if np.isinf(metrics["profit_factor"])
            else f"{metrics['profit_factor']:.3f}"
        )

        print(
            f"{str(group_name):>16} | "
            f"Trades {metrics['trades']:>4} | "
            f"WR {metrics['win_rate']:>6.2f}% | "
            f"PnL {metrics['total_pnl']:>8.2f}% | "
            f"PF {pf_text:>6}"
        )


# ============================================================
# BACKTEST
# ============================================================

def run_c2g_backtest(
    file_path="btc_data.csv",

    # Supertrend
    supertrend_length=10,
    supertrend_multiplier=3.0,

    # ADX
    adx_length=14,
    adx_min=20.0,

    # ATR
    atr_length=14,
    stop_atr=1.5,
    target_atr=3.0,

    # Filtro de tendência maior
    use_ema_filter=False,
    ema_filter_length=200,

    # Operações
    allow_shorts=True,

    # Custos
    fee_pct=0.0,
    slippage_pct=0.0,

    # Nome do teste
    test_name="BASELINE",
):
    print("\n")
    print("=" * 76)
    print(f"C2G SYSTEM PRO - {test_name}")
    print("=" * 76)
    print("Carregando dados...")

    # ========================================================
    # 1. CARREGAR CSV
    # ========================================================

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

    required_columns = ["open", "high", "low", "close"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df = df.dropna(subset=required_columns)

    print(f"Candles carregados: {len(df)}")

    # ========================================================
    # 2. SUPERTREND
    # ========================================================

    supertrend = ta.supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=supertrend_length,
        multiplier=supertrend_multiplier
    )

    if supertrend is None:
        raise RuntimeError("Não foi possível calcular o Supertrend.")

    trend_direction_col = find_column(
        supertrend.columns,
        "SUPERTd_"
    )

    trend_line_col = find_column(
        supertrend.columns,
        "SUPERT_"
    )

    df["Trend_Direction"] = supertrend[trend_direction_col]
    df["Trend_Line"] = supertrend[trend_line_col]

    # ========================================================
    # 3. ADX
    # ========================================================

    adx_df = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=adx_length
    )

    if adx_df is None:
        raise RuntimeError("Não foi possível calcular o ADX.")

    adx_col = find_column(adx_df.columns, "ADX_")
    df["ADX"] = adx_df[adx_col]

    # ========================================================
    # 4. ATR
    # ========================================================

    df["ATR"] = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=atr_length
    )

    # ========================================================
    # 5. EMA DE TENDÊNCIA MAIOR
    # ========================================================

    df["EMA_Filter"] = ta.ema(
        df["close"],
        length=ema_filter_length
    )

    # ========================================================
    # 6. SINAIS
    # ========================================================

    df["Signal"] = df["Trend_Direction"].diff()

    buy_condition = (
        (df["Signal"] == 2)
        &
        (df["ADX"] > adx_min)
    )

    sell_condition = (
        (df["Signal"] == -2)
        &
        (df["ADX"] > adx_min)
    )

    # PRIMEIRA MUDANÇA EXPERIMENTAL:
    # BUY somente acima da EMA 200
    # SELL somente abaixo da EMA 200
    if use_ema_filter:
        buy_condition = (
            buy_condition
            &
            (df["close"] > df["EMA_Filter"])
        )

        sell_condition = (
            sell_condition
            &
            (df["close"] < df["EMA_Filter"])
        )

    df["Buy_Signal"] = buy_condition.astype(int)
    df["Sell_Signal"] = sell_condition.astype(int)

    print(f"BUY encontrados:  {int(df['Buy_Signal'].sum())}")
    print(f"SELL encontrados: {int(df['Sell_Signal'].sum())}")

    # ========================================================
    # 7. MOTOR DE BACKTEST
    # ========================================================

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
        # A) Se não há posição, entra no OPEN do candle atual
        #    usando apenas o sinal confirmado no candle anterior.
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
        # B) Gerencia inclusive o candle de entrada.
        # ----------------------------------------------------
        if position == "BUY":
            exit_price = None
            exit_reason = None

            stop_hit = current_low <= stop_loss
            target_hit = current_high >= take_profit

            if stop_hit:
                exit_price = stop_loss
                exit_reason = "STOP"

            elif target_hit:
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT"

            elif current["Trend_Direction"] == -1:
                exit_price = current_close
                exit_reason = "TREND_REVERSAL"

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
                    "reason": exit_reason,
                    "pnl_pct": pnl
                })

                position = None

        elif position == "SELL":
            exit_price = None
            exit_reason = None

            stop_hit = current_high >= stop_loss
            target_hit = current_low <= take_profit

            if stop_hit:
                exit_price = stop_loss
                exit_reason = "STOP"

            elif target_hit:
                exit_price = take_profit
                exit_reason = "TAKE_PROFIT"

            elif current["Trend_Direction"] == 1:
                exit_price = current_close
                exit_reason = "TREND_REVERSAL"

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
                    "reason": exit_reason,
                    "pnl_pct": pnl
                })

                position = None

    # ========================================================
    # 8. FECHAR POSIÇÃO NO FIM DO HISTÓRICO
    # ========================================================

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

    # ========================================================
    # 9. RELATÓRIO COMPLETO
    # ========================================================

    metrics = calculate_metrics(trades_df)

    pf_text = (
        "infinito"
        if np.isinf(metrics["profit_factor"])
        else f"{metrics['profit_factor']:.3f}"
    )

    print("\n" + "=" * 76)
    print(f"RELATÓRIO - {test_name}")
    print("=" * 76)
    print(f"Total de candles:             {len(df)}")
    print(f"Total de trades:              {metrics['trades']}")
    print(f"Vitórias:                     {metrics['wins']}")
    print(f"Derrotas:                     {metrics['losses']}")
    print(f"Win Rate:                     {metrics['win_rate']:.2f}%")
    print(f"PnL acumulado simples:        {metrics['total_pnl']:.4f}%")
    print(f"Média por trade / Expectancy: {metrics['expectancy']:.4f}%")
    print(f"Média dos ganhos:             {metrics['avg_win']:.4f}%")
    print(f"Média das perdas:             {metrics['avg_loss']:.4f}%")
    print(f"Profit Factor:                {pf_text}")
    print(f"Maior sequência de perdas:    {metrics['max_loss_streak']}")
    print(f"Max Drawdown comparativo:     {metrics['max_drawdown']:.2f}%")
    print(f"Banca-base 100 terminou em:   {metrics['ending_equity']:.2f}")
    print(f"ADX mínimo:                   {adx_min}")
    print(f"EMA maior ativada:            {use_ema_filter}")
    print(f"EMA maior:                    {ema_filter_length}")
    print(f"Stop:                         {stop_atr} ATR")
    print(f"Take Profit:                  {target_atr} ATR")
    print(f"Taxa por lado:                {fee_pct:.4f}%")
    print(f"Slippage por lado:            {slippage_pct:.4f}%")
    print("=" * 76)

    if not trades_df.empty:
        trades_df["entry_year"] = pd.to_datetime(
            trades_df["entry_time"]
        ).dt.year

        print_group_report(
            trades_df,
            "side",
            "RESULTADO BUY x SELL"
        )

        print_group_report(
            trades_df,
            "entry_year",
            "RESULTADO POR ANO"
        )

        print_group_report(
            trades_df,
            "reason",
            "RESULTADO POR MOTIVO DE SAÍDA"
        )

    return df, trades_df, metrics


# ============================================================
# COMPARAÇÃO AUTOMÁTICA
# ============================================================

def print_comparison(baseline_metrics, filtered_metrics):
    print("\n")
    print("=" * 88)
    print("COMPARAÇÃO DIRETA - BASELINE x EMA200")
    print("=" * 88)

    rows = [
        ("Trades", baseline_metrics["trades"], filtered_metrics["trades"]),
        ("Win Rate", baseline_metrics["win_rate"], filtered_metrics["win_rate"]),
        ("PnL simples", baseline_metrics["total_pnl"], filtered_metrics["total_pnl"]),
        ("Profit Factor", baseline_metrics["profit_factor"], filtered_metrics["profit_factor"]),
        ("Expectancy", baseline_metrics["expectancy"], filtered_metrics["expectancy"]),
        ("Max Drawdown", baseline_metrics["max_drawdown"], filtered_metrics["max_drawdown"]),
        ("Loss Streak", baseline_metrics["max_loss_streak"], filtered_metrics["max_loss_streak"]),
    ]

    for name, base, filtered in rows:
        if isinstance(base, (float, np.floating)):
            base_text = "inf" if np.isinf(base) else f"{base:.3f}"
        else:
            base_text = str(base)

        if isinstance(filtered, (float, np.floating)):
            filtered_text = "inf" if np.isinf(filtered) else f"{filtered:.3f}"
        else:
            filtered_text = str(filtered)

        print(
            f"{name:<18} | "
            f"Baseline: {base_text:>12} | "
            f"EMA200: {filtered_text:>12}"
        )

    print("=" * 88)


# ============================================================
# GRÁFICO
# ============================================================

def plot_c2g(
    df_processed,
    trades_df,
    metrics,
    bars=200,
    title_suffix="EMA200"
):
    df_plot = df_processed.tail(bars).copy()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }

    if "volume" in df_plot.columns:
        rename_map["volume"] = "Volume"

    df_plot = df_plot.rename(columns=rename_map)

    apds = [
        mpf.make_addplot(
            df_plot["Trend_Line"],
            color="cyan",
            width=1.8,
            secondary_y=False
        )
    ]

    # EMA200 para visualizar o novo filtro
    if "EMA_Filter" in df_plot.columns:
        apds.append(
            mpf.make_addplot(
                df_plot["EMA_Filter"],
                color="white",
                width=1.0,
                secondary_y=False
            )
        )

    buy_points = np.where(
        df_plot["Buy_Signal"] == 1,
        df_plot["Low"] * 0.998,
        np.nan
    )

    sell_points = np.where(
        df_plot["Sell_Signal"] == 1,
        df_plot["High"] * 1.002,
        np.nan
    )

    if not np.isnan(buy_points).all():
        apds.append(
            mpf.make_addplot(
                buy_points,
                type="scatter",
                markersize=120,
                marker="^",
                color="dodgerblue"
            )
        )

    if not np.isnan(sell_points).all():
        apds.append(
            mpf.make_addplot(
                sell_points,
                type="scatter",
                markersize=120,
                marker="v",
                color="gold"
            )
        )

    market_colors = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        wick="inherit",
        edge="inherit"
    )

    style = mpf.make_mpf_style(
        marketcolors=market_colors,
        base_mpf_style="nightclouds"
    )

    has_volume = "Volume" in df_plot.columns

    fig, axes = mpf.plot(
        df_plot,
        type="candle",
        style=style,
        addplot=apds,
        title=f"\nC2G System Pro - {title_suffix}",
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

    info_text = (
        f"Win Rate: {metrics['win_rate']:.2f}% | "
        f"Trades: {metrics['trades']} | "
        f"PF: {pf_text} | "
        f"PnL: {metrics['total_pnl']:.2f}% | "
        f"Max DD: {metrics['max_drawdown']:.2f}%"
    )

    fig.text(
        0.13,
        0.93,
        info_text,
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
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    COMMON = dict(
        file_path="btc_data.csv",

        # Supertrend
        supertrend_length=10,
        supertrend_multiplier=3.0,

        # ADX
        adx_length=14,

        # ATR / risco
        atr_length=14,
        stop_atr=1.5,
        target_atr=3.0,

        allow_shorts=True,

        # Por enquanto 0 para comparar apenas a lógica.
        # Depois coloque a taxa/slippage reais da sua execução.
        fee_pct=0.0,
        slippage_pct=0.0,
    )

    # ========================================================
    # TESTE A - SEU SISTEMA ATUAL
    # ========================================================

    baseline_df, baseline_trades, baseline_metrics = run_c2g_backtest(
        **COMMON,
        adx_min=20.0,
        use_ema_filter=False,
        ema_filter_length=200,
        test_name="BASELINE - ADX20"
    )

    # ========================================================
    # TESTE B - PRIMEIRA MUDANÇA: EMA200
    # ========================================================

    ema_df, ema_trades, ema_metrics = run_c2g_backtest(
        **COMMON,
        adx_min=20.0,
        use_ema_filter=True,
        ema_filter_length=200,
        test_name="V1.1 - ADX20 + EMA200"
    )

    # ========================================================
    # COMPARAÇÃO
    # ========================================================

    print_comparison(
        baseline_metrics,
        ema_metrics
    )

    # Salva os trades da versão filtrada
    ema_trades.to_csv(
        "c2g_trades_v11_ema200.csv",
        index=False
    )

    # Gráfico da versão filtrada
    plot_c2g(
        ema_df,
        ema_trades,
        ema_metrics,
        bars=200,
        title_suffix="V1.1 - EMA200 Filter"
    )
