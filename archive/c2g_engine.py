import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import numpy as np


# ============================================================
# AUXILIAR: LOCALIZAR COLUNAS GERADAS PELO PANDAS_TA
# ============================================================

def find_column(columns, prefix):
    matches = [
        col for col in columns
        if str(col).startswith(prefix)
    ]

    if not matches:
        raise KeyError(
            f"Nenhuma coluna começando com '{prefix}' foi encontrada.\n"
            f"Colunas disponíveis: {list(columns)}"
        )

    return matches[0]


# ============================================================
# CÁLCULO DE PNL
# ============================================================

def calculate_pnl(
    side,
    entry_price,
    exit_price,
    fee_pct=0.0,
    slippage_pct=0.0
):
    if side == "BUY":
        gross_pct = (
            (exit_price - entry_price)
            / entry_price
        ) * 100

    else:
        gross_pct = (
            (entry_price - exit_price)
            / entry_price
        ) * 100

    # Taxa na entrada + saída
    total_fees = fee_pct * 2

    # Slippage na entrada + saída
    total_slippage = slippage_pct * 2

    net_pct = (
        gross_pct
        - total_fees
        - total_slippage
    )

    return net_pct


# ============================================================
# BACKTEST
# ============================================================

def run_c2g_backtest(
    file_path="btc_data.csv",

    supertrend_length=10,
    supertrend_multiplier=3.0,

    adx_length=14,
    adx_min=20.0,

    atr_length=14,

    stop_atr=1.5,
    target_atr=3.0,

    allow_shorts=True,

    fee_pct=0.0,
    slippage_pct=0.0
):

    print("\nCarregando dados...")

    # ========================================================
    # 1. CARREGAR CSV
    # ========================================================

    df = pd.read_csv(
        file_path,
        index_col="timestamp",
        parse_dates=True
    )

    df = df.sort_index()

    # Remove timestamps duplicados
    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # Padroniza nomes
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    required_columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    for col in required_columns:

        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada.\n"
                f"Colunas disponíveis: {list(df.columns)}"
            )

    # Converter OHLC para número
    for col in required_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if "volume" in df.columns:

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    print(
        f"Candles carregados: {len(df)}"
    )

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
        raise RuntimeError(
            "Não foi possível calcular o Supertrend."
        )

    print(
        "\nColunas geradas pela Supertrend:"
    )

    print(
        list(supertrend.columns)
    )

    trend_direction_col = find_column(
        supertrend.columns,
        "SUPERTd_"
    )

    trend_line_col = find_column(
        supertrend.columns,
        "SUPERT_"
    )

    print(
        f"Direção encontrada: {trend_direction_col}"
    )

    print(
        f"Linha encontrada: {trend_line_col}"
    )

    # Criamos nomes fixos
    df["Trend_Direction"] = (
        supertrend[
            trend_direction_col
        ]
    )

    df["Trend_Line"] = (
        supertrend[
            trend_line_col
        ]
    )

    # ========================================================
    # 3. ADX
    # ========================================================

    adx = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=adx_length
    )

    if adx is None:
        raise RuntimeError(
            "Não foi possível calcular o ADX."
        )

    adx_col = find_column(
        adx.columns,
        "ADX_"
    )

    df["ADX"] = adx[adx_col]

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
    # 5. SINAIS
    # ========================================================

    # Supertrend:
    #
    # +1 = tendência compradora
    # -1 = tendência vendedora
    #
    # -1 -> +1 = diff +2 = BUY
    # +1 -> -1 = diff -2 = SELL

    df["Signal"] = (
        df["Trend_Direction"].diff()
    )

    df["Buy_Signal"] = (
        (df["Signal"] == 2)
        &
        (df["ADX"] > adx_min)
    ).astype(int)

    df["Sell_Signal"] = (
        (df["Signal"] == -2)
        &
        (df["ADX"] > adx_min)
    ).astype(int)

    print(
        f"\nBUY encontrados: {df['Buy_Signal'].sum()}"
    )

    print(
        f"SELL encontrados: {df['Sell_Signal'].sum()}"
    )

    # ========================================================
    # 6. MOTOR DE BACKTEST
    # ========================================================

    trades = []

    position = None

    entry_price = None
    entry_time = None

    stop_loss = None
    take_profit = None

    # Começa em 1 porque usamos candle anterior
    for i in range(1, len(df)):

        previous = df.iloc[i - 1]
        current = df.iloc[i]

        current_open = float(
            current["open"]
        )

        current_high = float(
            current["high"]
        )

        current_low = float(
            current["low"]
        )

        current_close = float(
            current["close"]
        )

        # ====================================================
        # GERENCIAR BUY
        # ====================================================

        if position == "BUY":

            exit_price = None
            exit_reason = None

            stop_hit = (
                current_low <= stop_loss
            )

            target_hit = (
                current_high >= take_profit
            )

            # Se TP e SL forem atingidos
            # no mesmo candle:
            # usamos STOP primeiro por conservadorismo

            if stop_hit:

                exit_price = stop_loss
                exit_reason = "STOP"

            elif target_hit:

                exit_price = take_profit
                exit_reason = "TAKE_PROFIT"

            elif (
                current["Trend_Direction"] == -1
            ):

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

        # ====================================================
        # GERENCIAR SELL
        # ====================================================

        elif position == "SELL":

            exit_price = None
            exit_reason = None

            stop_hit = (
                current_high >= stop_loss
            )

            target_hit = (
                current_low <= take_profit
            )

            if stop_hit:

                exit_price = stop_loss
                exit_reason = "STOP"

            elif target_hit:

                exit_price = take_profit
                exit_reason = "TAKE_PROFIT"

            elif (
                current["Trend_Direction"] == 1
            ):

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

        # ====================================================
        # ABRIR NOVA OPERAÇÃO
        #
        # IMPORTANTE:
        # sinal ocorreu no fechamento anterior
        # entrada acontece no OPEN seguinte
        # ====================================================

        if position is None:

            previous_atr = previous["ATR"]

            if (
                pd.notna(previous_atr)
                and previous_atr > 0
            ):

                # ============================================
                # BUY
                # ============================================

                if (
                    previous["Buy_Signal"] == 1
                ):

                    position = "BUY"

                    entry_price = (
                        current_open
                    )

                    entry_time = (
                        df.index[i]
                    )

                    stop_loss = (
                        entry_price
                        -
                        (
                            stop_atr
                            *
                            previous_atr
                        )
                    )

                    take_profit = (
                        entry_price
                        +
                        (
                            target_atr
                            *
                            previous_atr
                        )
                    )

                # ============================================
                # SELL
                # ============================================

                elif (
                    allow_shorts
                    and
                    previous["Sell_Signal"] == 1
                ):

                    position = "SELL"

                    entry_price = (
                        current_open
                    )

                    entry_time = (
                        df.index[i]
                    )

                    stop_loss = (
                        entry_price
                        +
                        (
                            stop_atr
                            *
                            previous_atr
                        )
                    )

                    take_profit = (
                        entry_price
                        -
                        (
                            target_atr
                            *
                            previous_atr
                        )
                    )

    # ========================================================
    # FECHAR POSIÇÃO NO FINAL
    # ========================================================

    if position is not None:

        final_price = float(
            df["close"].iloc[-1]
        )

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

    # ========================================================
    # 7. RELATÓRIO
    # ========================================================

    trades_df = pd.DataFrame(
        trades
    )

    if len(trades_df) > 0:

        winners = trades_df[
            trades_df["pnl_pct"] > 0
        ]

        losers = trades_df[
            trades_df["pnl_pct"] < 0
        ]

        win_rate = (
            len(winners)
            /
            len(trades_df)
        ) * 100

        total_pnl = (
            trades_df["pnl_pct"].sum()
        )

        avg_trade = (
            trades_df["pnl_pct"].mean()
        )

        gross_profit = (
            winners["pnl_pct"].sum()
        )

        gross_loss = abs(
            losers["pnl_pct"].sum()
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        else:

            profit_factor = np.inf

    else:

        win_rate = 0.0
        total_pnl = 0.0
        avg_trade = 0.0
        profit_factor = 0.0

    print("\n")
    print("=" * 60)
    print("          C2G SYSTEM PRO - BACKTEST")
    print("=" * 60)

    print(
        f"Total de candles:        {len(df)}"
    )

    print(
        f"Total de trades:         {len(trades_df)}"
    )

    print(
        f"Win Rate:                {win_rate:.2f}%"
    )

    print(
        f"PnL acumulado:           {total_pnl:.4f}%"
    )

    print(
        f"Média por trade:         {avg_trade:.4f}%"
    )

    if np.isinf(profit_factor):

        print(
            "Profit Factor:           infinito"
        )

    else:

        print(
            f"Profit Factor:           {profit_factor:.3f}"
        )

    print(
        f"Stop:                    {stop_atr} ATR"
    )

    print(
        f"Take Profit:             {target_atr} ATR"
    )

    print(
        f"ADX mínimo:              {adx_min}"
    )

    print(
        f"Taxa por lado:           {fee_pct:.4f}%"
    )

    print(
        f"Slippage por lado:       {slippage_pct:.4f}%"
    )

    print("=" * 60)

    if not trades_df.empty:

        print("\nÚLTIMOS TRADES:\n")

        print(
            trades_df[
                [
                    "side",
                    "entry_time",
                    "exit_time",
                    "reason",
                    "pnl_pct"
                ]
            ]
            .tail(20)
            .to_string(
                index=False
            )
        )

    return df, trades_df


# ============================================================
# GRÁFICO
# ============================================================

def plot_c2g(
    df_processed,
    bars=200
):

    df_plot = (
        df_processed
        .tail(bars)
        .copy()
    )

    # ========================================================
    # MPLFINANCE PRECISA DE:
    #
    # Open
    # High
    # Low
    # Close
    # Volume
    # ========================================================

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }

    if "volume" in df_plot.columns:

        rename_map[
            "volume"
        ] = "Volume"

    df_plot = df_plot.rename(
        columns=rename_map
    )

    # ========================================================
    # SUPERTREND
    #
    # ESSA É A CORREÇÃO DO SEU ERRO:
    #
    # usamos Trend_Line
    #
    # e NÃO SUPERT_10_3
    # ========================================================

    apds = [
        mpf.make_addplot(
            df_plot["Trend_Line"],
            color="cyan",
            width=1.8,
            secondary_y=False
        )
    ]

    # ========================================================
    # BUY
    # ========================================================

    buy_points = np.where(
        df_plot["Buy_Signal"] == 1,
        df_plot["Low"] * 0.998,
        np.nan
    )

    # ========================================================
    # SELL
    # ========================================================

    sell_points = np.where(
        df_plot["Sell_Signal"] == 1,
        df_plot["High"] * 1.002,
        np.nan
    )

    if not np.isnan(
        buy_points
    ).all():

        apds.append(
            mpf.make_addplot(
                buy_points,
                type="scatter",
                markersize=120,
                marker="^",
                color="dodgerblue"
            )
        )

    if not np.isnan(
        sell_points
    ).all():

        apds.append(
            mpf.make_addplot(
                sell_points,
                type="scatter",
                markersize=120,
                marker="v",
                color="gold"
            )
        )

    # ========================================================
    # CORES
    # ========================================================

    market_colors = (
        mpf.make_marketcolors(
            up="#26a69a",
            down="#ef5350",
            wick="inherit",
            edge="inherit"
        )
    )

    style = (
        mpf.make_mpf_style(
            marketcolors=market_colors,
            base_mpf_style="nightclouds"
        )
    )

    # ========================================================
    # VOLUME
    # ========================================================

    has_volume = (
        "Volume"
        in
        df_plot.columns
    )

    print(
        "\nRenderizando gráfico..."
    )

    fig, axes = mpf.plot(
        df_plot,

        type="candle",

        style=style,

        addplot=apds,

        title=(
            "\nC2G System Pro "
            "- Risk Managed Backtest"
        ),

        ylabel="Price (USDT)",

        volume=has_volume,

        figratio=(16, 9),

        figscale=1.1,

        returnfig=True
    )

    mpf.show()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    df_processed, trades_df = run_c2g_backtest(

        file_path="btc_data.csv",

        # SUPERTREND
        supertrend_length=10,
        supertrend_multiplier=3.0,

        # ADX
        adx_length=14,
        adx_min=20.0,

        # ATR
        atr_length=14,

        # RISCO
        stop_atr=1.5,
        target_atr=3.0,

        # SHORT
        allow_shorts=True,

        # TAXAS
        fee_pct=0.0,

        # SLIPPAGE
        slippage_pct=0.0
    )

    plot_c2g(
        df_processed,
        bars=200
    )
    
    