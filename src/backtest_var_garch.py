import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import apply_finance_style, format_time_axis, add_subtitle

TICKER = "SPY"
ALPHA = 0.99  # 99% VaR


def main() -> None:
    print("RUNNING backtest_var_garch.py main()")
    os.makedirs("reports/figures", exist_ok=True)
    apply_finance_style()

    path = f"data/{TICKER}_garch_var.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return", "var_99_garch"]).copy()
    df["breach"] = df["log_return"] < (-df["var_99_garch"])

    n = int(len(df))
    breaches = int(df["breach"].sum())
    breach_rate = breaches / n if n > 0 else float("nan")

    print(f"Obs used: {n}")
    print(f"Breaches: {breaches}")
    print(f"Breach rate: {breach_rate:.4%}")

    fig, ax = plt.subplots()

    ax.plot(df["Date"], df["log_return"], label="log return")
    ax.plot(df["Date"], -df["var_99_garch"], label="-VaR(99%) GARCH")

    bdf = df[df["breach"]]
    ax.scatter(bdf["Date"], bdf["log_return"], label="breaches", s=14, zorder=3)

    ax.set_title(f"{TICKER} — GARCH(1,1) VaR(99%) Backtest")
    add_subtitle(ax, f"Obs={n:,} | breaches={breaches} | breach rate={breach_rate:.2%} (expected {(1-ALPHA):.2%})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")

    format_time_axis(ax)
    ax.legend(loc="upper left", frameon=False)

    fig_path = f"reports/figures/{TICKER}_var99_garch_backtest.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()