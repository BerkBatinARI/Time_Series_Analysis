import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_style import apply_finance_style, format_time_axis, add_subtitle

TICKER = "SPY"
ALPHA = 0.99


def main() -> None:
    print("RUNNING plot_breaches.py main()")
    os.makedirs("reports/figures", exist_ok=True)
    apply_finance_style()

    path = f"data/{TICKER}_ewma_var_es.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return", "var_99"]).copy()

    df["breach"] = df["log_return"] < (-df["var_99"])
    n = int(len(df))
    breaches = int(df["breach"].sum())
    breach_rate = breaches / n if n > 0 else float("nan")

    fig, ax = plt.subplots()

    # Plot returns as a thin line for context
    ax.plot(df["Date"], df["log_return"], label="log return", alpha=0.75)

    # Mark breaches
    bdf = df[df["breach"]]
    ax.scatter(bdf["Date"], bdf["log_return"], label="VaR breaches", s=18, zorder=3)

    ax.axhline(0.0, linewidth=1.0, alpha=0.4)

    ax.set_title(f"{TICKER} — VaR(99%) Breach Timeline (EWMA)")
    add_subtitle(ax, f"Obs={n:,} | breaches={breaches} | breach rate={breach_rate:.2%} (expected {(1-ALPHA):.2%})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")

    format_time_axis(ax)
    ax.legend(loc="upper left", frameon=False)

    fig_path = f"reports/figures/{TICKER}_var99_breaches.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()