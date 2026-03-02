import os
import pandas as pd
import matplotlib.pyplot as plt

TICKER = "SPY"

def main() -> None:
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(f"data/{TICKER}_ewma_var_es.csv", parse_dates=["Date"])
    df = df.dropna(subset=["ewma_sigma", "var_99"]).copy()

    # A VaR breach happens if realised loss > VaR threshold
    # loss = -return (since negative returns are losses)
    df["loss"] = -df["log_return"]
    df["breach"] = df["loss"] > df["var_99"]

    n = len(df)
    k = int(df["breach"].sum())
    rate = k / n if n else float("nan")

    print(f"Obs: {n}")
    print(f"VaR(99%) breaches: {k} ({rate:.3%})  | expected ~1.000%")

    # Plot: losses and VaR threshold (recent window for readability)
    view = df.tail(800).copy()

    fig, ax = plt.subplots()
    ax.plot(view["Date"], view["loss"], label="1-day loss (-log return)")
    ax.plot(view["Date"], view["var_99"], label="VaR 99% (EWMA)")
    ax.set_title(f"{TICKER} — VaR(99%) backtest (last 800 days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    out_path = f"reports/{TICKER}_var99_backtest.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()