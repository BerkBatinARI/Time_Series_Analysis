import os
import pandas as pd
import matplotlib.pyplot as plt

TICKER = "SPY"

def main() -> None:
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv("data/returns.csv", parse_dates=["Date"])
    df = df[df["ticker"] == TICKER].sort_values("Date")

    # 21-day rolling volatility (annualised, using sqrt(252))
    df["roll_vol_21"] = df["log_return"].rolling(21).std() * (252 ** 0.5)

    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["roll_vol_21"])
    ax.set_title(f"{TICKER}: 21-day rolling vol (annualised)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vol")
    fig.tight_layout()
    out_path = f"reports/{TICKER}_rolling_vol_21.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()