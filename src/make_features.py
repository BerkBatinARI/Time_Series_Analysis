import os
import pandas as pd
import numpy as np

TICKERS = ["SPY", "TLT", "GLD"]

def load_prices(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/{ticker}.csv", parse_dates=["Date"])
    df = df.sort_values("Date")
    # Use Adjusted Close for return calculations
    df = df[["Date", "Adj Close"]].rename(columns={"Adj Close": "adj_close"})
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    df = df.dropna(subset=["adj_close"])
    df["ticker"] = ticker
    return df

def main() -> None:
    os.makedirs("data", exist_ok=True)

    frames = [load_prices(t) for t in TICKERS]
    px = pd.concat(frames, ignore_index=True)

    # log returns
    px["log_return"] = px.groupby("ticker")["adj_close"].transform(lambda s: np.log(s).diff())
    out_path = "data/returns.csv"
    px.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(px)} rows)")

if __name__ == "__main__":
    main()