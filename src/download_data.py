import os
from datetime import datetime
import pandas as pd
import yfinance as yf

# Simple, reproducible universe (liquid, finance-relevant)
TICKERS = {
    "SPY": "US equities (S&P 500 ETF)",
    "TLT": "US long bonds (20Y+ Treasury ETF)",
    "GLD": "Gold ETF",
}

START = "2005-01-01"


def download_one(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    df = df.rename_axis("Date").reset_index()
    return df


def main() -> None:
    os.makedirs("data", exist_ok=True)

    meta_rows = []
    for ticker, desc in TICKERS.items():
        df = download_one(ticker, START)
        out_path = f"data/{ticker}.csv"
        df.to_csv(out_path, index=False)

        meta_rows.append(
            {
                "ticker": ticker,
                "description": desc,
                "start": df["Date"].min(),
                "end": df["Date"].max(),
                "rows": len(df),
                "downloaded_utc": datetime.utcnow().isoformat(timespec="seconds"),
                "source": "yfinance",
            }
        )
        print(f"Saved {out_path} ({len(df)} rows)")

    pd.DataFrame(meta_rows).to_csv("data/metadata.csv", index=False)
    print("Saved data/metadata.csv")


if __name__ == "__main__":
    main()