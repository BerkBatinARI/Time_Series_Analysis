import pandas as pd

TICKER = "SPY"

def main() -> None:
    df = pd.read_csv(f"data/{TICKER}_ewma_t_var_es.csv", parse_dates=["Date"])
    df = df.dropna(subset=["ewma_sigma", "var_99_t"]).copy()

    df["loss"] = -df["log_return"]
    df["breach_t"] = df["loss"] > df["var_99_t"]

    n = len(df)
    k = int(df["breach_t"].sum())
    rate = k / n if n else float("nan")

    print(f"Obs: {n}")
    print(f"Student-t VaR(99%) breaches: {k} ({rate:.3%})  | expected ~1.000%")

if __name__ == "__main__":
    main()