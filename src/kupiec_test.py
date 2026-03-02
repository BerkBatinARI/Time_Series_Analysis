import numpy as np
import pandas as pd

TICKER = "SPY"
ALPHA = 0.99  # VaR confidence level


def kupiec_lr_uc(n: int, x: int, p: float) -> float:
    """
    Kupiec (1995) unconditional coverage LR test:

      LR_uc = -2 * ln( L0 / L1 )
      L0: likelihood under expected breach probability p
      L1: likelihood under observed breach probability phat = x/n

    Under H0 (correct coverage), LR_uc ~ Chi-square(df=1).
    """
    if n <= 0:
        return float("nan")

    phat = x / n

    # numerical safety (avoid log(0))
    eps = 1e-12
    phat = min(max(phat, eps), 1 - eps)
    p = min(max(p, eps), 1 - eps)

    l0 = (n - x) * np.log(1 - p) + x * np.log(p)
    l1 = (n - x) * np.log(1 - phat) + x * np.log(phat)

    return -2.0 * (l0 - l1)


def run_one(path: str, var_col: str) -> dict:
    df = pd.read_csv(path, parse_dates=["Date"]).dropna(subset=[var_col, "log_return"]).copy()
    df["loss"] = -df["log_return"]
    df["breach"] = df["loss"] > df[var_col]

    n = len(df)
    x = int(df["breach"].sum())
    p = 1 - ALPHA
    lr = kupiec_lr_uc(n, x, p)

    return {
        "model": var_col,
        "obs": n,
        "breaches": x,
        "breach_rate": x / n if n else float("nan"),
        "expected_rate": p,
        "LR_uc": lr,
    }


def main() -> None:
    normal = run_one(f"data/{TICKER}_ewma_var_es.csv", "var_99")
    student_t = run_one(f"data/{TICKER}_ewma_t_var_es.csv", "var_99_t")

    out = pd.DataFrame([normal, student_t])
    out.to_csv(f"data/{TICKER}_kupiec_summary.csv", index=False)

    print(out.to_string(index=False))
    print(f"\nSaved data/{TICKER}_kupiec_summary.csv")


if __name__ == "__main__":
    main()