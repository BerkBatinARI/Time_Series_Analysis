import numpy as np
import pandas as pd

TICKER = "SPY"
ALPHA = 0.99  # 99% VaR


def christoffersen_lr_cc(breaches: np.ndarray, alpha: float) -> dict:
    """
    Christoffersen (1998) Conditional Coverage test:
    LR_cc = LR_uc + LR_ind

    breaches: array of 0/1 where 1 = VaR exception (loss exceeds VaR)
    alpha: VaR confidence level (e.g., 0.99)
    """
    x = breaches.astype(int)

    # Transition counts
    n00 = np.sum((x[:-1] == 0) & (x[1:] == 0))
    n01 = np.sum((x[:-1] == 0) & (x[1:] == 1))
    n10 = np.sum((x[:-1] == 1) & (x[1:] == 0))
    n11 = np.sum((x[:-1] == 1) & (x[1:] == 1))

    n0 = n00 + n01
    n1 = n10 + n11

    # Unconditional coverage (Kupiec-style)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)  # observed exception probability
    p = 1 - alpha  # expected exception probability

    def safe_log(a: float) -> float:
        return np.log(a) if a > 0 else -np.inf

    # Likelihood under null (p) vs alternative (pi)
    ll_null = (n01 + n11) * safe_log(p) + (n00 + n10) * safe_log(1 - p)
    ll_alt = (n01 + n11) * safe_log(pi) + (n00 + n10) * safe_log(1 - pi)
    lr_uc = -2 * (ll_null - ll_alt)

    # Independence test
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0

    ll_ind_alt = (
        n00 * safe_log(1 - pi01)
        + n01 * safe_log(pi01)
        + n10 * safe_log(1 - pi11)
        + n11 * safe_log(pi11)
    )

    ll_ind_null = (
        (n00 + n10) * safe_log(1 - pi)
        + (n01 + n11) * safe_log(pi)
    )

    lr_ind = -2 * (ll_ind_null - ll_ind_alt)

    lr_cc = lr_uc + lr_ind

    return {
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11),
        "pi_obs": float(pi),
        "p_exp": float(p),
        "LR_uc": float(lr_uc),
        "LR_ind": float(lr_ind),
        "LR_cc": float(lr_cc),
    }


def load_breaches_from_file(path: str, var_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_return", var_col]).copy()
    df["breach"] = df["log_return"] < (-df[var_col])
    return df


def main() -> None:
    print("RUNNING christoffersen_test.py main()")

    # Model files + VaR column names
    models = [
        ("ewma_normal", f"data/{TICKER}_ewma_var_es.csv", "var_99"),
        ("ewma_t", f"data/{TICKER}_ewma_t_var_es.csv", "var_99_t"),
        ("garch_normal", f"data/{TICKER}_garch_var.csv", "var_99_garch"),
    ]

    rows = []
    for name, path, var_col in models:
        df = load_breaches_from_file(path, var_col)
        stats = christoffersen_lr_cc(df["breach"].to_numpy(dtype=int), ALPHA)
        stats["model"] = name
        stats["obs"] = int(len(df))
        stats["breaches"] = int(df["breach"].sum())
        stats["breach_rate"] = float(df["breach"].mean())
        rows.append(stats)

    out = pd.DataFrame(rows)[
        ["model", "obs", "breaches", "breach_rate", "p_exp", "LR_uc", "LR_ind", "LR_cc", "n00", "n01", "n10", "n11", "pi_obs"]
    ]

    out_path = f"data/{TICKER}_christoffersen_summary.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()