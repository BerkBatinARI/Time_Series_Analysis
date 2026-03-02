import os
import numpy as np
import pandas as pd

TICKER = "SPY"
LAMBDA = 0.94          # RiskMetrics-style daily EWMA
Z_99 = 2.326347874      # ~N(0,1) 99% quantile (one-sided)
WINDOW_MIN = 60         # don’t start until we have enough history

def ewma_sigma2(r: np.ndarray, lam: float) -> np.ndarray:
    """
    EWMA variance recursion:
      sigma2[t] = lam * sigma2[t-1] + (1-lam) * r[t-1]^2
    so sigma2[t] is a forecast made using info up to t-1.
    """
    sigma2 = np.full_like(r, fill_value=np.nan, dtype=float)
    # initialise with sample variance of early returns
    init = np.nanvar(r[:WINDOW_MIN])
    sigma2[WINDOW_MIN] = init

    for t in range(WINDOW_MIN + 1, len(r)):
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * (r[t - 1] ** 2)
    return sigma2

def main() -> None:
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv("data/returns.csv", parse_dates=["Date"])
    df = df[df["ticker"] == TICKER].sort_values("Date").reset_index(drop=True)

    r = df["log_return"].to_numpy(dtype=float)

    sigma2 = ewma_sigma2(r, LAMBDA)
    sigma = np.sqrt(sigma2)

    # 1-day parametric VaR/ES under Normal assumption
    # VaR is a positive number representing loss threshold
    var_99 = Z_99 * sigma

    # ES for Normal: ES = sigma * phi(z) / (1-alpha)
    alpha = 0.99
    phi = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * (Z_99**2))
    es_99 = sigma * (phi / (1.0 - alpha))

    out = df[["Date", "ticker", "log_return"]].copy()
    out["ewma_sigma"] = sigma
    out["var_99"] = var_99
    out["es_99"] = es_99

    out_path = f"data/{TICKER}_ewma_var_es.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()