import os
import numpy as np
import pandas as pd
from scipy.stats import t

TICKER = "SPY"
LAMBDA = 0.94
ALPHA = 0.99
WINDOW_MIN = 60
DF_T = 6  # degrees of freedom for Student-t (fatter tails than Normal)

def ewma_sigma2(r: np.ndarray, lam: float) -> np.ndarray:
    sigma2 = np.full_like(r, fill_value=np.nan, dtype=float)
    init = np.nanvar(r[:WINDOW_MIN])
    sigma2[WINDOW_MIN] = init
    for t_idx in range(WINDOW_MIN + 1, len(r)):
        sigma2[t_idx] = lam * sigma2[t_idx - 1] + (1.0 - lam) * (r[t_idx - 1] ** 2)
    return sigma2

def main() -> None:
    os.makedirs("data", exist_ok=True)

    df = pd.read_csv("data/returns.csv", parse_dates=["Date"])
    df = df[df["ticker"] == TICKER].sort_values("Date").reset_index(drop=True)

    r = df["log_return"].to_numpy(dtype=float)
    sigma = np.sqrt(ewma_sigma2(r, LAMBDA))

    # Student-t quantile for 1% left tail => loss quantile for ALPHA
    q = t.ppf(1 - ALPHA, df=DF_T)  # negative number

    # Scale factor so variance matches sigma^2
    scale = sigma * np.sqrt((DF_T - 2) / DF_T)

    # VaR as positive loss threshold
    var_99_t = -(q * scale)

    # ES for Student-t (closed form)
    # ES_alpha = scale * ( pdf(q) * (df + q^2) / ((df-1)*(1-alpha)) )  with q negative
    pdf_q = t.pdf(q, df=DF_T)
    es_99_t = scale * (pdf_q * (DF_T + q*q) / ((DF_T - 1) * (1 - ALPHA)))

    out = df[["Date", "ticker", "log_return"]].copy()
    out["ewma_sigma"] = sigma
    out["var_99_t"] = var_99_t
    out["es_99_t"] = es_99_t

    out_path = f"data/{TICKER}_ewma_t_var_es.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()