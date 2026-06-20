"""Systematic anomaly check on all aggregated columns across 5 historical tables."""
import pandas as pd
import numpy as np

DATA = "data/original"


def describe_table(name, path, cols, sentinel_checks=None):
    print(f"\n{'='*60}")
    print(f"  TABLE: {name}")
    print(f"{'='*60}")
    df = pd.read_csv(path, usecols=cols)
    print(f"  Rows: {len(df):,}")

    for c in cols:
        s = df[c]
        stats = {
            "count": int(s.count()),
            "missing": int(s.isna().sum()),
            "min": s.min(),
            "max": s.max(),
            "mean": s.mean() if np.issubdtype(s.dtype, np.number) else None,
            "p01": s.quantile(0.01) if np.issubdtype(s.dtype, np.number) else None,
            "p99": s.quantile(0.99) if np.issubdtype(s.dtype, np.number) else None,
            "neg": int((s < 0).sum()) if np.issubdtype(s.dtype, np.number) else None,
            "zero": int((s == 0).sum()) if np.issubdtype(s.dtype, np.number) else None,
        }
        sentinel_info = ""
        if sentinel_checks and c in sentinel_checks:
            for sentinel_val in sentinel_checks[c]:
                n_sentinel = int((s == sentinel_val).sum())
                if n_sentinel > 0:
                    sentinel_info += f"  sentinel({sentinel_val})={n_sentinel}"
        print(f"\n  [{c}]")
        print(f"    count={stats['count']:,}  missing={stats['missing']:,}"
              f"  min={stats['min']}  max={stats['max']}")
        if stats['p01'] is not None:
            print(f"    p01={stats['p01']:.4f}  p99={stats['p99']:.4f}"
                  f"  neg={stats['neg']:,}  zero={stats['zero']:,}"
                  f"{sentinel_info}")


# === bureau ===
bureau_cols = [
    "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DAYS_CREDIT_UPDATE", "DAYS_ENDDATE_FACT",
]
describe_table("bureau", f"{DATA}/bureau.csv", bureau_cols)

# === previous_application ===
prev_cols = [
    "AMT_APPLICATION", "AMT_CREDIT", "DAYS_DECISION", "CNT_PAYMENT",
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION",
]
describe_table("previous_application", f"{DATA}/previous_application.csv", prev_cols,
               sentinel_checks={
                   "DAYS_FIRST_DRAWING": [365243],
                   "DAYS_FIRST_DUE": [365243],
                   "DAYS_LAST_DUE_1ST_VERSION": [365243],
                   "DAYS_LAST_DUE": [365243],
                   "DAYS_TERMINATION": [365243],
               })

# === credit_card ===
cc_cols = [
    "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD",
    "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
]
describe_table("credit_card_balance", f"{DATA}/credit_card_balance.csv", cc_cols)

# === installments ===
inst_cols = [
    "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT",
]
describe_table("installments_payments", f"{DATA}/installments_payments.csv", inst_cols)

# === POS_CASH ===
pos_cols = ["SK_DPD", "SK_DPD_DEF"]
describe_table("POS_CASH_balance", f"{DATA}/POS_CASH_balance.csv", pos_cols)

print("\n\nDone.")
