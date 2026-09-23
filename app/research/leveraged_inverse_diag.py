#!/usr/bin/env python3
"""Diagnostic: how do inverse leveraged funds track the underlying around the open?

Questions answered:
1. Overnight gap beta: inv fund prior-close->open vs underlying prior-close->open.
2. Morning continuation beta: inv fund open->10:00 vs underlying open->10:00
   (should be ~ -leverage if the fund tracks properly).
3. On trigger days (underlying prior session <= -2%): how often is the inverse
   fund's OPEN gap in the "right" direction (i.e. fund UP when underlying gaps
   down), and what is the fund's open->10:00 return distribution?
Read-only.
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from alpaca.data.enums import Adjustment  # noqa: E402

from data_provider import data_provider  # noqa: E402
from strategies.leveraged_rebalance_signal import prepare_features  # noqa: E402
from strategies.leveraged_single_stock_etfs import (  # noqa: E402
    LEVERAGED_SINGLE_STOCK_ETFS,
)

OPEN_TOD = 570
T10_TOD = 600


def main():
    end = datetime.now()
    start = end - timedelta(days=450)

    rows = []
    for underlying in sorted({f.underlying for f in LEVERAGED_SINGLE_STOCK_ETFS.values()}):
        invs = [f for f in LEVERAGED_SINGLE_STOCK_ETFS.values()
                if f.underlying == underlying and f.leverage < 0]
        if not invs:
            continue
        fund = max(invs, key=lambda f: abs(f.leverage))
        lev = fund.leverage

        ub = data_provider.get_single_stock_bars(
            underlying, start - timedelta(days=10), end,
            timeframe="5m", adjustment=Adjustment.SPLIT)
        fb = data_provider.get_single_stock_bars(
            fund.etf, start - timedelta(days=10), end,
            timeframe="5m", adjustment=Adjustment.SPLIT)
        if ub is None or fb is None or ub.empty or fb.empty:
            continue
        uf, ff = prepare_features(ub), prepare_features(fb)

        def px(df, tod):
            bars = df[df["tod"] == tod]
            return bars.set_index("date")["close"]

        def daily_close(df):
            return df.groupby("date")["close"].last().sort_index()

        u_pc, f_pc = px(uf, OPEN_TOD), px(ff, OPEN_TOD)
        u_t10, f_t10 = px(uf, T10_TOD), px(ff, T10_TOD)
        u_last, f_last = daily_close(uf), daily_close(ff)

        idx = u_pc.index.intersection(f_pc.index)
        u_pc_d, f_pc_d = u_pc.reindex(idx), f_pc.reindex(idx)
        u_t10_d, f_t10_d = u_t10.reindex(idx), f_t10.reindex(idx)
        u_last_s = u_last.shift(1).reindex(idx)
        f_last_s = f_last.shift(1).reindex(idx)
        u_prev_s = u_last.shift(2).reindex(idx)
        # Prior-session return at date d = close(d-1)/close(d-2) - 1.
        prior_u = u_last_s / u_prev_s - 1.0
        u_gap = u_pc_d / u_last_s - 1.0
        f_gap = f_pc_d / f_last_s - 1.0
        u_cont = u_t10_d / u_pc_d - 1.0
        f_cont = f_t10_d / f_pc_d - 1.0
        ok = (u_pc_d.notna() & f_pc_d.notna() & u_t10_d.notna() & f_t10_d.notna()
              & u_last_s.notna() & f_last_s.notna() & u_prev_s.notna()
              & (u_last_s > 0) & (f_last_s > 0) & (u_prev_s > 0)
              & np.isfinite(u_gap) & np.isfinite(f_gap)
              & np.isfinite(u_cont) & np.isfinite(f_cont)
              & np.isfinite(prior_u))
        for d in idx[ok]:
            rows.append({
                "date": d, "underlying": underlying, "etf": fund.etf,
                "lev": lev, "u_gap": float(u_gap[d]), "f_gap": float(f_gap[d]),
                "u_cont": float(u_cont[d]), "f_cont": float(f_cont[d]),
                "prior_u_ret": float(prior_u[d]),
            })

    df = pd.DataFrame(rows)
    print(f"sessions with both legs: {len(df)}")

    print("\n--- 1. OVERNIGHT GAP: inv fund gap vs underlying gap (open vs prior close) ---")
    b = np.polyfit(df["u_gap"], df["f_gap"], 1)
    corr = df["u_gap"].corr(df["f_gap"])
    print(f"  beta={b[0]:+.2f}  alpha/day={b[1]:+.4%}  corr={corr:+.2f}")
    print(f"  (perfect tracking would be beta={df['lev'].mean():+.0f} approx)")
    agree_down = df[df["u_gap"] < 0]
    print(f"  underlying gaps DOWN ({len(agree_down)} rows): "
          f"inv fund gap mean={agree_down['f_gap'].mean():+.3%}, "
          f"P(fund gap>0)={ (agree_down['f_gap']>0).mean():.0%}")

    print("\n--- 2. MORNING CONTINUATION: inv fund open->10:00 vs underlying open->10:00 ---")
    b2 = np.polyfit(df["u_cont"], df["f_cont"], 1)
    corr2 = df["u_cont"].corr(df["f_cont"])
    print(f"  beta={b2[0]:+.2f}  corr={corr2:+.2f}  (perfect inverse tracking ≈ lev)")
    print(f"  mean u_cont={df['u_cont'].mean():+.3%}  mean f_cont={df['f_cont'].mean():+.3%}")
    # if beta ≈ lev, buying the fund long ≈ -lev exposure to u_cont: implied pnl
    implied = -abs(df["lev"].median()) * df["u_cont"] * 0  # placeholder
    implied_pnl = df["f_cont"]
    print(f"  buying fund open->10:00 unconditionally: mean={implied_pnl.mean():+.3%} "
          f"t={implied_pnl.mean()/(implied_pnl.std()/np.sqrt(len(implied_pnl))):+.2f} n={len(df)}")

    print("\n--- 3. TRIGGER DAYS (underlying prior session <= -2%) ---")
    trig = df[df["prior_u_ret"] <= -0.02].copy()
    print(f"  n={len(trig)}")
    print(f"  underlying open->10:00 continuation mean={trig['u_cont'].mean():+.3%} "
          f"(t={trig['u_cont'].mean()/(trig['u_cont'].std()/np.sqrt(len(trig))):+.2f})")
    print(f"  inv fund open->10:00 mean={trig['f_cont'].mean():+.3%} "
          f"(t={trig['f_cont'].mean()/(trig['f_cont'].std()/np.sqrt(len(trig))):+.2f})")
    # within trigger days: correlation between u_cont and f_cont
    print(f"  corr(u_cont, f_cont) on trigger days={trig['u_cont'].corr(trig['f_cont']):+.2f}")
    fit = np.polyfit(trig["u_cont"], trig["f_cont"], 1)
    print(f"  trigger-day beta(u_cont->f_cont)={fit[0]:+.2f}")
    print(f"  P(fund up open->10:00 on trigger days)={(trig['f_cont']>0).mean():.0%}")
    # where does the edge go? decompose: fund ret = beta*u_cont + residual
    resid = trig["f_cont"] - fit[0] * trig["u_cont"]
    print(f"  residual (fund-specific) mean={resid.mean():+.3%} std={resid.std():.3%}")
    # implied strategy pnl per session: mean f_cont = what we measured before
    print("\n  DECOMPOSITION: expected fund ret if perfect tracking = -lev * u_cont")
    perfect = -abs(trig["lev"].median()) * trig["u_cont"]
    print(f"    perfect-tracking mean={perfect.mean():+.3%} vs actual fund mean={trig['f_cont'].mean():+.3%}")
    print(f"    shortfall (alpha drag)={trig['f_cont'].mean() - perfect.mean():+.3%}/sess")

    # Down-day gap direction agreement
    gap_down = trig[trig["u_gap"] < 0]
    print(f"\n  trigger days where underlying ALSO gaps down at open ({len(gap_down)}):")
    print(f"    P(inv fund gaps UP)={(gap_down['f_gap']>0).mean():.0%}")
    print(f"    inv fund gap mean={gap_down['f_gap'].mean():+.3%}")
    gap_up = trig[trig["u_gap"] >= 0]
    print(f"  trigger days where underlying gaps UP at open ({len(gap_up)}):")
    print(f"    P(inv fund gaps UP)={(gap_up['f_gap']>0).mean():.0%}")
    print(f"    inv fund gap mean={gap_up['f_gap'].mean():+.3%}")

    df.to_csv("/tmp/inv_diag.csv", index=False)
    print("\nwrote /tmp/inv_diag.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())