"""Tests for app/portfolio_backtest.py (synthetic data, no network)."""
import os
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import portfolio_backtest as pb  # noqa: E402

CFG = SimpleNamespace(MAX_POSITIONS=3, MAX_NEW_POSITIONS_PER_DAY=2, POSITION_SIZE_PCT=0.1,
                      MIN_CASH_PCT=0.1, MIN_WIN_RATE=0.0, MIN_NUM_TRADES=0)


def tr(sym, e, x, score=1.0, ep=10.0, xp=11.0, side="long"):
    return dict(symbol=sym, entry_date=e, exit_date=x, entry_price=ep, exit_price=xp,
                side=side, score=score)


def sim(trades, cfg=CFG, cash=10_000.0, **kw):
    return pb.simulate_portfolio(trades, cash, cfg, **kw)


def test_position_cap_and_skip_reason():
    r = sim([tr(s, "2024-01-02", "2024-02-01", score=i) for i, s in enumerate("ABCD")],
            SimpleNamespace(**{**CFG.__dict__, "MAX_POSITIONS": 2, "MAX_NEW_POSITIONS_PER_DAY": 9}))
    assert len(r.trades_taken) == 2
    assert set(r.trades_skipped["reason"]) == {"max_positions"}
    assert set(r.trades_taken["symbol"]) == {"C", "D"}  # highest scores


def test_daily_new_entry_cap_and_score_order():
    r = sim([tr(s, "2024-01-02", "2024-02-01", score=i) for i, s in enumerate("ABC")])
    assert list(r.trades_taken["symbol"]) == ["C", "B"]
    assert list(r.trades_skipped["reason"]) == ["max_new_per_day"]


def test_cash_buffer_respected():
    cfg = SimpleNamespace(**{**CFG.__dict__, "POSITION_SIZE_PCT": 0.5, "MAX_POSITIONS": 10,
                             "MAX_NEW_POSITIONS_PER_DAY": 10, "MIN_CASH_PCT": 0.2})
    r = sim([tr(s, "2024-01-02", "2024-02-01", score=i) for i, s in enumerate("ABC")], cfg)
    # first 50%, second limited to 30% by the 20% buffer, third has no room
    assert (r.equity >= 0).all()
    spent = (r.trades_taken["shares"] * r.trades_taken["entry_price"]).sum()
    assert spent <= 10_000 * 0.8 + 1e-9
    assert list(r.trades_skipped["reason"]) == ["insufficient_cash"]


def test_exits_before_entries_same_day():
    cfg = SimpleNamespace(**{**CFG.__dict__, "MAX_POSITIONS": 1})
    r = sim([tr("A", "2024-01-02", "2024-01-10"), tr("B", "2024-01-10", "2024-01-20")], cfg)
    assert list(r.trades_taken["symbol"]) == ["A", "B"]
    assert r.trades_skipped.empty


def test_duplicate_symbol_and_zero_shares():
    r = sim([tr("A", "2024-01-02", "2024-02-01"), tr("A", "2024-01-03", "2024-02-01"),
             tr("Z", "2024-01-04", "2024-02-01", ep=50_000.0, xp=50_000.0)])
    assert set(r.trades_skipped["reason"]) == {"duplicate_symbol", "zero_shares"}


def test_metrics_and_mark_to_market():
    px = {"A": pd.Series([10.0, 12.0, 11.0], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))}
    r = sim([tr("A", "2024-01-02", "2024-01-04", xp=11.0)], prices=px)
    assert r.equity.loc["2024-01-03"] > r.equity.loc["2024-01-02"]  # marked up
    assert r.metrics["win_rate"] == 1.0 and 0 < r.metrics["exposure"] < 1
    assert abs(r.equity.iloc[-1] - (10_000 + 100 * 1)) < 1e-6  # 100 shares * $1


def test_short_pnl():
    r = sim([tr("A", "2024-01-02", "2024-01-05", ep=10.0, xp=9.0, side="short")])
    assert r.trades_taken["pnl"].iloc[0] == 100.0


# --- walk-forward / placebo with a fake strategy -----------------------------------

class FakeStrategy:
    """optimize() picks 'long' if the slice it is given trended up, else 'short'."""
    seen_max: list = []

    def warmup_days(self):
        return 10

    def get_param_grid(self, direction="long"):
        return [{"side": "long"}, {"side": "short"}, {"side": "long"}, {"side": "short"}]

    def _res(self, side, ok=True):
        return SimpleNamespace(params={"side": side}, alpha=0.1 if ok else -1, win_rate=1.0, num_trades=5,
                               profitable=True, composite_score=0.0, sharpe_ratio=1.0, calmar_ratio=1.0)

    def optimize(self, data, symbol, direction="long", initial_cash=10000):
        FakeStrategy.seen_max.append(data.index.max())
        up = data["close"].iloc[-1] > data["close"].iloc[0]
        return self._res("long" if up else "short")

    def backtest(self, data, symbol, initial_cash=10000, prepared=None, **params):
        side = params.get("side", "long")
        out = [dict(entry_date=data.index[i], exit_date=data.index[i + 3], entry_price=100.0,
                    exit_price=float(data["close"].iloc[i + 3]), side=side)
               for i in range(0, len(data) - 3, 8)]
        r = self._res(side)
        r.trade_details = out
        return r


def test_walk_forward_never_peeks_at_oos():
    FakeStrategy.seen_max = []
    # IS (first 6 months) trends up; the following OOS trends sharply down.
    idx = pd.bdate_range("2023-01-02", periods=400)
    close = np.where(idx < pd.Timestamp("2023-07-01"), 100 + np.arange(400) * 0.1,
                     130 - (np.arange(400) - 130) * 1.0)
    data = {"X": pd.DataFrame({"close": close}, index=idx)}
    wins = pb._windows(idx[0].to_pydatetime(), idx[-1].to_pydatetime(), 6, 2, 2)
    cands = pb.generate_candidates_walk_forward(["X"], data, FakeStrategy, 6, 2, 2, CFG)
    assert cands
    # first window: IS uptrend => 'long' chosen despite OOS decline (peeking would pick 'short')
    first_oos_end = pd.Timestamp(wins[0][3])
    first = [c for c in cands if c["entry_date"] < first_oos_end]
    assert first and all(c["side"] == "long" for c in first)
    assert all(c["entry_date"] >= pd.Timestamp(wins[0][2]) for c in first)
    # every IS slice ended strictly before its window's OOS start
    for m, w in zip(FakeStrategy.seen_max, wins):
        assert pd.Timestamp(m) < pd.Timestamp(w[2])


def test_windows_match_validator_convention():
    from walk_forward import WalkForwardValidator
    g = SimpleNamespace(WF_IS_MONTHS=6, WF_OOS_MONTHS=2, WF_STEP_MONTHS=2)
    s, e = datetime(2023, 1, 1), datetime(2024, 6, 1)
    with patch("walk_forward.globalConfig", g):
        ref = WalkForwardValidator(None)._compute_window_boundaries(s, e)
    assert pb._windows(s, e, 6, 2, 2) == ref


def test_placebo_reproducible_and_percentile():
    idx = pd.bdate_range("2023-01-02", periods=400)
    close = 100 + np.arange(400) * 0.2
    data = {"X": pd.DataFrame({"close": close}, index=idx), "Y": pd.DataFrame({"close": close}, index=idx)}
    kw = dict(symbols=["X", "Y"], data_by_symbol=data, strategy_factory=FakeStrategy,
              is_months=6, oos_months=2, step_months=2, n_runs=6, config=CFG)
    a = pb.placebo_test(seed=7, **kw)
    b = pb.placebo_test(seed=7, **kw)
    assert a["placebo_sharpes"] == b["placebo_sharpes"]
    assert a["placebo_cagrs"] == b["placebo_cagrs"]
    assert 0 <= a["sharpe_percentile"] <= 100 and len(a["placebo_sharpes"]) == 6


def test_percentile_midrank():
    assert pb._percentile(5, [1, 2, 3, 4]) == 100.0
    assert pb._percentile(2, [1, 2, 3]) == 50.0


# --- window-end forced exits ---------------------------------------------------------

class OpenAtEndStrategy(FakeStrategy):
    """Enters on the 2nd bar of every slice and never exits unless force-closed."""

    def backtest(self, data, symbol, initial_cash=10000, prepared=None, close_open_at_end=False, **params):
        r = self._res("long")
        r.trade_details = []
        if close_open_at_end:
            r.trade_details = [dict(entry_date=data.index[-5], exit_date=data.index[-1], entry_price=100.0,
                                    exit_price=101.0, side="long", exit_reason="window_end")]
        return r


def _flat_data():
    idx = pd.bdate_range("2023-01-02", periods=400)
    return {"X": pd.DataFrame({"close": 100 + np.arange(400) * 0.1}, index=idx)}


def test_open_trade_at_window_end_emitted():
    data = _flat_data()
    cands = pb.generate_candidates_walk_forward(["X"], data, OpenAtEndStrategy, 6, 2, 2, CFG)
    assert cands and all(c["exit_reason"] == "window_end" for c in cands)
    wins = pb._windows(data["X"].index[0].to_pydatetime(), data["X"].index[-1].to_pydatetime(), 6, 2, 2)
    for c, w in zip(sorted(cands, key=lambda c: c["entry_date"]), wins):
        last_bar = data["X"].index[data["X"].index <= pd.Timestamp(w[3]) - pd.Timedelta(days=1)].max()
        assert c["exit_date"] == last_bar
    off = pb.generate_candidates_walk_forward(["X"], data, OpenAtEndStrategy, 6, 2, 2, CFG,
                                              close_open_at_end=False)
    assert off == []


def test_no_same_symbol_overlap_across_windows():
    data = _flat_data()
    # step < oos => overlapping OOS windows
    cands = pb.generate_candidates_walk_forward(["X"], data, OpenAtEndStrategy, 6, 3, 1, CFG)
    cs = sorted(cands, key=lambda c: c["entry_date"])
    assert len(cs) > 1
    for a, b in zip(cs, cs[1:]):
        assert a["exit_date"] <= b["entry_date"]
    r = sim(cs, cfg=CFG)
    assert not (r.trades_skipped["reason"] == "duplicate_symbol").any() if len(r.trades_skipped) else True
