import math
import statistics

import numpy as np

from squeeze_scanner import (
    Candle, SqueezeConfig, compute_squeeze, ema, last_closed_open_time, rolling_max, rolling_min, rolling_std, sma,
    true_range,
)
from synth import candles_for

X = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0], dtype=float)


def test_sma_matches_loop():
    got = sma(X, 4)
    assert np.isnan(got[:3]).all()
    for i in range(3, len(X)):
        assert math.isclose(got[i], X[i - 3:i + 1].mean())


def test_rolling_std_is_population():
    got = rolling_std(X, 5)
    for i in range(4, len(X)):
        assert math.isclose(got[i], statistics.pstdev(X[i - 4:i + 1]))
    # textbook check: 1..20 → σ = sqrt((n²−1)/12)
    assert math.isclose(rolling_std(np.arange(1.0, 21.0), 20)[-1], math.sqrt(399 / 12))


def test_ema_seeded_with_sma():
    got = ema(X, 4)
    assert np.isnan(got[:3]).all()
    a = 2 / 5
    ref = X[:4].mean()
    assert math.isclose(got[3], ref)
    for i in range(4, len(X)):
        ref = a * X[i] + (1 - a) * ref
        assert math.isclose(got[i], ref)


def test_true_range_and_rolling_extrema():
    h = np.array([10.0, 12.0, 11.0, 15.0])
    l = np.array([8.0, 9.0, 10.5, 13.0])
    c = np.array([9.0, 11.5, 10.8, 14.0])
    tr = true_range(h, l, c)
    assert tr[0] == 2.0
    assert tr[1] == max(12 - 9, abs(12 - 9), abs(9 - 9))
    assert tr[3] == max(15 - 13, abs(15 - 10.8), abs(13 - 10.8))
    assert rolling_max(h, 2)[-1] == 15.0 and rolling_min(l, 3)[-1] == 9.0


def test_last_closed_open_time():
    t = 1_700_003_600_123  # arbitrary wall time
    assert last_closed_open_time("1h", t) == (t // 3_600_000) * 3_600_000 - 3_600_000
    assert last_closed_open_time("4h", t) % 14_400_000 == 0
    assert last_closed_open_time("4h", t) < t - 14_400_000 + 1


def _bars(coin: str, tf: str = "1h") -> list:
    end = 1_760_000_000_000
    raw = candles_for(coin, tf, 0, end)
    closed = [Candle.from_api(d) for d in raw if d["t"] <= last_closed_open_time(tf, end)]
    return closed


def test_squeeze_regimes():
    cfg = SqueezeConfig()
    trend = compute_squeeze("1h", _bars("BBB"), cfg, 20)
    top = compute_squeeze("1h", _bars("AAA"), cfg, 20)
    rel = compute_squeeze("1h", _bars("GGG"), cfg, 20)

    assert trend is not None and not trend.squeeze_on and trend.squeeze_ratio > 1.0 and not trend.released

    assert top is not None and top.squeeze_on and top.squeeze_ratio < 1.0
    assert top.bb_upper < top.kc_upper and top.bb_lower > top.kc_lower
    assert top.squeeze_bars >= 3 and top.range_pos >= 0.75 and not top.released
    assert 0.0 <= top.bbw_pctile <= 0.5  # current width ranks among the tightest in the lookback

    assert rel is not None and rel.released and not rel.squeeze_on and rel.release_dir == 1
    assert rel.prev_squeeze_bars >= 3 and rel.squeeze_bars == 0
    assert rel.close > rel.bb_upper or rel.close > rel.kc_upper


def test_squeeze_needs_history():
    bars = _bars("AAA")[:15]
    assert compute_squeeze("1h", bars, SqueezeConfig(), 20) is None
