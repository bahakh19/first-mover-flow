"""Synthetic market + sell-side data with the statistical properties of the real datasets.

The generator is a structural model of how prices and analyst targets co-evolve:

prices   r_{i,t} = beta_i * m_t + g_i * s_{c(i),t} + mu_{i,t} + e_{i,t} + J_{i,t}
         m_t       market factor with GJR-GARCH(1,1)-t volatility (leverage effect, fat tails)
         s_c       sector factors (stocks are split into clusters)
         mu_{i,t}  latent expected idiosyncratic return  = strength * sigma_i * z_{i,t-1}
         z_{i,t}   latent fundamental state, AR(1) with a quarterly half-life, shocked at earnings
         J_{i,t}   earnings-day jumps (about 4x a normal day), partly in the direction of the z shock
analysts each stock is covered by n_i analysts drawn from a shared pool (so many analysts cover several stocks).
         Analysts publish around earnings (creating same-day clusters of >= 5 ratings) and in between.
         target = price * exp(optimism_a + info * z_{i,t} + noise)    -> targets chase price (excess revision ~ 1)
         FIRST MOVERS who publish after a quiet period see z early (informative); HERDERS revise 3-15 sessions after
         a cluster in the cluster's direction regardless of z (uninformative copycats); some ratings are reiterations.

`strength = 0` gives a NULL world (no predictable returns): the strategy should not find alpha there.
Default parameters reproduce the development dataset's moments (see `describe_dataset` and DATA.md):
average pairwise correlation ~0.40, annual volatility 26-53%, 17-72 analysts per stock, ~65 calendar days between an
analyst's updates, ~53% up / 17% down / 21% reiterations / 8% initiations, median target/price ~1.17.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    """Parameters of the structural synthetic market (see module docstring); presets via classmethods."""
    n_stocks: int = 10
    n_sessions: int = 1512
    start: str = "2016-01-04"
    n_sectors: int = 2
    vol_range: tuple = (0.26, 0.53)          # annualised total volatility per stock
    market_vol: float = 0.20
    market_drift: float = 0.25             # the development sample is a bull market (avg stock x6.8 in 6 years)
    sector_vol: float = 0.13
    stock_drift_sd: float = 0.15             # cross-sectional dispersion of annual drifts (winners and losers)
    strength: float = 0.15                   # predictability: daily expected idio return = strength * sigma_i,d * z
    z_half_life: int = 63
    earnings_every: int = 63
    earnings_jump_mult: float = 4.0
    analysts_per_stock: tuple = (17, 72)
    analyst_pool_ratio: float = 0.7          # pool size = ratio * sum(n_i): < 1 means multi-stock coverage
    p_publish_at_earnings: float = 0.30
    off_event_rate: float = 1 / 130           # per-session probability of an off-event publication
    p_reiterate: float = 0.05
    adjust_speed: float = 0.6                # partial adjustment of targets towards price * optimism
    p_sticky_down: float = 0.45              # reluctance to cut: a planned cut becomes a reiteration with this probability
    p_herd: float = 0.25                     # probability that a non-event analyst copies a recent cluster
    optimism_mu: float = np.log(1.17)
    optimism_sd: float = 0.08
    info_weight: float = 0.06                # how much an informed analyst's target loads on z
    target_noise: float = 0.035
    p_late_start: float = 0.25               # share of analyst-stock pairs that start covering mid-sample
    p_early_stop: float = 0.20               # share of pairs that stop covering before the end
    inject_errors: bool = False              # fat-finger targets, missing analyst ids, text values, duplicates
    seed: int = 7

    @classmethod
    def null(cls, **kw) -> SyntheticConfig:
        """A world with NO cross-sectional predictability: no analyst information, no persistent drift dispersion and
        no market premium (with a premium, beta dispersion alone creates persistent winners and losers). Persistent drift
        differences make momentum - and hence price-chasing breadth - profitable, which is not the analyst channel."""
        return cls(**{**dict(strength=0.0, stock_drift_sd=0.0, market_drift=0.0), **kw})

    @classmethod
    def development_like(cls, **kw) -> SyntheticConfig:
        """10 large caps, 6 years, dense coverage (the development dataset)."""
        return cls(**kw)

    @classmethod
    def open_universe_like(cls, **kw) -> SyntheticConfig:
        """30 stocks, ~3.6 years, sparse coverage by multi-stock analysts (the open out-of-sample dataset)."""
        base = dict(n_stocks=30, n_sessions=899, start="2023-01-04", n_sectors=4, vol_range=(0.20, 0.75), market_vol=0.24,
                    market_drift=0.30, sector_vol=0.16, stock_drift_sd=0.30, analysts_per_stock=(3, 27), analyst_pool_ratio=0.12,
                    p_publish_at_earnings=0.33, off_event_rate=1 / 320, p_sticky_down=0.70, p_late_start=0.35,
                    p_reiterate=0.06, adjust_speed=0.7, optimism_mu=float(np.log(1.15)))
        base.update(kw)
        return cls(**base)


def _gjr_t(n: int, vol_ann: float, rng: np.random.Generator) -> np.ndarray:
    alpha, gamma, beta, nu = 0.03, 0.12, 0.88, 5.0
    target_var = (vol_ann * 100 / np.sqrt(252)) ** 2
    omega = target_var * (1 - alpha - gamma / 2 - beta)
    s2, out = target_var, np.empty(n)
    for t in range(n):
        z = rng.standard_t(nu) / np.sqrt(nu / (nu - 2))
        e = np.sqrt(s2) * z
        out[t] = e
        s2 = omega + (alpha + gamma * (e < 0)) * e ** 2 + beta * s2
    return out / 100.0


def generate(cfg: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, ratings) long tables in the input contract of DATA.md."""
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_stocks, cfg.n_sessions
    bdays = pd.bdate_range(cfg.start, periods=int(T * 1.04) + 10)
    holidays = set(rng.choice(len(bdays), size=len(bdays) - T, replace=False))
    dates = pd.DatetimeIndex([d for i, d in enumerate(bdays) if i not in holidays])[:T]
    stocks = [f"stock_{i + 1}" for i in range(N)]

    # ---- returns -------------------------------------------------------------------------------------------------------
    m = _gjr_t(T, cfg.market_vol, rng) + cfg.market_drift / 252
    sectors = rng.integers(0, cfg.n_sectors, N)
    s = rng.normal(0, cfg.sector_vol / np.sqrt(252), (T, cfg.n_sectors))
    beta = rng.uniform(0.8, 1.3, N)
    tot_vol = rng.uniform(*cfg.vol_range, N) / np.sqrt(252)
    sys_var = (beta * cfg.market_vol / np.sqrt(252)) ** 2 + (cfg.sector_vol / np.sqrt(252)) ** 2
    idio = np.sqrt(np.maximum(tot_vol ** 2 - sys_var, (0.10 / np.sqrt(252)) ** 2))
    phi = 0.5 ** (1 / cfg.z_half_life)
    z = np.zeros((T, N))
    earn_off = rng.integers(0, cfg.earnings_every, N)
    is_earn = np.zeros((T, N), bool)
    for i in range(N):
        days = np.arange(earn_off[i], T, cfg.earnings_every) + rng.integers(-3, 4, size=len(range(earn_off[i], T, cfg.earnings_every)))
        is_earn[np.clip(days, 0, T - 1), i] = True
    shocks = np.zeros((T, N))
    for t in range(1, T):
        shock = rng.normal(0, 1, N) * np.where(is_earn[t], 1.0, 0.15)
        shocks[t] = shock
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * shock
    drift = rng.normal(cfg.market_drift, cfg.stock_drift_sd, N) / 252 - cfg.market_drift / 252
    mu = cfg.strength * idio * np.vstack([np.zeros(N), z[:-1]])
    e = rng.standard_t(5, (T, N)) / np.sqrt(5 / 3) * idio
    jump = np.where(is_earn, cfg.earnings_jump_mult * idio * (0.6 * np.sign(shocks) * np.abs(rng.standard_t(5, (T, N))) / np.sqrt(5 / 3)
                                                                + 0.4 * rng.standard_t(5, (T, N)) / np.sqrt(5 / 3)), 0.0)
    r = beta * m[:, None] + s[:, sectors] + drift + mu + e + jump
    r = np.clip(r, -0.6, 1.5)
    px = np.cumprod(1 + r, axis=0)
    px = px / px[0]

    # ---- analysts and ratings ---------------------------------------------------------------------------------------------
    n_cov = rng.integers(cfg.analysts_per_stock[0], cfg.analysts_per_stock[1] + 1, N)
    pool = max(int(cfg.analyst_pool_ratio * n_cov.sum()), int(n_cov.max()))
    coverage = {i: rng.choice(pool, size=min(n_cov[i], pool), replace=False) for i in range(N)}
    optimism = rng.normal(cfg.optimism_mu, cfg.optimism_sd, pool)
    rows = []
    for i in range(N):
        last_cluster_t, last_cluster_dir = -100, 0.0
        tp = {}
        quiet_since = 0
        start = {a: (int(rng.integers(0, int(0.8 * T))) if rng.random() < cfg.p_late_start else 0) for a in coverage[i]}
        stop = {a: (int(rng.integers(start[a] + 63, T)) if (rng.random() < cfg.p_early_stop and start[a] + 63 < T) else T) for a in coverage[i]}
        for t in range(T):
            active = [a for a in coverage[i] if start[a] <= t < stop[a]]
            pubs = []
            if is_earn[t, i]:
                pubs += [a for a in active if rng.random() < cfg.p_publish_at_earnings]
            off = [a for a in active if rng.random() < cfg.off_event_rate]
            pubs += off
            if not pubs:
                continue
            day_dir = []
            for a in pubs:
                informed = is_earn[t, i] or (t - quiet_since >= 10)
                herding = (not is_earn[t, i]) and 3 <= t - last_cluster_t <= 15 and rng.random() < cfg.p_herd
                prev = tp.get(a)
                if prev is not None and rng.random() < cfg.p_reiterate and not herding:
                    new = prev
                elif herding and prev is not None:
                    new = prev * np.exp(last_cluster_dir * abs(rng.normal(0.04, 0.02)))
                else:
                    info = cfg.info_weight * z[t, i] if informed else 0.0
                    anchor = np.log(px[t, i]) + optimism[a]
                    if prev is None:
                        new = np.exp(anchor + info + rng.normal(0, cfg.target_noise))
                    else:
                        step = cfg.adjust_speed * (anchor - np.log(prev)) + info + rng.normal(0, cfg.target_noise)
                        new = prev if (step < 0 and rng.random() < cfg.p_sticky_down) else prev * np.exp(step)
                tp[a] = new
                if prev is not None:
                    day_dir.append(np.sign(np.log(new / prev)))
                rows.append((dates[t], new, stocks[i], f"analyst_{a + 1}"))
            if len(pubs) >= 5:
                last_cluster_t, last_cluster_dir = t, float(np.sign(np.sum(day_dir))) if day_dir else 0.0
            quiet_since = t
    ratings = pd.DataFrame(rows, columns=["date", "estimate_value", "stock", "analyst"])
    prices = pd.DataFrame({"date": np.repeat(dates, N), "stock": np.tile(stocks, T), "price": px.ravel()})
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    ratings["date"] = pd.to_datetime(ratings["date"]).dt.strftime("%Y-%m-%d")
    ratings["estimate_value"] = ratings["estimate_value"].round(6)
    if cfg.inject_errors:
        k = max(1, len(ratings) // 1000)
        idx = rng.choice(len(ratings), size=4 * k, replace=False)
        ratings.loc[idx[:k], "estimate_value"] *= 1e6                       # fat-finger targets
        ratings.loc[idx[k:2 * k], "analyst"] = np.nan                        # missing analyst ids
        ratings["estimate_value"] = ratings["estimate_value"].astype(object)
        ratings.loc[idx[2 * k:3 * k], "estimate_value"] = "Not Available"    # text values
        ratings = pd.concat([ratings, ratings.iloc[idx[3 * k:]]], ignore_index=True)  # duplicates
    return prices, ratings


def describe_dataset(prices: pd.DataFrame, ratings: pd.DataFrame) -> pd.Series:
    """Moments used to calibrate the generator (computed identically on real and synthetic data)."""
    from fmflow.config import DataConfig
    from fmflow.data.cleaning import clean_prices, clean_ratings
    from fmflow.data.snapshot import classify_ratings
    px = clean_prices(prices, DataConfig())
    rt = clean_ratings(ratings.astype({"estimate_value": str}), px, DataConfig())
    ev = classify_ratings(rt, px.index, DataConfig())
    ret = px.pct_change(fill_method=None)
    C = ret.corr().values
    n = px.shape[1]
    kinds = ev["kind"].value_counts(normalize=True)
    gap = rt.sort_values("pub_date").groupby(["stock", "analyst"])["pub_date"].diff().dt.days
    return pd.Series({
        "stocks": n, "sessions": len(px), "avg pairwise corr": (C.sum() - n) / (n * (n - 1)),
        "ann vol min": ret.std().min() * np.sqrt(252), "ann vol max": ret.std().max() * np.sqrt(252),
        "ratings": len(rt), "analysts": rt["analyst"].nunique(),
        "analysts per stock (median)": rt.groupby("stock")["analyst"].nunique().median(),
        "stocks per analyst (median)": rt.groupby("analyst")["stock"].nunique().median(),
        "median days between an analyst's updates": gap.median(),
        "share up": kinds.get("up", 0), "share down": kinds.get("down", 0),
        "share reiteration": kinds.get("reiteration", 0), "share initiation": kinds.get("initiation", 0),
        "median target/price": rt["tp_ratio"].median(),
        "event clusters (>=5 same-day)": int(rt[rt["event_day"]].drop_duplicates(["stock", "pub_date"]).shape[0]),
    })
