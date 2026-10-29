# Section 17.2 — Objections 2-4: "Execution at the signal close is optimistic", "shorts are not free", "the Sharpe is
# inflated by smoothed (autocorrelated) P&L", "zero beta on average hides crash beta".
from fmflow.backtest.engine import backtest, breakeven_cost_bps, cost_sensitivity

IMPL = {}
for nm, o, W in (("dev", DEV_OUT, W_AUD), ("oos", OOS_OUT, W_OOS)):
    w_full = list(W.values())[-1]
    grid = cost_sensitivity(o.weights, o.returns, cost_grid=(0, 5, 10, 20, 30), borrow_grid=(0, 50, 100, 200),
                            execution_lags=(0, 1, 2), window=w_full)
    IMPL[f"{nm} grid"] = grid
    res = backtest(o.weights, o.returns, o.config.cost_bps)
    lo = lo_adjusted_sharpe(o.net_returns.loc[w_full])
    cb = conditional_betas(o.net_returns.loc[w_full], o.returns.mean(axis=1).loc[w_full])
    IMPL[nm] = {"breakeven cost (bps per unit traded)": breakeven_cost_bps(res, w_full.start, w_full.stop),
                "Sharpe, 1-day execution delay": float(grid.query("execution_lag == 1 and borrow_bps == 0 and cost_bps == 5")["Sharpe"].iloc[0]),
                "Sharpe, 1-day delay + 100 bp borrow + 10 bp cost": float(grid.query("execution_lag == 1 and borrow_bps == 100 and cost_bps == 10")["Sharpe"].iloc[0]),
                "Sharpe, 2-day delay + 200 bp borrow + 20 bp cost": float(grid.query("execution_lag == 2 and borrow_bps == 200 and cost_bps == 20")["Sharpe"].iloc[0]),
                **{f"Lo(2002) {k}": v for k, v in lo.items()}, **cb}
    print(f"\n{nm}: Sharpe by cost (rows) and borrow fee (columns), execution delay 0 / 1 session")
    for lag in (0, 1):
        display(grid.query("execution_lag == @lag").pivot(index="cost_bps", columns="borrow_bps", values="Sharpe").round(2))
IMPL_TBL = pd.DataFrame({k: v for k, v in IMPL.items() if not k.endswith("grid")})
AUD["IMPL"] = IMPL
display(IMPL_TBL.round(3))
