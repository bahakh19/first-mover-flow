"""Command-line interface (`fmflow --help`, `fmflow <command> --help`).

    fmflow doctor                                                         # check the environment
    fmflow validate --prices P.csv --ratings R.csv
    fmflow run      --prices P.csv --ratings R.csv [--config C.json] [--leg-fraction 0.4] [--out results/]
    fmflow compare  --prices P.csv --ratings R.csv [--config ...] [--out results/compare/]
    fmflow synth    --preset development|open|null [--seed 7] [--out data/synthetic/]
    fmflow describe --prices P.csv --ratings R.csv
    fmflow evaluate --prices P.csv --ratings R.csv [--config ...] [--placebo] [--loo] [--out results/evaluate/]
    fmflow register --config configs/fmf_n.json --title "..." --out configs/my_test.json
    fmflow infer    --prices P.csv --ratings R.csv [--config ...]          # Bayesian Sharpe, SPA / Romano-Wolf vs comparables
    fmflow multiverse --prices P.csv --ratings R.csv [--n-random 200]     # specification curve + SPA + PBO over the family
    fmflow book     --prices P.csv --ratings R.csv [--nav 1e7] [--prices-ccy C.csv] [--holdings H.csv]
    fmflow index    --prices P.csv --ratings R.csv [--fee-bps 50] [--vol-target 0.1] [--out results/index/]
    fmflow stress   --prices P.csv --ratings R.csv [--generator fhs|bootstrap|diffusion] [--paths 1000]

`python -m fmflow` is equivalent to `fmflow`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def _cfg(args):
    from fmflow.config import StrategyConfig
    cfg = StrategyConfig.load(args.config) if getattr(args, "config", None) else StrategyConfig.fmf_n()
    lf, alloc = getattr(args, "leg_fraction", None), getattr(args, "allocator", None)
    if lf is not None and (cfg.k is not None or cfg.leg_fraction != float(lf)):
        cfg = cfg.with_(k=None, leg_fraction=float(lf), name=f"{cfg.name} (leg fraction {float(lf):g})")
    if alloc and alloc != cfg.allocator:
        cfg = cfg.with_(allocator=alloc, name=f"{cfg.name} ({alloc})")
    cfg.validate()
    return cfg


def _outdir(args) -> Path | None:
    """The --out directory (created), or None when the command only prints."""
    if not getattr(args, "out", None):
        return None
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    return od


def _load(args):
    from fmflow.data.io import load_prices, load_ratings
    prices = load_prices(args.prices)
    return prices, load_ratings(args.ratings, prices)


def cmd_validate(args) -> int:
    """Structural contract checks, then the cleaning rules, then the minimum-history requirements of DATA.md."""
    from fmflow.config import StrategyConfig
    from fmflow.pipeline import prepare_data
    p, r = _load(args)
    print(f"Contract: OK (prices {p.shape[0]:,} rows, {p['stock'].nunique()} stocks; ratings {r.shape[0]:,} rows, "
          f"{r['analyst'].nunique()} analysts)")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)     # the empty-ratings case is reported below as an error
        px, rt, _, log = prepare_data(p, r, StrategyConfig.fmf_n())
    audit = log.to_frame()
    acted = audit[audit["n_rows"] > 0]
    print(f"Cleaning: {len(px):,} session{'s' * (len(px) != 1)} ({px.index[0]:%Y-%m-%d} to {px.index[-1]:%Y-%m-%d}), {px.shape[1]} stocks, "
          f"{len(rt):,} accepted ratings" + ("; rules that removed or changed rows:" if len(acted) else "; nothing removed"))
    for _, row in acted.iterrows():
        print(f"  {row['check']:<6}{row['finding']}: {row['n_rows']:,} ({row['decision']})")
    problems, warnings_ = [], []
    if rt.empty:
        problems.append("no rating survived cleaning: every analyst signal would be NaN")
    if len(px) < 2:
        problems.append("fewer than two sessions: no return can be computed")
    elif len(px) <= 252:
        warnings_.append(f"{len(px)} sessions: the signals need about 252 sessions of history, so nothing can be evaluated")
    elif len(px) < 252 + 126:
        warnings_.append(f"{len(px)} sessions: after the 252-session warm-up only {len(px) - 252} remain for evaluation")
    covered = rt.groupby("stock")["analyst"].nunique()
    thin = sorted(set(px.columns) - set(covered[covered >= 3].index), key=str)
    if thin and not rt.empty:
        warnings_.append(f"{len(thin)} of {px.shape[1]} stocks have fewer than 3 analysts and rarely get a breadth score: "
                         + ", ".join(map(str, thin[:8])) + (" ..." if len(thin) > 8 else ""))
    for m in problems:
        print(f"Error: {m}")
    for m in warnings_:
        print(f"Warning: {m}")
    if problems:
        print("Not usable: fix the errors above (DATA.md describes the contract).")
        return 1
    print("Ready to run." if not warnings_ else "Usable, with the warnings above.")
    return 0


def cmd_run(args) -> int:
    from fmflow.pipeline import run_strategy
    from fmflow.report import default_windows, strategy_report
    cfg = _cfg(args)
    p, r = _load(args)
    out = run_strategy(p, r, cfg, cache_dir=args.cache_dir)
    od = _outdir(args) or Path("results")
    od.mkdir(parents=True, exist_ok=True)
    out.weights.to_csv(od / "weights.csv")
    pd.DataFrame({"gross": out.result.gross, "costs": out.result.costs, "net": out.result.net}).to_csv(od / "returns.csv")
    out.audit.to_csv(od / "cleaning_log.csv", index=False)
    cfg.save(od / "config_used.json")
    rep = strategy_report(out, default_windows(out.prices.index))
    rep.to_csv(od / "summary.csv")
    print(rep.round(3).to_string())
    files = ["weights.csv", "returns.csv", "summary.csv", "cleaning_log.csv", "config_used.json"]
    try:
        import matplotlib
        matplotlib.use("Agg")                       # files only: never open a window, works without a display
        import matplotlib.pyplot as plt

        from fmflow.plotting import equity_curves
        w = default_windows(out.prices.index)["evaluation"]
        fig = equity_curves({cfg.name: out.net_returns, "Equal-weight market": out.returns.mean(axis=1)}, w,
                            f"{cfg.name}: net of {cfg.cost_bps:g} bp", str(od / "equity.png"))
        plt.close(fig)
        files.append("equity.png")
    except ImportError:                             # matplotlib is optional (pip install "fmflow[plot]")
        pass
    print(f"\nwrote {', '.join(files)} to {od}")
    return 0


def cmd_compare(args) -> int:
    from fmflow.benchmarks import comparable_books
    from fmflow.pipeline import run_strategy
    from fmflow.report import default_windows, summary_table
    cfg = _cfg(args)
    p, r = _load(args)
    out = run_strategy(p, r, cfg, cache_dir=args.cache_dir)
    books = {cfg.name: out.net_returns, **comparable_books(out)}
    w = {"evaluation": default_windows(out.prices.index)["evaluation"]}
    tab = summary_table(books, w, out.returns.mean(axis=1), reference=out.net_returns).droplevel(1)
    tab = tab.sort_values("Sharpe", ascending=False)
    print(tab.round(2).to_string())
    od = _outdir(args)
    if od:
        tab.to_csv(od / "comparables.csv")
        print(f"\nwrote comparables.csv to {od}")
    return 0


def cmd_synth(args) -> int:
    from fmflow.synthetic import SyntheticConfig, generate
    preset = {"development": SyntheticConfig.development_like, "open": SyntheticConfig.open_universe_like,
              "null": SyntheticConfig.null}[args.preset]
    p, r = generate(preset(seed=args.seed))
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    p.to_csv(od / "prices.csv", index=False)
    r.to_csv(od / "ratings.csv", index=False)
    print(f"wrote prices.csv ({len(p):,} rows) and ratings.csv ({len(r):,} rows) to {od}")
    return 0


def cmd_describe(args) -> int:
    from fmflow.synthetic import describe_dataset
    p, r = _load(args)
    print(describe_dataset(p, r).round(3).to_string())
    return 0


def cmd_evaluate(args) -> int:
    """The paper's validation battery on any dataset: signal ICs, performance, comparables, placebos, leave-one-out."""
    from fmflow.benchmarks import comparable_books
    from fmflow.pipeline import run_strategy
    from fmflow.report import default_windows, strategy_report, summary_table
    from fmflow.robustness import leave_one_out, signal_ic_table, strategy_placebo
    cfg = _cfg(args)
    p, r = _load(args)
    out = run_strategy(p, r, cfg, cache_dir=args.cache_dir)
    win = default_windows(out.prices.index)
    od = _outdir(args) or Path("results/evaluate")
    od.mkdir(parents=True, exist_ok=True)
    tables = {"performance": strategy_report(out, win), "signal_ic21": signal_ic_table(out, 21, win["evaluation"])}
    books = {cfg.name: out.net_returns, **comparable_books(out)}
    tables["comparables"] = summary_table(books, {"evaluation": win["evaluation"]}, out.returns.mean(axis=1),
                                          reference=out.net_returns).droplevel(1).sort_values("Sharpe", ascending=False)
    if args.placebo:
        tables["placebo"] = strategy_placebo(out, {"evaluation": win["evaluation"]})
    if args.loo:
        tables["leave_one_out"] = leave_one_out(p, r, cfg, {"evaluation": win["evaluation"]}, sig_vol=out.sig_vol)
    for name, t in tables.items():
        t.to_csv(od / f"{name}.csv")
        print(f"\n== {name} ==\n{t.round(3).to_string()}")
    cfg.save(od / "config_used.json")
    print(f"\nwrote {', '.join(f'{t}.csv' for t in tables)} and config_used.json to {od}")
    return 0


def cmd_register(args) -> int:
    from fmflow.config import StrategyConfig
    from fmflow.prereg import register
    cfg = StrategyConfig.load(args.config) if args.config else StrategyConfig.fmf_n()
    doc = register({"title": args.title, "frozen": cfg.to_dict(), "decision_rule": args.rule}, args.out)
    print(f"registered {args.out}: sha256_16 = {doc['sha256_16']} at {doc['registered_utc']}")
    return 0


def _run(args):
    from fmflow.pipeline import run_strategy
    p, r = _load(args)
    return p, r, run_strategy(p, r, _cfg(args), cache_dir=getattr(args, "cache_dir", None))


def cmd_infer(args) -> int:
    from fmflow.analytics.bayes import bayesian_sharpe
    from fmflow.analytics.snooping import romano_wolf, spa_test
    from fmflow.benchmarks import comparable_books
    from fmflow.report import default_windows
    _, _, out = _run(args)
    w = default_windows(out.prices.index)["evaluation"]
    f = out.net_returns.loc[w]
    rows = {pr: bayesian_sharpe(f, prior_sr_sd=v).summary() for pr, v in (("flat", None), ("sceptical N(0,1)", 1.0),
                                                                          ("very sceptical N(0,0.5)", 0.5))}
    bayes = pd.DataFrame(rows).T
    print("Bayesian Sharpe ratio (posterior):\n", bayes.round(3).to_string())
    # each comparable rescaled to the strategy's volatility (comparables with zero volatility cannot be rescaled)
    scaled = {k: v.loc[w] / v.loc[w].std() * f.std() for k, v in comparable_books(out).items() if v.loc[w].std() > 0}
    D = pd.DataFrame({k: f - v for k, v in scaled.items()}).dropna()
    rw = romano_wolf(D)
    print("\nFMF minus each comparable (equal volatility), Romano-Wolf step-down:\n", rw.round(3).to_string())
    spa = spa_test(-D)
    print(f"\nSPA p-value, H0 'no comparable beats the strategy': {spa['p_value']:.3f}")
    od = _outdir(args)
    if od:
        bayes.to_csv(od / "bayesian_sharpe.csv")
        rw.to_csv(od / "romano_wolf.csv")
        pd.Series({"spa_p_value": spa["p_value"]}).to_csv(od / "spa.csv", header=["value"])
        print(f"\nwrote bayesian_sharpe.csv, romano_wolf.csv and spa.csv to {od}")
    return 0


def cmd_multiverse(args) -> int:
    from fmflow.analytics.inference import cscv_pbo
    from fmflow.analytics.snooping import spa_test
    from fmflow.report import default_windows
    from fmflow.robustness import multiverse
    p, r, out = _run(args)
    win = {"evaluation": default_windows(out.prices.index)["evaluation"]}
    grid = {"leader_gap": [5, 10, 15, 20], "herd_window": [(2, 10), (3, 15), (5, 20)], "lmh_window": [63, 126, 189, 252],
            "breadth_window": [42, 63, 126], "tranche_period": [21, 42, 63, 84], "allocator": ["EW", "IV-GJR"],
            "data.tp_expiry_days": [63, 126, 252], "data.event_min_ratings": [4, 5, 6]}
    tab, nets = multiverse(p, r, out.config, grid, win, sig_vol=out.sig_vol, mode="random", n_random=args.n_random, n_jobs=args.jobs)
    s = tab["Sharpe evaluation"].dropna()
    fam = nets.loc[win["evaluation"]].fillna(0.0)
    print(tab.sort_values("Sharpe evaluation", ascending=False).round(3).head(15).to_string())
    print(f"\n{len(s)} specifications: share Sharpe > 0 = {(s > 0).mean():.2f}, median {s.median():.2f}, "
          f"10th pct {s.quantile(0.1):.2f}; SPA p = {spa_test(fam)['p_value']:.3f}; PBO = {cscv_pbo(fam, 12)['PBO']:.2f}")
    od = _outdir(args)
    if od:
        tab.to_csv(od / "multiverse.csv")
        nets.to_csv(od / "multiverse_net_returns.csv")
        print(f"\nwrote multiverse.csv and multiverse_net_returns.csv to {od}")
    return 0


def cmd_book(args) -> int:
    from fmflow.data.io import read_table
    from fmflow.production import latest_book, risk_report, trade_list
    if args.holdings and not args.prices_ccy:
        raise ValueError("--holdings needs --prices-ccy (currency prices are required to turn weights into shares)")
    _, _, out = _run(args)
    b = latest_book(out)
    risk = risk_report(out)
    print(f"Book on {b.attrs['date'].date()}:\n", b.round(4).to_string())
    print("\nRisk report:\n", risk.to_string())
    trades = None
    if args.prices_ccy:
        def series(path: str) -> pd.Series:          # first column = stock id, second = value
            t = read_table(path)
            from fmflow.data.cleaning import to_float
            return pd.Series(to_float(t.iloc[:, 1]).to_numpy(),
                             index=t.iloc[:, 0].astype(str).str.strip())
        cur = series(args.holdings) if args.holdings else None
        trades = trade_list(b["held_weight"], cur, series(args.prices_ccy), args.nav, args.lot)
        print("\nTrades:\n", trades.to_string())
    od = _outdir(args)
    if od:
        b.to_csv(od / "book.csv")
        risk.to_csv(od / "risk_report.csv", header=["value"])
        written = ["book.csv", "risk_report.csv"]
        if trades is not None:
            trades.to_csv(od / "trades.csv")
            written.append("trades.csv")
        print(f"\nwrote {', '.join(written)} to {od}")
    return 0


def cmd_index(args) -> int:
    from fmflow.index import IndexRules, compute_index, index_factsheet, rulebook_markdown
    _, _, out = _run(args)
    rules = IndexRules(name=args.name, fee_bps=args.fee_bps, replication_cost_bps=args.cost_bps, vol_target=args.vol_target)
    idx = compute_index(out, rules)
    fs = index_factsheet(idx["level"])
    print(fs.round(4).to_string())
    od = _outdir(args) or Path("results/index")
    od.mkdir(parents=True, exist_ok=True)
    idx.to_csv(od / "index_levels.csv")
    fs.to_csv(od / "factsheet.csv")
    (od / "RULEBOOK.md").write_text(rulebook_markdown(rules, out.config), encoding="utf-8")
    print(f"\nwrote index_levels.csv, factsheet.csv and RULEBOOK.md to {od}")
    return 0


def cmd_stress(args) -> int:
    from fmflow.benchmarks import comparable_books
    from fmflow.generative import stress as GS
    from fmflow.generative.baselines import block_bootstrap_paths, fhs_paths
    from fmflow.report import default_windows
    _, _, out = _run(args)
    w = default_windows(out.prices.index)["evaluation"]
    mom = comparable_books(out, include_allocations=False)["12-1 momentum (Jegadeesh-Titman)"]
    fr = pd.DataFrame({"FMF": out.net_returns, "MKT": out.returns.mean(axis=1), "MOM": mom}).loc[w].dropna()
    if args.generator == "bootstrap":
        P = block_bootstrap_paths(fr, args.paths, args.horizon)
    elif args.generator == "fhs":
        P = fhs_paths(fr, args.paths, args.horizon)
    else:
        from fmflow.generative import jax_status
        ok, msg = jax_status()
        if not ok:
            print(f"diffusion generator unavailable: {msg}\nUse --generator fhs or --generator bootstrap.")
            return 2
        from fmflow.generative.diffusion_fhs import DiffusionFHS
        P = DiffusionFHS(list(fr.columns)).fit([fr]).simulate(args.paths, args.horizon)
    summary = GS.summarise(GS.path_stats(P))
    print(f"{args.generator}: {args.paths} paths x {args.horizon} sessions (drift as in the sample)")
    print(summary.round(3).to_string())
    od = _outdir(args)
    if od:
        summary.to_csv(od / f"stress_{args.generator}.csv")
        print(f"\nwrote stress_{args.generator}.csv to {od}")
    return 0


def cmd_doctor(args) -> int:
    """Check the environment: versions, the imports that break in mixed conda/pip installs, JAX, data files."""
    import importlib
    import platform
    import warnings
    rows, failed = [], False

    def check(name, fn, required=True, fix="", missing="pip install -e ."):
        """ok / FAIL (required component broken or missing) / warn (optional component broken) / opt (not installed)."""
        nonlocal failed
        try:
            rows.append(("ok", name, fn()))
        except ModuleNotFoundError as e:
            failed |= required
            rows.append(("FAIL" if required else "opt", name, f"not installed -> {missing}" if missing else str(e)))
        except Exception as e:  # noqa: BLE001 -- the point is to report anything
            failed |= required
            rows.append(("FAIL" if required else "warn", name, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
                         + (f"  -> {fix}" if fix else "")))

    from fmflow import __version__
    rows.append(("ok", "python", f"{platform.python_version()} on {platform.system()} {platform.machine()} ({sys.executable})"))
    rows.append(("ok", "fmflow", f"{__version__} ({Path(__file__).resolve().parent})"))
    for mod in ("numpy", "pandas", "scipy"):
        check(mod, lambda m=mod: importlib.import_module(m).__version__)

    def _sm():
        import statsmodels
        import statsmodels.api  # noqa: F401 -- fails with scipy >= 1.16 and statsmodels < 0.14.5
        return statsmodels.__version__
    check("statsmodels", _sm, fix='pip install -U "statsmodels>=0.14.5" (older releases import a function SciPy removed)')

    def _arch():
        import arch
        import numpy as np
        from arch import arch_model
        x = np.random.default_rng(0).standard_t(5, 400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arch_model(x, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off", show_warning=False)
        return arch.__version__
    check("arch (GJR-GARCH fit)", _arch, fix="pip install -U arch")

    def _mpl():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401 -- fails when matplotlib was compiled against NumPy 1.x
        return matplotlib.__version__
    check("matplotlib", _mpl, required=False, missing='optional, for figures: pip install "fmflow[plot]"',
          fix="matplotlib is compiled against another NumPy: pip install -U --force-reinstall matplotlib, or use a fresh venv")

    def _jax():
        from fmflow.generative import jax_status
        ok, msg = jax_status()
        if not ok:                                  # not installed (optional) or installed but unusable (warn)
            raise (ModuleNotFoundError if "not installed" in msg or "no wheels" in msg else RuntimeError)(msg)
        return msg
    check("jax (diffusion, TimeGAN)", _jax, required=False, missing="",
          fix="optional; bootstrap and FHS generators, and everything else, work without it")
    check("scikit-learn (fidelity tests)", lambda: importlib.import_module("sklearn").__version__, required=False,
          missing='optional: pip install "fmflow[gen]"')
    check("openpyxl, pyarrow (Excel, Parquet)",
          lambda: f"openpyxl {importlib.import_module('openpyxl').__version__}, pyarrow {importlib.import_module('pyarrow').__version__}",
          required=False, missing='optional, for Excel and Parquet input: pip install "fmflow[io]"')
    check("jupyterlab (notebooks)", lambda: importlib.import_module("jupyterlab").__version__, required=False,
          missing='optional: pip install "fmflow[notebooks]"')
    root = Path(args.root)
    in_repo = (root / "pyproject.toml").exists() and (root / "data").is_dir()
    for f in ("data/open/MarketData_Normalized.csv", "data/open/Ratings_Normalized.csv",
              "data/private/MarketData.csv", "data/private/Ratings.csv"):
        if (root / f).exists():
            rows.append(("ok", f, "present"))
        elif "private" in f:
            rows.append(("info", f, "not distributed (proprietary)"))
        else:
            rows.append(("info", f, "missing" if in_repo else
                         f"not found under {root.resolve()} (run from the repository root, or pass --root)"))
    w = max(len(r[1]) for r in rows)
    for st, name, info in rows:
        print(f"[{st:>4}] {name:<{w}}  {info}")
    print("\nAll required components work." + (" Optional ones marked 'opt' can be added later." if any(r[0] == "opt" for r in rows)
                                                 else "") if not failed else "\nSome required components fail: see the fixes above.")
    return 1 if failed else 0


def _robust_console() -> None:
    """Never crash on a character the console cannot encode (Windows code pages, redirected output)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


# (name, function, one-line help) of the subcommands that read the two input tables
_DATA_COMMANDS = (
    ("validate", cmd_validate, "check that the two input tables satisfy the data contract"),
    ("describe", cmd_describe, "summary moments of a dataset (coverage, revision mix, correlations)"),
    ("run", cmd_run, "run the strategy; write weights, returns, summary, cleaning log and equity curve"),
    ("compare", cmd_compare, "compare the strategy with 13 institutional signals and 3 allocation benchmarks"),
    ("evaluate", cmd_evaluate, "the validation battery: ICs, performance, comparables (+ --placebo, --loo)"),
    ("infer", cmd_infer, "Bayesian Sharpe ratio and snooping-robust tests against the comparables"),
    ("multiverse", cmd_multiverse, "specification curve over reasonable settings, with SPA and PBO"),
    ("book", cmd_book, "today's book, risk report and (with --prices-ccy) the trade list"),
    ("index", cmd_index, "QIS index levels, factsheet and a generated rulebook"),
    ("stress", cmd_stress, "generative stress test (bootstrap, FHS or Diffusion-FHS paths)"),
)
_DEFAULT_OUT = {"run": "results", "evaluate": "results/evaluate", "index": "results/index"}
_EPILOG = """examples:
  fmflow doctor
  fmflow run --prices data/open/MarketData_Normalized.csv --ratings data/open/Ratings_Normalized.csv --leg-fraction 0.4
  fmflow evaluate --prices P.csv --ratings R.csv --placebo --out results/eval
  fmflow synth --preset open --out data/synthetic

Input files may be CSV (any delimiter, decimal comma, gzip), Excel or Parquet. See DATA.md for the contract.
Set FMFLOW_DEBUG=1 (or pass --debug) to see full tracebacks."""


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the `fmflow` command (exposed for documentation and tests)."""
    from fmflow import __version__
    ap = argparse.ArgumentParser(prog="fmflow", description="First-Mover Flow strategy toolkit",
                                 epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"fmflow {__version__}")
    ap.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    sub = ap.add_subparsers(dest="cmd", metavar="<command>", title="commands")
    for name, fn, text in _DATA_COMMANDS:
        sp = sub.add_parser(name, help=text, description=text[0].upper() + text[1:] + ".")
        sp.add_argument("--prices", required=True, help="price table: date, stock, price")
        sp.add_argument("--ratings", required=True, help="ratings table: date, estimate_value, stock, analyst")
        if name not in ("validate", "describe"):
            sp.add_argument("--config", help="StrategyConfig JSON (default: FMF-N)")
            sp.add_argument("--leg-fraction", dest="leg_fraction", type=float,
                            help="legs = this share of the names with a score (e.g. 0.4) instead of k names")
            sp.add_argument("--allocator", choices=["EW", "IV-GJR"], help="weights within each leg")
            sp.add_argument("--cache-dir", dest="cache_dir", default=".fmflow_cache",
                            help="directory of the GARCH forecast cache (default: %(default)s)")
            sp.add_argument("--out", default=_DEFAULT_OUT.get(name),
                            help="output directory" + (" (default: %(default)s)" if name in _DEFAULT_OUT else
                                                       " (default: print only)"))
        if name == "multiverse":
            sp.add_argument("--n-random", dest="n_random", type=int, default=200, help="random specifications (default: 200)")
            sp.add_argument("--jobs", type=int, default=1, help="parallel processes (default: 1)")
        if name == "book":
            sp.add_argument("--nav", type=float, default=1e7, help="net asset value in currency (default: 1e7)")
            sp.add_argument("--lot", type=int, default=1, help="round share quantities to this lot size")
            sp.add_argument("--prices-ccy", dest="prices_ccy", help="CSV stock,price in currency (for share quantities)")
            sp.add_argument("--holdings", help="CSV stock,shares currently held (needs --prices-ccy)")
        if name == "index":
            sp.add_argument("--name", default="FMF-N Excess Return Index")
            sp.add_argument("--fee-bps", dest="fee_bps", type=float, default=0.0, help="annual index fee (bp)")
            sp.add_argument("--cost-bps", dest="cost_bps", type=float, default=5.0, help="replication cost per unit traded (bp)")
            sp.add_argument("--vol-target", dest="vol_target", type=float, default=None, help="annualised, e.g. 0.10")
        if name == "stress":
            sp.add_argument("--generator", choices=["bootstrap", "fhs", "diffusion"], default="fhs")
            sp.add_argument("--paths", type=int, default=1000)
            sp.add_argument("--horizon", type=int, default=1260, help="sessions per path (default: 1260, five years)")
        if name == "evaluate":
            sp.add_argument("--placebo", action="store_true", help="IID / permutation / time-shift nulls (slow)")
            sp.add_argument("--loo", action="store_true", help="leave-one-stock-out rebuilds (slow)")
        sp.set_defaults(func=fn)
    sp = sub.add_parser("synth", help="generate a realistic synthetic dataset (prices and ratings)",
                        description="Generate a realistic synthetic dataset (prices and ratings).")
    sp.add_argument("--preset", choices=["development", "open", "null"], default="development")
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--out", default="data/synthetic", help="output directory (default: %(default)s)")
    sp.set_defaults(func=cmd_synth)
    sp = sub.add_parser("register", help="freeze a configuration and fingerprint it before evaluation",
                        description="Freeze a configuration and fingerprint it before evaluation (see fmflow.prereg).")
    sp.add_argument("--config", help="StrategyConfig JSON (default: FMF-N)")
    sp.add_argument("--title", required=True)
    sp.add_argument("--rule", default="No parameter is changed after this file is hashed. Results are reported as they come out.")
    sp.add_argument("--out", required=True, help="registration file to write (never overwritten)")
    sp.set_defaults(func=cmd_register)
    sp = sub.add_parser("doctor", help="check the environment (versions, broken imports, JAX, data files)",
                        description="Check the environment: versions, broken imports, JAX, data files.")
    sp.add_argument("--root", default=".", help="repository root (for the data-file checks)")
    sp.set_defaults(func=cmd_doctor)
    return ap


# errors caused by the user's input or environment: reported in one line instead of a traceback
_USER_ERRORS = (ValueError, KeyError, FileNotFoundError, FileExistsError, IsADirectoryError, PermissionError,
                ImportError, RuntimeError)


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """Warnings as one readable line on stderr (no source path or code line) when running as a command."""
    print(f"warning: {message}", file=sys.stderr)


def main(argv=None) -> int:
    """Entry point of the `fmflow` command; returns the exit code (0 ok, 1 check failed, 2 input error)."""
    import warnings
    _robust_console()
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 0
    previous = warnings.showwarning
    warnings.showwarning = _show_warning
    try:
        return int(args.func(args) or 0)
    except _USER_ERRORS as e:
        if args.debug or os.environ.get("FMFLOW_DEBUG"):
            raise
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"fmflow {args.cmd}: error: {msg}\n(run with --debug for the full traceback)", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:                          # output piped into `head` or a closed pager: not an error
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 0
    finally:
        warnings.showwarning = previous


if __name__ == "__main__":
    raise SystemExit(main())
