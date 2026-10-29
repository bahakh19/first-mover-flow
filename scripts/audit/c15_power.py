# Section 17.5 — Objection 7: "With 10 names you cannot detect anything, so a positive result is luck and a negative one
# is uninformative." Power analysis, analytic and by Monte Carlo on the calibrated synthetic generator.
from fmflow.analytics.power import mde_ic, ic_power, simulate_power

MDE = pd.DataFrame({f"{y} years": {n: mde_ic(n, int(y * 252 / 21)) for n in (10, 20, 30, 50, 100, 500)} for y in (3, 5, 10)})
MDE.index.name = "stocks"
print("Minimum detectable mean IC (21-session horizon, one-sided 5%, 80% power):")
display(MDE.round(3))
print(f"Power to detect the observed LMH IC (~0.10): 10 stocks x 3 years = {ic_power(0.10, 10, 36):.2f}; "
      f"30 stocks x 2.6 years = {ic_power(0.10, 30, 31):.2f}; pooled evidence 3 samples ~ {ic_power(0.10, 17, 36 + 24 + 31):.2f}")
POWER = simulate_power(n_stocks=(10, 30), years=(3, 5), strengths=(0.0, 0.4), seeds=20)
display(POWER.round(3))
AUD["POWER"] = {"MDE": MDE, "sim": POWER}
