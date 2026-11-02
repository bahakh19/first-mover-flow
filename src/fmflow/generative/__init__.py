"""Generative scenario models for stress-testing a strategy's path risk (optional extra: pip install "fmflow[gen]").

Small samples make path statistics such as the maximum drawdown and the Calmar ratio extremely noisy: one five-year
history contains one or two drawdown episodes. Generative models trained on the JOINT daily returns of the strategy and
of market factors produce many plausible histories that keep the fat tails, volatility clustering and the nonlinear
co-movement of the strategy with the market, which block bootstraps only partly preserve and Gaussian models miss.

    windows    standardised training windows, pooling several samples (e.g. development + a second universe)
    baselines  stationary block bootstrap and filtered historical simulation (GJR-GARCH-t, joint residual resampling)
    ddpm       denoising diffusion probabilistic model (Ho et al., 2020) with an MLP denoiser; unconditional sampling and
               CONDITIONAL scenarios by inpainting (RePaint, Lugmayr et al., 2022): fix the market path, sample the strategy
    timegan    TimeGAN (Yoon, Jarrett and van der Schaar, 2019): embedding, supervised and adversarial losses on GRUs
    fidelity   marginal moments, autocorrelation of |r|, cross-correlation, tail dependence, drawdown distribution,
               discriminative score, train-on-synthetic-test-on-real predictive score, memorisation check
    stress     path-level statistics (Sharpe, max drawdown, Calmar) over simulated horizons; conditional stress tables

`ddpm` and `timegan` need JAX (CPU is enough); `baselines`, `fidelity` and `stress` need only numpy / scipy / sklearn.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def jax_status() -> tuple[bool, str]:
    """(usable, message) for the JAX backend of `ddpm`, `diffusion_fhs` and `timegan`.

    JAX can be installed yet unusable: recent jaxlib wheels need a CPU with AVX instructions and raise a RuntimeError
    at import on older processors. Everything else in `fmflow.generative` (bootstrap, FHS, fidelity, stress) works
    without JAX, so callers check this first and fall back instead of failing."""
    try:
        import jax
        import jax.numpy as jnp
        float(jnp.ones(2).sum())                          # forces backend initialisation
        return True, f"jax {jax.__version__} on {jax.default_backend()}"
    except ImportError:
        import platform
        if platform.system() == "Darwin" and platform.machine() == "x86_64":
            return False, "JAX has no wheels for Intel Macs (bootstrap and FHS generators work without it)"
        return False, 'JAX is not installed: pip install "fmflow[gen]" (bootstrap and FHS generators work without it)'
    except Exception as e:                                # e.g. jaxlib built with AVX on a CPU without AVX
        return False, f"JAX is installed but cannot run on this machine ({type(e).__name__}: {str(e).splitlines()[0]})"


def jax_available() -> bool:
    """True if the JAX backend can run on this machine (see `jax_status`)."""
    return jax_status()[0]

