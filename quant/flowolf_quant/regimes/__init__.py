"""Regime detection: volatility (GARCH), trend (TSMOM), change points (BOCPD).

Every signal here is computed from data up to and including day t. The
backtester (``flowolf_quant.backtest.run``) applies weights one day later;
do not shift outputs yourself. See docs/regimes.md.
"""

from .changepoint import BocpdResult, bocpd, detect_changepoints
from .garch import (REGIME_NAMES, GarchFit, apply_garch, fit_garch, forecast_vol,
                    simulate_garch, vol_regimes, walk_forward_vol_regimes)
from .tsmom import ex_ante_vol, trend_regime, tsmom_signal, tsmom_weights

__all__ = [
    "GarchFit", "fit_garch", "apply_garch", "forecast_vol", "simulate_garch",
    "vol_regimes", "walk_forward_vol_regimes", "REGIME_NAMES",
    "tsmom_signal", "tsmom_weights", "trend_regime", "ex_ante_vol",
    "BocpdResult", "bocpd", "detect_changepoints",
]
