"""Validation gate: tests for luck and overfitting that every strategy must pass."""

from .cv import cv_select_and_score, purged_kfold
from .gate import GateConfig, Verdict, evaluate
from .holdout import HoldoutLedger, split_holdout
from .pbo import pbo_cscv
from .sharpe import (deflated_sharpe, expected_max_sharpe, min_track_record_length,
                     probabilistic_sharpe)
from .trials import TrialRegistry
from .walkforward import walk_forward

__all__ = ["GateConfig", "Verdict", "evaluate", "pbo_cscv", "deflated_sharpe",
           "probabilistic_sharpe", "expected_max_sharpe", "min_track_record_length",
           "purged_kfold", "cv_select_and_score", "walk_forward", "split_holdout",
           "HoldoutLedger", "TrialRegistry"]
