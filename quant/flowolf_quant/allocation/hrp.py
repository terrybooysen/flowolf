"""Hierarchical Risk Parity and simple baselines.

Reference
---------
López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
Out of Sample." Journal of Portfolio Management 42(4), 59-69.

Steps (as in the paper):
1. Correlation distance d_ij = sqrt((1 - rho_ij) / 2).
2. Distance-of-distances: Euclidean distance between columns of d.
3. Single-linkage hierarchical clustering.
4. Quasi-diagonalisation: order assets by the dendrogram leaves so similar
   assets sit next to each other.
5. Recursive bisection (see ``hrp_weights(split=...)``): split the ordered list in halves; give each half
   weight in inverse proportion to its cluster variance, where cluster
   variance uses inverse-variance weights inside the cluster.

No matrix inversion is needed, which is why HRP is more stable than
mean-variance on noisy covariance estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
from scipy.spatial.distance import pdist

from ._rebalance import rolling_weights


def _clean_window(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Keep assets with complete data and non-zero variance in the window."""
    X = returns_window.dropna(axis=1, how="any")
    sd = X.std(ddof=1)
    return X.loc[:, sd > 0]


def equal_weights(returns_window: pd.DataFrame) -> pd.Series:
    """1/N across assets with complete data in the window.

    Reference: DeMiguel, Garlappi & Uppal (2009), RFS 22(5) - the 1/N
    benchmark that optimised portfolios often fail to beat out of sample.
    Known at t: uses only which assets have data in the window.
    """
    cols = _clean_window(returns_window).columns
    if len(cols) == 0:
        return pd.Series(0.0, index=returns_window.columns)
    w = pd.Series(1.0 / len(cols), index=cols)
    return w.reindex(returns_window.columns).fillna(0.0)


def inverse_vol_weights(returns_window: pd.DataFrame) -> pd.Series:
    """Weights proportional to 1 / volatility, long-only, summing to 1.

    Naive risk parity baseline (ignores correlations). Known at t: uses only
    the returns in the window passed in. Limits: two highly correlated low-vol
    assets get double the weight of a diversifying one.
    """
    X = _clean_window(returns_window)
    if X.shape[1] == 0:
        return pd.Series(0.0, index=returns_window.columns)
    iv = 1.0 / X.std(ddof=1)
    w = iv / iv.sum()
    return w.reindex(returns_window.columns).fillna(0.0)


def _ivp(cov: np.ndarray) -> np.ndarray:
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    w = _ivp(sub)
    return float(w @ sub @ w)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    w = np.ones(cov.shape[0])
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1.0 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    return w


def _tree_bisection(cov: np.ndarray, link: np.ndarray) -> np.ndarray:
    """Split at the dendrogram's own branches instead of list halves."""
    w = np.ones(cov.shape[0])
    stack = [to_tree(link)]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            continue
        left, right = node.get_left(), node.get_right()
        li, ri = left.pre_order(), right.pre_order()
        v_l, v_r = _cluster_var(cov, li), _cluster_var(cov, ri)
        alpha = 1.0 - v_l / (v_l + v_r)
        w[li] *= alpha
        w[ri] *= 1.0 - alpha
        stack += [left, right]
    return w


def hrp_weights(returns_window: pd.DataFrame, split: str = "tree") -> pd.Series:
    """Hierarchical Risk Parity weights for one estimation window.

    Reference: López de Prado (2016), JPM 42(4).

    Known at t: if ``returns_window`` ends at t, the weights use only data up
    to t. Assets with any missing value or zero variance in the window get 0.

    Returns a long-only Series summing to 1 (index = all input columns).

    ``split``: ``'bisect'`` is the paper's recursive bisection, which halves
    the quasi-diagonal *list* of assets at each step. ``'tree'`` (default)
    splits at the dendrogram's own branches, so a whole cluster is never cut
    in two. With unequal cluster sizes, list halving can hand most of the
    weight to the larger cluster (e.g. an 8+2 two-cluster panel gets roughly
    0.7-0.8 in the big cluster); tree splitting keeps clusters balanced by
    risk. Both use inverse-variance weights inside each cluster for the
    cluster variance.

    Limits: the covariance matrix is still estimated from the past window and
    correlations shift in crises; single linkage can chain clusters; the
    paper's list-halving split can cut across a cluster (see ``split``).
    HRP is a risk allocator - it has no view on expected returns.
    """
    X = _clean_window(returns_window)
    cols = list(X.columns)
    if len(cols) == 0:
        return pd.Series(0.0, index=returns_window.columns)
    if len(cols) == 1:
        return pd.Series(1.0, index=cols).reindex(returns_window.columns).fillna(0.0)
    cov = X.cov().to_numpy()
    corr = X.corr().to_numpy()
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    dd = pdist(dist, metric="euclidean")            # distance of distances
    link = linkage(dd, method="single")
    if split == "tree":
        w = _tree_bisection(cov, link)
    elif split == "bisect":
        w = _recursive_bisection(cov, leaves_list(link).tolist())
    else:
        raise ValueError("split must be 'tree' or 'bisect'")
    w = pd.Series(w, index=cols)
    w = w / w.sum()
    return w.reindex(returns_window.columns).fillna(0.0)


def rolling_hrp(asset_returns: pd.DataFrame, window: int = 252,
                rebalance_every: int = 21, method: str = "hrp",
                split: str = "tree") -> pd.DataFrame:
    """Target weights recomputed every ``rebalance_every`` days on a trailing window.

    ``method``: ``'hrp'`` (default), ``'ivol'`` (inverse-vol) or ``'equal'``.

    Known at t: on a rebalance date t the weights use rows (t-window, t]; in
    between, the last target is held. Before the first full window the target
    is zero. Feed the result straight into ``backtest.run`` (which shifts).

    Limits: see ``hrp_weights``. Fixed calendar rebalancing ignores drift in
    holdings between rebalances (the backtester holds constant weights, i.e.
    implicitly daily rebalances to the target, which is optimistic on costs).
    """
    fn = {"hrp": lambda x: hrp_weights(x, split=split), "ivol": inverse_vol_weights, "equal": equal_weights}[method]

    def _w(hist: pd.DataFrame, t) -> pd.Series:
        return fn(hist.iloc[-window:])

    return rolling_weights(asset_returns, _w, first=window - 1, every=rebalance_every)
