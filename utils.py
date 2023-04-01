"""Utilities for bivariate copulas."""
import numpy as np


def pseudo_inv_ecdf(F, x):
    Fx, Fy = F.x[1:], F.y[1:]
    if x < 0 or x > 1:
        return np.nan
    else:
        idx_m = next(idx for idx, y_m in enumerate(Fy) if x <= y_m)
        return Fx[idx_m]