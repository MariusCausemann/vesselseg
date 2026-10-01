"""Measurement model of the extracted radius, shared by extract.py and fit_radius.py.

A radius sample r (mm) at evidence P is read as

    log r = log r_true + e,   e ~ scale sigma(r),   with probability rho(P)
    anything,                                        with probability 1 - rho(P)

    sigma(r)^2 = sigma0^2 (1 + (h / r)^2)

h the voxel size. The relative error sigma0 is what remains for a well-resolved
vessel; the second part is an absolute boundary error of ~sigma0 h mm, which
dominates for a vessel a few voxels wide (the extraction clips the radius at
h/2, so a floor reading is the censored end of this: an error of sqrt(5) sigma0
there). rho = P is the probability that the sample measures a lumen at all: a
faint vessel's strict mask is defined by a threshold the evidence barely
crosses, and its radius says little. The model replaces the fixed rules for
"unmeasured" samples (radius below r_floor_tol x the floor, P below p_meas).

extract.py, radius growth from parent to child (the excess of the log
ratio over log r_tol, v >= 0, Laplace errors): the negative log-likelihood of
the mixture,

    cost = -log(rho exp(-w v / s) + 1 - rho),   s^2 = mean of (sigma / sigma0)^2,

rho = rho_max P_parent P_child. With rho_max = 1 - exp(-w), a full-evidence pair
reproduces the former min(w v, w) for a well-resolved pair; the cost is
bounded by -log(1 - rho), so a faint pair costs little whatever its ratio.
Capacity band: the log half-width of log(c0 r^3) scales with sigma(r) / sigma0,
and the penalty per mm off the band with rho = the segment's median P.

fit_radius.py: data weight length x rho / (sigma / sigma0)^2 = l P r^2 / (r^2 + h^2).
"""

import numpy as np


def rel_var(r, h):
    """(sigma(r) / sigma0)^2: the error variance of log r relative to a resolved vessel."""
    r = np.maximum(np.asarray(r, float), 1e-6)
    return 1.0 + (h / r) ** 2


def validity(p):
    """rho: probability that a sample at evidence P measures a lumen."""
    return np.clip(np.asarray(p, float), 0.0, 1.0)


def growth_cost(r_parent, r_child, p_parent, p_child, h, w, r_tol):
    """Cost of a child thicker than r_tol x its parent (see the module docstring)."""
    v = max(0.0, float(np.log(r_child / max(r_parent, 1e-3)) - np.log(r_tol)))
    if v == 0.0:
        return 0.0
    s = float(np.sqrt((rel_var(r_parent, h) + rel_var(r_child, h)) / 2))
    rho = (1 - np.exp(-w)) * float(validity(p_parent) * validity(p_child))
    return float(-np.log(rho * np.exp(-w * v / s) + 1 - rho))
