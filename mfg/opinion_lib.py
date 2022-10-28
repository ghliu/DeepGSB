import logging
import math
from typing import Optional

import numpy as np
import torch

from options import Options

log = logging.getLogger(__file__)


@torch.no_grad()
@torch.jit.script
def est_directional_similarity(xs: torch.Tensor, n_est: int = 1000) -> torch.Tensor:
    """xs: (batch, nx). Returns (n_est, ) between 0 and 1."""
    # xs: (batch, nx)
    batch, nx = xs.shape

    # Center first.
    xs = xs - torch.mean(xs, dim=0, keepdim=True)

    rand_idxs1 = torch.randint(batch, [n_est], dtype=torch.long)
    rand_idxs2 = torch.randint(batch, [n_est], dtype=torch.long)

    # (n_est, nx)
    xs1 = xs[rand_idxs1]
    # (n_est, nx)
    xs2 = xs[rand_idxs2]

    # Normalize to unit vector.
    xs1 /= torch.linalg.norm(xs1, dim=1, keepdim=True)
    xs2 /= torch.linalg.norm(xs2, dim=1, keepdim=True)

    # (n_est, )
    cos_angle = torch.sum(xs1 * xs2, dim=1).clip(-1.0, 1.0)
    assert cos_angle.shape == (n_est,)

    # Should be in [0, pi).
    angle = torch.acos(cos_angle)
    assert (0 <= angle).all()
    assert (angle <= torch.pi).all()

    D_ij = 1.0 - angle / torch.pi
    assert D_ij.shape == (n_est,)

    return D_ij


def opinion_thresh(inner: torch.Tensor) -> torch.Tensor:
    return 2.0 * (inner > 0) - 1.0


@torch.jit.script
def compute_mean_drift_term(mf_x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Decompose the polarize dynamic Eq (18) in paper into 2 parts for faster computation:
          f_polarize(x,p,ξ)
        = E_{y~p}[a(x,y,ξ) * bar_y],                  where a(x,y,ξ) = sign(<x,ξ>)*sign(<y,ξ>)
                                                      and      bar_y = y / |y|^{0.5}
        = sign(<x,ξ>) * E_{y~p}[sign(<y,ξ>) * bar_y], since sign(<x,ξ>) is independent of y
        = A(x,ξ)      * B(p,ξ)
    Hence, bar_f_polarize = bar_A(x,ξ) * bar_B(p,ξ)
    This function computes only bar_B(p,ξ).
    """
    # mf_x: (b, nx), xi: (nx,)
    # output: (nx,)

    b, nx = mf_x.shape
    assert xi.shape == (nx,)

    mf_x_norm = torch.linalg.norm(mf_x, dim=-1, keepdim=True)
    assert torch.all(mf_x_norm > 0.0)

    normalized_mf_x = mf_x / torch.sqrt(mf_x_norm)
    assert normalized_mf_x.shape == (b, nx)

    # Compute the mean drift term:   1/J sum_j a(y_j) y_j / sqrt(| y_j |).
    mf_agree_j = opinion_thresh(torch.sum(mf_x * xi, dim=-1, keepdim=True))
    assert mf_agree_j.shape == (b, 1)

    mean_drift_term = torch.mean(mf_agree_j * normalized_mf_x, dim=0)
    assert mean_drift_term.shape == (nx,)

    mean_drift_term_norm = torch.linalg.norm(mean_drift_term, dim=-1, keepdim=True)
    mean_drift_term = mean_drift_term / torch.sqrt(mean_drift_term_norm)
    assert mean_drift_term.shape == (nx,)

    return mean_drift_term


@torch.jit.script
def opinion_f(x: torch.Tensor, mf_drift: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """This function computes the polarize dynamic in Eq (18) by
        bar_f_polarize(x,p,ξ) = bar_A(x,ξ) * bar_B(p,ξ)
    where bar_B(p,ξ) is pre-computed in func compute_mean_drift_term and passed in as mf_drift.
    """
    # x: (b, T, nx), mf_drift: (T, nx), xi: (T, nx)
    # out: (b, T, nx)

    b, T, nx = x.shape
    assert xi.shape == mf_drift.shape == (T, nx)

    agree_i = opinion_thresh(torch.sum(x * xi, dim=-1, keepdim=True))
    # Make sure we are not dividing by 0.
    agree_i[agree_i == 0] = 1.0

    abs_sqrt_agree_i = torch.sqrt(torch.abs(agree_i))
    assert torch.all(abs_sqrt_agree_i > 0.0)

    norm_agree_i = agree_i / abs_sqrt_agree_i
    assert norm_agree_i.shape == (b, T, 1)

    f = norm_agree_i * mf_drift
    assert f.shape == (b, T, nx)

    return f


def build_f_mul(opt: Options) -> torch.Tensor:
    # set f_mul with some heuristic so that it doesn't diverge exponentially fast
    # and yield bad normalization, since the more polarized the opinion is the faster it will grow
    ts = torch.linspace(opt.t0, opt.T, opt.interval)
    coeff = 8.0
    f_mul = torch.clip(1.0 - torch.exp(coeff * (ts - opt.T)) + 1e-5, min=1e-4, max=1.0)
    f_mul = f_mul ** 5.0
    return f_mul


def build_xis(opt: Options) -> torch.Tensor:
    # Generate random unit vectors.
    rng = np.random.default_rng(seed=4078213)
    xis = rng.standard_normal([opt.interval, opt.x_dim])

    # Construct a xis that has some degree of "continuous" over time, as a brownian motion.
    xi = xis[0]
    bm_xis = [xi]
    std = 0.4
    for t in range(1, opt.interval):
        xi = xi - (2.0 * xi) * 0.01 + std * math.sqrt(0.01) * xis[t]
        bm_xis.append(xi)
    assert len(bm_xis) == xis.shape[0]

    xis = torch.Tensor(np.stack(bm_xis))
    xis /= torch.linalg.norm(xis, dim=-1, keepdim=True)

    # Just safeguard if the self.xis becomes different.
    log.info("USING BM XI! xis.sum(): {}".format(torch.sum(xis)))
    return xis
