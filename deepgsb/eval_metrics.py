import gc
import logging
import math
import time
from typing import Dict, List, Tuple, Union, TYPE_CHECKING

import torch
from geomloss import SamplesLoss

from mfg import MFG
from options import Options

from . import util

if TYPE_CHECKING:
    from deepgsb.deepgsb import DeepGSB

log = logging.getLogger(__file__)


def get_bound(mfg: MFG, xs: torch.Tensor, train_direction: str):
    return {
        "forward": [xs[:, 0], mfg.p0, "0"],  # train forward, xs are sampled from backward
        "backward": [xs[:, -1], mfg.pT, "T"],  # train backward, xs are sampled from forward
    }.get(train_direction)


@torch.no_grad()
def compute_conv_l1_metrics(mfg: MFG, xs: torch.Tensor, train_direction: str) -> Dict[str, torch.Tensor]:
    b, T, nx = xs.shape

    metrics = {}

    # (samp_bs, ...)
    x_bound, p_bound, time_str = get_bound(mfg, xs, train_direction)

    # Max singular value / min singular value.
    S = torch.linalg.svdvals(x_bound)
    # Largest to smallest.
    eigvals = (S ** 2) / (b - 1)
    cond_number = eigvals[0] / eigvals[-1]
    del S, eigvals

    # Eigvalsh returns smallest to largest.
    true_cov = p_bound.distribution.covariance_matrix.cpu()
    true_eigvals = torch.linalg.eigvalsh(true_cov)
    true_cond_number = true_eigvals[-1] / true_eigvals[0]

    metrics[f"cond_num diff {time_str}"] = torch.abs(cond_number - true_cond_number)

    # Pred stds (project on each axis).
    pred_stds = torch.std(x_bound, dim=0)
    # True stds (project on each axis).
    true_stds = p_bound.distribution.stddev.cpu()

    std_l1 = torch.abs(true_stds - pred_stds).sum()
    metrics[f"std l1 {time_str}"] = std_l1

    # Also just compare the covariance matrix.
    pred_cov = torch.cov(x_bound.T)
    assert pred_cov.shape == true_cov.shape
    cov_l1 = torch.abs(true_cov - pred_cov).sum()
    metrics[f"cov l1 {time_str}"] = cov_l1

    return metrics


@torch.no_grad()
def compute_sinkhorn_metrics(opt: Options, mfg: MFG, xs: torch.Tensor, train_direction: str) -> Dict[str, torch.Tensor]:
    b, T, nx = xs.shape

    metrics = {}

    # Higher scaling results in more accurate estimate.
    if opt.x_dim > 2:
        # L1 norm is better in high dimensional setting.
        # If it's high dimension, empiricla Wasserstein is probably not that good of a metric, so we don't need
        # to spend as much time on computing this.
        sinkhorn = SamplesLoss("sinkhorn", p=1, blur=5e-2, scaling=0.9)
    else:
        sinkhorn = SamplesLoss("sinkhorn", p=2, blur=5e-3, scaling=0.9)

    x_bound, p_bound, time_str = get_bound(mfg, xs, train_direction)
    W2_name = f"W2_{time_str}"

    p_samples = p_bound.sample(batch=b)

    log.info("Computing sinkhorn....")
    t1 = time.time()
    x_bound = x_bound.to(opt.device)
    metrics[W2_name] = sinkhorn(x_bound, p_samples)
    log.info("Done! sinkhorn took {:.1f} s".format(time.time() - t1))

    return metrics


@torch.no_grad()
def compute_control_cost(opt: Options, zs: torch.Tensor, dt: float) -> torch.Tensor:
    b, T, nx = zs.shape

    zs = zs.to(opt.device)
    control_cost = torch.square(zs).sum((1, 2)) * dt
    return control_cost.detach().cpu()


@torch.no_grad()
def compute_state_cost(
    opt: Options, mfg: MFG, xs: torch.Tensor, ts: torch.Tensor, xs_all: torch.Tensor, logp: torch.Tensor, dt: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, T, nx = xs.shape
    dim01 = (b, T)

    flat_xs = util.flatten_dim01(xs).to(opt.device)
    s_cost, mf_cost = mfg.state_cost_fn(flat_xs, ts, logp, xs_all)
    del flat_xs

    # (samp_bs, T)
    s_cost, mf_cost = util.unflatten_dim01(s_cost, dim01), util.unflatten_dim01(mf_cost, dim01)

    s_cost = (s_cost.sum(1) * dt).detach().cpu()
    mf_cost = (mf_cost.sum(1) * dt).detach().cpu()

    return s_cost, mf_cost


@torch.jit.script
def serial_logkde(train_xs: torch.Tensor, xs: torch.Tensor, bw: float, max_batch: int = 32) -> torch.Tensor:
    # xs: (b1, T, *)
    # train_xs: (b2, T, *)
    # out: (b1, T)

    b1, T, nx = xs.shape
    b2, T, nx = train_xs.shape
    assert xs.shape == (b1, T, nx) and train_xs.shape == (b2, T, nx)

    coeff = b2 * math.sqrt(2 * math.pi * bw)
    log_coeff = math.log(coeff)

    xs_chunks = torch.split(xs, max_batch, dim=0)
    out_chunks = []
    for xs in xs_chunks:
        # 1: Compute diffs. (b1, b2, T, *)
        diffs = train_xs.unsqueeze(0) - xs.unsqueeze(1)
        # (b1, b2, T)
        norm_sq = torch.sum(torch.square(diffs), dim=-1)

        # (b1, b2, T) -> (b1, T)
        logsumexp = torch.logsumexp(-norm_sq / (2 * bw), dim=1)

        out = logsumexp - log_coeff
        out_chunks.append(out)

    out = torch.cat(out_chunks, dim=0)
    assert out.shape == (b1, T)

    return out


@torch.no_grad()
def compute_est_mf_cost(
    opt: Options,
    runner: "DeepGSB",
    xs: torch.Tensor,
    ts: torch.Tensor,
    dt: float,
    xs_all: torch.Tensor,
    return_logp: bool = False,
) -> Union[List[torch.Tensor], torch.Tensor]:
    if not runner.mfg.uses_logp():
        return [torch.zeros(1), None] if return_logp else torch.zeros(1)

    b, T, nx = xs.shape
    dim01 = (b, T)

    bw = 0.2 ** 2

    xs = xs.to(opt.device)
    if b > 500:
        rand_idxs = torch.randperm(b)[:500]
        reduced_xs = xs[rand_idxs]
        del rand_idxs
    else:
        reduced_xs = xs

    max_batch = 32 if opt.x_dim == 2 else 2
    logp = serial_logkde(reduced_xs, xs, bw=bw, max_batch=max_batch)
    del reduced_xs

    assert logp.shape == (b, T)
    logp = util.flatten_dim01(logp).detach().cpu()

    gc.collect()

    # Evaluate logp in chunks to prevent cuda OOM.
    ts = ts.repeat(b)
    flat_xs = util.flatten_dim01(xs)

    max_chunk_size = 512
    flat_xs_chunks = torch.split(flat_xs, max_chunk_size, dim=0)
    ts_chunks = torch.split(ts, max_chunk_size, dim=0)

    est_logp = []
    for flat_xs_chunk, ts_chunk in zip(flat_xs_chunks, ts_chunks):
        flat_xs_chunk = flat_xs_chunk.to(opt.device)
        ts_chunk = ts_chunk.to(opt.device)
        value_t_ema = runner.z_f.compute_value(flat_xs_chunk, ts_chunk, use_ema=True)
        value_impt_t_ema = runner.z_b.compute_value(flat_xs_chunk, ts_chunk, use_ema=True)
        est_logp.append((value_t_ema + value_impt_t_ema).detach().cpu())
        del flat_xs_chunk, ts_chunk, value_t_ema, value_impt_t_ema
    est_logp = torch.cat(est_logp, dim=0)

    # flat_xs, ts = flat_xs.to(opt.device), ts.to(opt.device)
    # value_t_ema = self.z_f.compute_value(flat_xs, ts, use_ema=True)
    # value_impt_t_ema = self.z_b.compute_value(flat_xs, ts, use_ema=True)
    # est_logp = value_t_ema + value_impt_t_ema

    assert logp.shape == est_logp.shape == (b * T,)

    _, est_mf_cost = runner.mfg.state_cost_fn(flat_xs, ts, est_logp, xs_all)
    del est_logp

    est_mf_cost = util.unflatten_dim01(est_mf_cost, dim01)
    est_mf_cost = torch.mean(est_mf_cost.sum(1) * dt).detach().cpu()

    return [est_mf_cost, logp] if return_logp else est_mf_cost
