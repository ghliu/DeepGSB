from typing import Tuple, Union, Iterable

import torch
from torch.nn.functional import huber_loss

from mfg import MFG, MFGPolicy
from mfg.sde import SimpleSDE
from options import Options

from . import util
from .replay_buffer import Buffer

DivGZRetType = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
KLLossRetType = Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@torch.jit.script
def rev_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.flip(x, dims=[dim])
    x = torch.cumsum(x, dim=dim)
    return torch.flip(x, dims=[dim])


def sample_e(opt: Options, x: torch.Tensor) -> torch.Tensor:
    if opt.noise_type == "gaussian":
        return torch.randn_like(x)
    elif opt.noise_type == "rademacher":
        return torch.randint(low=0, high=2, size=x.shape).to(x.device) * 2 - 1
    else:
        raise ValueError(f"Unsupport noise type {opt.noise_type}!")


def compute_div_gz(
    opt: Options, mfg: MFG, ts: torch.Tensor, xs: torch.Tensor, policy: MFGPolicy, return_zs: bool = False
) -> DivGZRetType:
    b, T, nx = xs.shape
    assert ts.shape == (T,)

    if mfg.uses_state_dependent_drift():
        f = mfg.f(xs, ts, policy.direction)
        assert f.shape == (b, T, nx)
        f = util.flatten_dim01(f)
    else:
        f = 0

    ts = ts.repeat(xs.shape[0])
    xs = util.flatten_dim01(xs)
    zs = policy(xs, ts)

    g_ts = mfg.g(ts)
    g_ts = g_ts[:, None]
    gzs = g_ts * zs - f

    e = sample_e(opt, xs)
    e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    div_gz = e_dzdx * e

    return [div_gz, zs] if return_zs else div_gz


def compute_kl(
    opt: Options,
    mfg: MFG,
    ts: torch.Tensor,
    xs: torch.Tensor,
    zs_impt: torch.Tensor,
    policy: MFGPolicy,
    return_zs: bool = False,
) -> DivGZRetType:
    b, T, nx = xs.shape
    assert ts.shape == (T,)

    zs_impt = util.flatten_dim01(zs_impt)
    assert zs_impt.shape == (b * T, nx)

    with torch.enable_grad():
        _xs = xs.detach()
        xs = _xs.requires_grad_(True)

        div_gz, zs = compute_div_gz(opt, mfg, ts, xs, policy, return_zs=True)
        assert div_gz.shape == zs.shape == (b * T, nx)

        # (b * T, xdim)
        kl = zs * (0.5 * zs + zs_impt) + div_gz
    assert kl.shape == (b * T, nx)

    return [kl, zs] if return_zs else kl


def compute_norm_loss(
    norm: str, predict: torch.Tensor, label: torch.Tensor, batch_x: int, dt: float, delta: float = 1.0
) -> torch.Tensor:
    assert norm in ["l1", "l2", "huber"]
    assert predict.shape == label.shape

    if norm == "l1":
        return 0.5 * ((predict - label).abs() * dt).sum() / batch_x
    elif norm == "l2":
        return 0.5 * ((predict - label) ** 2 * dt).sum() / batch_x
    elif norm == "huber":
        return huber_loss(predict, label, reduction="sum", delta=delta) * dt / batch_x


def compute_kl_loss(
    opt: Options,
    dim01: Iterable[int],
    mfg: MFG,
    samp_direction: str,
    ts: torch.Tensor,
    xs: torch.Tensor,
    zs_impt: torch.Tensor,
    policy: MFGPolicy,
    return_all: bool = False,
) -> KLLossRetType:

    kl, zs = compute_kl(opt, mfg, ts, xs, zs_impt, policy, return_zs=True)
    zs_impt = util.flatten_dim01(zs_impt)
    assert kl.shape == zs.shape == zs_impt.shape

    _, x_init, p_init, _, x_term, p_term = mfg.get_init_term_bound(xs, samp_direction)

    # computationally same as kl yet better interpretation
    bsde_y_yhat = util.unflatten_dim01((kl + 0.5 * zs_impt ** 2).sum(dim=-1) * mfg.dt, dim01).sum(
        dim=1
    )  # (batch_x, len_t) --> (batch_x)
    loss_kl = (p_init.log_prob(x_init) + bsde_y_yhat - p_term.log_prob(x_term)).mean()

    return [loss_kl, zs, kl, bsde_y_yhat] if return_all else loss_kl


def compute_bsde_td_loss_multistep(
    mfg: MFG,
    samp_direction: str,
    ts: torch.Tensor,
    xs: torch.Tensor,
    zs: torch.Tensor,
    dw: torch.Tensor,
    kl: torch.Tensor,
    policy: MFGPolicy,
    policy_impt: MFGPolicy,
    xs_all: torch.Tensor,
    norm: str = "huber",
) -> torch.Tensor:
    # xs, xs_all: (b, T, nx)
    # zs: (b * T, nx)
    # dw: (b, T, nx)
    # kl: (b * T, nx)
    b, T, nx = xs.shape
    ts = ts.repeat(xs.shape[0])
    xs = util.flatten_dim01(xs)
    dw = util.flatten_dim01(dw)
    zs = zs.reshape(*dw.shape)

    dim01 = [b, T]

    # (b * T, )
    value_ema = policy.compute_value(xs, ts, use_ema=True)
    # ( b * T, )
    value_impt_ema = policy_impt.compute_value(xs, ts, use_ema=True)

    log_p = value_ema + value_impt_ema
    state_cost = mfg.compute_state_cost(xs, ts, xs_all, log_p)

    # (b * T, )
    bsde2_1step = (kl.sum(dim=-1) - state_cost) * mfg.dt + (zs * dw).sum(dim=-1)

    # (b, T)
    bsde2_1step = util.unflatten_dim01(bsde2_1step, dim01)

    # (b * T, )
    value = policy.compute_value(xs, ts)

    # (b, T, nx)
    xs = util.unflatten_dim01(xs, dim01)

    t_bnd, x_bnd, p_bnd = mfg.get_bound(xs, samp_direction, "init")
    if samp_direction == "forward":
        with torch.no_grad():
            # (b, )
            value_impt_bnd = util.unflatten_dim01(value_impt_ema, dim01)[:, 0]
            # (b, )
            target_bnd = p_bnd.log_prob(x_bnd) - value_impt_bnd

        # (b, )
        bsde2_cumstep = torch.cumsum(bsde2_1step, dim=1)
        # (b, T) -> (b, T - 1)
        target = (target_bnd[:, None] + bsde2_cumstep)[:, :-1]
        # (b, T) -> (b, T - 1)
        predict = util.unflatten_dim01(value, dim01)[:, 1:]
    else:
        with torch.no_grad():
            # (b, )
            value_impt_bnd = util.unflatten_dim01(value_impt_ema, dim01)[:, -1]
            # (b, )
            target_bnd = p_bnd.log_prob(x_bnd) - value_impt_bnd

        # Cumulative sum from T to 0.
        bsde2_cumstep = rev_cumsum(bsde2_1step, dim=1)
        target = (target_bnd[:, None] + bsde2_cumstep)[:, 1:]
        predict = util.unflatten_dim01(value, dim01)[:, :-1]

    label = target.detach()
    return compute_norm_loss(norm, predict, label, b, mfg.dt, delta=2.0)


def compute_bsde_td_loss_singlestep(
    mfg: MFG,
    samp_direction: str,
    ts: torch.Tensor,
    xs: torch.Tensor,
    zs: torch.Tensor,
    dw: torch.Tensor,
    kl: torch.Tensor,
    policy: MFGPolicy,
    policy_impt: MFGPolicy,
    xs_all: torch.Tensor,
    norm: str = "l2",
) -> torch.Tensor:
    b, T, nx = xs.shape
    assert samp_direction in ["forward", "backward"]

    # (1) flattent all input from (b,T, ...) to (b*T, ...)
    ts = ts.repeat(xs.shape[0])
    xs = util.flatten_dim01(xs)
    dw = util.flatten_dim01(dw)
    zs = zs.reshape(*dw.shape)

    value = policy.compute_value(xs, ts)
    value_ema = policy.compute_value(xs, ts, use_ema=True)
    value_impt_ema = policy_impt.compute_value(xs, ts, use_ema=True)

    # (2) compute state cost (i.e., F in Eq 14)
    log_p = value_ema + value_impt_ema
    state_cost = mfg.compute_state_cost(xs, ts, xs_all, log_p)

    # (3) construct δY, predicted Y, target Y (from ema)
    bsde_1step = (kl.sum(dim=-1) - state_cost) * mfg.dt + (zs * dw).sum(dim=-1)
    bsde_predict = value
    bsde_target = value_ema

    assert bsde_predict.shape == bsde_1step.shape == bsde_target.shape

    bsde_predict = util.unflatten_dim01(bsde_predict, [b, T])
    bsde_1step = util.unflatten_dim01(bsde_1step, [b, T])
    bsde_target = util.unflatten_dim01(bsde_target, [b, T])

    # (4) compute TD prediction and target in Eq 14
    #    [forward] dYhat_t: predict=Yhat_{t+1}, label=Yhat_t + [...]
    #    [forward] dY_s:    predict Y_{s+1}, label=Y_s + [...]
    if samp_direction == "forward":
        label = (bsde_target + bsde_1step)[:, :-1].detach()
        predict = bsde_predict[:, 1:]
    elif samp_direction == "backward":
        label = (bsde_target + bsde_1step)[:, 1:].detach()
        predict = bsde_predict[:, :-1]
    else:
        raise ValueError(f"samp_direction should be either forward or backward, got {samp_direction}")

    return compute_norm_loss(norm, predict, label, b, mfg.dt)


def compute_bsde_td_loss(
    opt: Options,
    mfg: MFG,
    samp_direction: str,
    ts: torch.Tensor,
    xs: torch.Tensor,
    zs: torch.Tensor,
    dw: torch.Tensor,
    kl: torch.Tensor,
    policy: MFGPolicy,
    policy_impt: MFGPolicy,
    xs_all: torch.Tensor,
) -> torch.Tensor:
    if opt.multistep_td:
        bsde_td_loss = compute_bsde_td_loss_multistep
    else:
        bsde_td_loss = compute_bsde_td_loss_singlestep
    return bsde_td_loss(mfg, samp_direction, ts, xs, zs, dw, kl, policy, policy_impt, xs_all)


def compute_bsde_td_loss_from_buffer(
    opt: Options, mfg, buffer: Buffer, ts: torch.Tensor, policy: MFGPolicy, policy_impt: MFGPolicy, xs_all: torch.Tensor
) -> torch.Tensor:
    samp_direction = policy_impt.direction
    assert policy.direction != buffer.direction
    assert policy_impt.direction == buffer.direction

    batch_x = opt.rb_bs_x
    xs, zs_impt, dw = buffer.sample_traj(batch_x)

    assert xs.shape == zs_impt.shape == (batch_x, opt.interval, opt.x_dim)

    kl, zs = compute_kl(opt, mfg, ts, xs, zs_impt, policy, return_zs=True)
    assert kl.shape == zs.shape and kl.shape[0] == (batch_x * opt.interval)

    return compute_bsde_td_loss(opt, mfg, samp_direction, ts, xs, zs, dw, kl, policy, policy_impt, xs_all)


def compute_boundary_loss(
    opt: Options, mfg: MFG, ts: torch.Tensor, xs: torch.Tensor, policy2: MFGPolicy, policy: MFGPolicy, norm: str = "l2"
) -> torch.Tensor:

    t_bound, x_bound, p_bound = mfg.get_bound(xs.detach(), policy.direction, "term")

    batch_x, nx = x_bound.shape

    t_bound = t_bound.repeat(batch_x)

    value2_ema = policy2.compute_value(x_bound, t_bound, use_ema=True)
    label = (p_bound.log_prob(x_bound) - value2_ema).detach()
    predict = policy.compute_value(x_bound, t_bound)

    return compute_norm_loss(norm, predict, label, batch_x, mfg.dt)


def compute_grad_loss(
    opt: Options, ts: torch.Tensor, xs: torch.Tensor, dyn: SimpleSDE, policy: MFGPolicy, norm: str
) -> torch.Tensor:
    batch_x = xs.shape[0]
    ts = ts.repeat(batch_x)
    xs = util.flatten_dim01(xs)

    predict = policy(xs, ts)
    label = dyn.std * policy.compute_value_grad(xs, ts).detach()

    return compute_norm_loss(norm, predict, label, batch_x, dyn.dt, delta=1.0)
