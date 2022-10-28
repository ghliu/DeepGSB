import abc
import logging
import math
from typing import Callable, Optional, NamedTuple

import numpy as np
import torch

from options import Options

from . import opinion_lib, util
from .constraint import Sampler

log = logging.getLogger(__file__)

DriftFn = Callable[[torch.Tensor], torch.Tensor]

class SampledTraj(NamedTuple):
    xs: torch.Tensor
    zs: torch.Tensor
    ws: torch.Tensor
    x_term: torch.Tensor

def _assert_increasing(name: str, ts: torch.Tensor) -> None:
    assert (ts[1:] > ts[:-1]).all(), "{} must be strictly increasing".format(name)


def base_drift_builder(opt: Options) -> Optional[DriftFn]:
    if opt.problem_name in ["Vneck", "Stunnel"]:

        def base_drift(x):
            b, T, nx = x.shape
            const = torch.Tensor([6.0, 0.0], device=x.device)
            assert const.shape == (nx,)
            return const.repeat(b, T, 1)

    else:
        base_drift = None
    return base_drift


def t_to_idx(t: torch.Tensor, interval: int, T: float) -> torch.Tensor:
    return (t / T * (interval - 1)).round().long()


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt: Options, p0: Sampler, pT: Sampler):
        self.opt = opt
        self.dt = opt.T / opt.interval
        self.p0 = p0
        self.pT = pT
        self.mf_drifts: Optional[torch.Tensor] = None

    @abc.abstractmethod
    def _f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def f(self, x: torch.Tensor, t: torch.Tensor, direction: str) -> torch.Tensor:
        # x: (b, T, nx), t: (T,)
        # out: (b, T, nx)
        b, T, nx = x.shape
        assert t.shape == (T,)

        sign = 1.0 if direction == "forward" else -1.0
        _f = self._f(x, t)
        assert _f.shape == (b, T, nx)
        return sign * _f

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self._g(t)

    def dw(self, x: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        dt = self.dt if dt is None else dt
        return torch.randn_like(x) * np.sqrt(dt)

    def propagate(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        direction: str,
        f: Optional[DriftFn] = None,
        dw: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        g = self.g(t)
        f = self.f(x, t, direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x, dt) if dw is None else dw

        return x + (f + g * z) * dt + g * dw

    @torch.no_grad()
    def initialize_mf_drift(self, ts: torch.Tensor, policy) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def update_mf_drift(self, x: torch.Tensor, t_idx: int) -> None:
        raise NotImplementedError

    def sample_traj(self, ts: torch.Tensor, policy, update_mf_drift: bool = False) -> SampledTraj:

        # first we need to know whether we're doing forward or backward sampling
        direction = policy.direction
        assert direction in ["forward", "backward"]

        # set up ts and init_distribution
        _assert_increasing("ts", ts)
        init_dist = self.p0 if direction == "forward" else self.pT
        ts = ts if direction == "forward" else torch.flip(ts, dims=[0])

        x = init_dist.sample(batch=self.opt.samp_bs)
        (b, nx), T = x.shape, len(ts)
        assert nx == self.opt.x_dim

        xs = torch.empty((b, T, nx))
        zs = torch.empty_like(xs)
        ws = torch.empty_like(xs)
        if update_mf_drift:
            self.mf_drifts = torch.empty(T, nx)

        # don't use tqdm for fbsde since it'll resample every itr
        for idx, t in enumerate(ts):
            t_idx = idx if direction == "forward" else T - idx - 1
            assert t_idx == t_to_idx(t, self.opt.interval, self.opt.T), (t_idx, t)

            if update_mf_drift:
                self.update_mf_drift(x, t_idx)

            # f = self.f(x,t,direction)
            # handle propagation of single time step
            f = self.f(x.unsqueeze(1), t.unsqueeze(0), direction).squeeze(1)
            z = policy(x, t)
            dw = self.dw(x)
            util.assert_zero_grads(policy)

            xs[:, t_idx, ...] = x
            zs[:, t_idx, ...] = z
            ws[:, t_idx, ...] = dw

            x = self.propagate(t, x, z, direction, f=f, dw=dw)

        x_term = x

        return SampledTraj(xs, zs, ws, x_term)


class SimpleSDE(BaseSDE):
    def __init__(self, opt: Options, p: Sampler, q: Sampler, base_drift: Optional[DriftFn] = None):
        super(SimpleSDE, self).__init__(opt, p, q)
        self.std = opt.diffusion_std
        self.base_drift = base_drift

    def _f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x) if self.base_drift is None else self.base_drift(x)

    def _g(self, t: torch.Tensor) -> torch.Tensor:
        return torch.Tensor([self.std])


class OpinionSDE(SimpleSDE):
    """modified from the party model:
    See Eq (4) in https://www.cs.cornell.edu/home/kleinber/ec21-polarization.pdf
    """

    def __init__(self, opt: Options, p: Sampler, q: Sampler):
        super(OpinionSDE, self).__init__(opt, p, q)
        assert "opinion" in opt.problem_name

        self.f_mul = opinion_lib.build_f_mul(opt)
        self.xis = opinion_lib.build_xis(opt)
        self.polarize_strength = 1.0 if opt.x_dim == 2 else 6.0
        self.mf_drifts = None

    def _f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        b, T, nx = x.shape
        assert nx == self.opt.x_dim

        idx = t_to_idx(t, self.opt.interval, self.opt.T)

        fmul = self.f_mul[idx].to(x.device).unsqueeze(-1)
        xi = self.xis[idx].to(x.device)
        mf_drift = self.mf_drifts[idx].to(x.device)
        assert fmul.shape == (T, 1)
        assert xi.shape == mf_drift.shape == (T, nx)

        f = self.polarize_strength * opinion_lib.opinion_f(x, mf_drift, xi)
        assert f.shape == x.shape

        f = fmul * f
        assert f.shape == x.shape

        return f

    @torch.no_grad()
    def initialize_mf_drift(self, ts: torch.Tensor, policy) -> None:
        self.sample_traj(ts, policy, update_mf_drift=True)

    @torch.no_grad()
    def update_mf_drift(self, x: torch.Tensor, t_idx: int) -> None:
        xi = self.xis[t_idx].to(x.device)
        mf_drift = opinion_lib.compute_mean_drift_term(x, xi)
        self.mf_drifts[t_idx] = mf_drift.detach().cpu()


def build_sde(opt: Options, p: Sampler, q: Sampler) -> SimpleSDE:
    log.info("build base sde...")

    if "opinion" in opt.problem_name:
        return OpinionSDE(opt, p, q)
    else:
        base_drift = base_drift_builder(opt)
        return SimpleSDE(opt, p, q, base_drift=base_drift)
