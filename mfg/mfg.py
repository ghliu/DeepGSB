import logging
from typing import Callable, Optional, Tuple

import torch

from options import Options

from .constraint import Sampler, build_constraint
from .plotting import snapshot
from .sde import SampledTraj, build_sde
from .state_cost import build_state_cost_fn

log = logging.getLogger(__file__)

Bound = Tuple[torch.Tensor, torch.Tensor, Sampler]
InitTermBound = Tuple[torch.Tensor, torch.Tensor, Sampler, torch.Tensor, torch.Tensor, Sampler]

# if the state cost for this problem requires logp.
logp_list = [
    "Vneck",
    "opinion",
    "opinion_1k",
]

# if the state cost for this problem requires xs_all.
xs_all_list = ["GMM", "Stunnel", "Vneck"]

# if the uncontrolled drift use MF
mf_drift_list = ["opinion", "opinion_1k"]

# if the uncontrolled drift depends on state
state_dependent_drift_list = ["opinion", "opinion_1k"]


def get_bound_index(direction: str, bound: str) -> int:
    assert direction in ["forward", "backward"]
    assert bound in ["init", "term"]

    if direction == "forward" and bound == "init":
        return 0
    elif direction == "forward" and bound == "term":
        return -1
    elif direction == "backward" and bound == "init":
        return -1
    elif direction == "backward" and bound == "term":
        return 0


class MFGPolicy(torch.nn.Module):
    def __init__(self, direction, dyn):
        super(MFGPolicy, self).__init__()
        self.direction = direction
        self.dyn = dyn


class MFG:
    def __init__(self, opt: Options):

        self.opt = opt
        self.problem_name = opt.problem_name

        self.ts = torch.linspace(opt.t0, opt.T, opt.interval)
        self.mf_coeff = opt.MF_cost

        self.p0, self.pT = build_constraint(opt.problem_name, opt.samp_bs, opt.device)
        self.state_cost_fn, self.obstacle_cost_fn = build_state_cost_fn(opt.problem_name)

        self.pbound = [self.p0, self.pT]

        self.sde = build_sde(opt, self.p0, self.pT)

    @property
    def dt(self) -> float:
        return self.sde.dt

    def f(self, x: torch.Tensor, t: torch.Tensor, direction: str) -> torch.Tensor:
        return self.sde.f(x, t, direction)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self.sde.g(t)

    def uses_logp(self) -> bool:
        """Returns true if the state cost for this problem requires logp."""
        return self.problem_name in logp_list

    def uses_xs_all(self) -> bool:
        """Returns true if the state cost for this problem requires xs_all."""
        return self.problem_name in xs_all_list

    def uses_mf_drift(self) -> bool:
        """Returns true if the uncontrolled drift for this problem involves mean field."""
        return self.problem_name in mf_drift_list

    def uses_state_dependent_drift(self) -> bool:
        """Returns true if the uncontrolled drift for this problem depends on state x."""
        return self.problem_name in state_dependent_drift_list

    def initialize_mf_drift(self, policy: MFGPolicy) -> None:
        self.sde.initialize_mf_drift(self.ts, policy)

    def sample_traj(self, policy: MFGPolicy, **kwargs) -> SampledTraj:
        return self.sde.sample_traj(self.ts, policy, **kwargs)

    def compute_state_cost(
        self, xs: torch.Tensor, ts: torch.Tensor, xs_all: torch.Tensor, log_p: torch.Tensor
    ) -> torch.Tensor:
        s_cost, mf_cost = self.state_cost_fn(xs, ts, log_p, xs_all)
        return s_cost + self.mf_coeff * mf_cost

    def get_bound(self, xs: torch.Tensor, direction: str, bound: str) -> Bound:
        t_index = get_bound_index(direction, bound)

        b, T, nx = xs.shape
        assert len(self.ts) == T and nx == self.opt.x_dim

        return self.ts[t_index], xs[:, t_index, ...], self.pbound[t_index]

    def get_init_term_bound(self, xs: torch.Tensor, direction: str) -> InitTermBound:
        return (*self.get_bound(xs, direction, "init"), *self.get_bound(xs, direction, "term"))

    def save_snapshot(self, policy_f, policy_b, stage: int) -> None:
        plot_logp = self.opt.x_dim == 2
        snapshot(self.opt, policy_f, policy_b, self, stage, plot_logp=plot_logp)
