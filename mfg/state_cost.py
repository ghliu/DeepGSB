from typing import Callable
import torch

StateCostFn = Callable[[torch.Tensor], torch.Tensor]
MFCostFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def build_state_cost_fn(problem_name: str, return_obs_cost_fn: bool = False):
    obstacle_cost_fn = get_obstacle_cost_fn(problem_name)
    mf_cost_fn = get_mf_cost_fn(problem_name)

    def state_cost_fn(xs, ts, logp, xs_all):
        obstacle_cost = obstacle_cost_fn(xs)
        mf_cost = mf_cost_fn(xs, logp, xs_all)
        return obstacle_cost, mf_cost

    return state_cost_fn, obstacle_cost_fn if return_obs_cost_fn else state_cost_fn

def get_obstacle_cost_fn(problem_name: str) -> StateCostFn:
    return {
        'GMM': obstacle_cost_fn_gmm,
        'Stunnel': obstacle_cost_fn_vneck,
        'Vneck':obstacle_cost_fn_stunnel,
        'opinion': zero_cost_fn,
        'opinion_1k': zero_cost_fn,
    }.get(problem_name)

def get_mf_cost_fn(problem_name: str) -> MFCostFn:
    return {
        'GMM': zero_cost_fn,
        'Stunnel': congestion_cost,
        'Vneck': entropy_cost,
        'opinion': entropy_cost,
        'opinion_1k': entropy_cost,
    }.get(problem_name)

##########################################################
################ mean-field cost functions ###############
##########################################################

def entropy_cost(xs: torch.Tensor, logp: torch.Tensor, xs_all: torch.Tensor) -> torch.Tensor:
    if logp is None:
        raise ValueError("Add this problem to logp_list.")

    return logp + 1

def congestion_cost(xs: torch.Tensor, logp: torch.Tensor, xs_all: torch.Tensor) -> torch.Tensor:
    assert xs.ndim == 2
    xs = xs.reshape(-1, xs_all.shape[1], *xs.shape[1:])

    assert xs.ndim == 3  # should be (batch_x, batch_y, x_dim)
    assert xs.shape[1:] == xs_all.shape[1:]

    dd = xs - xs_all  # batch_x, batch_t, xdim
    dist = torch.sum(dd * dd, dim=-1)  # batch_x, batch_t
    out = 2.0 / (dist + 1.0)
    cost = out.reshape(-1, *out.shape[2:])
    return cost

##########################################################
################## obstacle cost functions ###############
##########################################################

def zero_cost_fn(x: torch.Tensor, *args) -> torch.Tensor:
    return torch.zeros(*x.shape[:-1])

def gmm_obstacle_cfg():
    centers = [[6,6], [6,-6], [-6,-6]]
    radius = 1.5
    return centers, radius

def stunnel_obstacle_cfg():
    a, b, c = 20, 1, 90
    centers = [[5,6], [-5,-6]]
    return a, b, c, centers

def vneck_obstacle_cfg():
    c_sq = 0.36
    coef = 5
    return c_sq, coef

@torch.jit.script
def obstacle_cost_fn_gmm(xs: torch.Tensor) -> torch.Tensor:
    xs = xs.reshape(-1,xs.shape[-1])

    batch_xt = xs.shape[0]

    centers, radius = gmm_obstacle_cfg()

    obs1 = torch.tensor(centers[0]).repeat((batch_xt,1)).to(xs.device)
    obs2 = torch.tensor(centers[1]).repeat((batch_xt,1)).to(xs.device)
    obs3 = torch.tensor(centers[2]).repeat((batch_xt,1)).to(xs.device)

    dist1 = torch.norm(xs - obs1, dim=-1)
    dist2 = torch.norm(xs - obs2, dim=-1)
    dist3 = torch.norm(xs - obs3, dim=-1)

    cost1 = 1500 * (dist1 < radius)
    cost2 = 1500 * (dist2 < radius)
    cost3 = 1500 * (dist3 < radius)

    return cost1 + cost2 + cost3

@torch.jit.script
def obstacle_cost_fn_vneck(xs: torch.Tensor) -> torch.Tensor:

    a, b, c, centers = stunnel_obstacle_cfg()

    _xs = xs.reshape(-1,xs.shape[-1])
    x, y = _xs[:,0], _xs[:,1]

    d = a*(x-centers[0][0])**2 + b*(y-centers[0][1])**2
    c1 = 1500 * (d < c)

    d = a*(x-centers[1][0])**2 + b*(y-centers[1][1])**2
    c2 = 1500 * (d < c)

    return c1+c2

@torch.jit.script
def obstacle_cost_fn_stunnel(xs: torch.Tensor) -> torch.Tensor:
    c_sq, coef = vneck_obstacle_cfg()

    xs_sq = torch.square(xs)
    d = coef * xs_sq[..., 0] - xs_sq[..., 1]

    return 1500 * (d < -c_sq)
