import logging
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.distributions as td

log = logging.getLogger(__file__)


class Sampler:
    def __init__(self, distribution: td.Distribution, batch_size: int, device: str):
        self.distribution = distribution
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)

    def sample(self, batch: Optional[int] = None) -> torch.Tensor:
        if batch is None:
            batch = self.batch_size
        return self.distribution.sample([batch]).to(self.device)


class ProblemDists(NamedTuple):
    p0: Sampler
    pT: Sampler


def build_constraint(problem_name: str, batch_size: int, device: str) -> ProblemDists:
    log.info("build distributional constraints ...")

    distribution_builder = {
        "GMM": gmm_builder,
        "Stunnel": stunnel_builder,
        "Vneck": vneck_builder,
        "opinion": opinion_builder,
        "opinion_1k": opinion_1k_builder,
    }.get(problem_name)

    return distribution_builder(batch_size, device)


def gmm_builder(batch_size: int, device: str) -> ProblemDists:

    # ----- pT -----
    radius, num = 16, 8
    arc = 2 * np.pi / num
    xs = [np.cos(arc * idx) * radius for idx in range(num)]
    ys = [np.sin(arc * idx) * radius for idx in range(num)]

    mix = td.Categorical(
        torch.ones(
            num,
        )
    )
    comp = td.Independent(td.Normal(torch.Tensor([[x, y] for x, y in zip(xs, ys)]), torch.ones(num, 2)), 1)
    dist = td.MixtureSameFamily(mix, comp)
    pT = Sampler(dist, batch_size, device)

    # ----- p0 -----
    dist = td.MultivariateNormal(torch.zeros(2), torch.eye(2))
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)


def vneck_builder(batch_size: int, device: str) -> ProblemDists:

    # ----- pT -----
    dist = td.MultivariateNormal(torch.Tensor([7, 0]), 0.2 * torch.eye(2))
    pT = Sampler(dist, batch_size, device)

    # ----- p0 -----
    dist = td.MultivariateNormal(-torch.Tensor([7, 0]), 0.2 * torch.eye(2))
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)


def stunnel_builder(batch_size: int, device: str) -> ProblemDists:

    # ----- pT -----
    dist = td.MultivariateNormal(torch.Tensor([11, 1]), 0.5 * torch.eye(2))
    pT = Sampler(dist, batch_size, device)

    # ----- p0 -----
    dist = td.MultivariateNormal(-torch.Tensor([11, 1]), 0.5 * torch.eye(2))
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)


def opinion_builder(batch_size: int, device: str) -> ProblemDists:

    p0_std = 0.25
    pT_std = 3.0

    # ----- p0 -----
    mu0 = torch.zeros(2)
    covar0 = p0_std * torch.eye(2)

    # Start with kind-of polarized opinions.
    covar0[0, 0] = 0.5

    # ----- pT -----
    muT = torch.zeros(2)
    # Want to finish with more homogenous opinions.
    covarT = pT_std * torch.eye(2)

    dist = td.MultivariateNormal(muT, covarT)
    pT = Sampler(dist, batch_size, device)

    dist = td.MultivariateNormal(mu0, covar0)
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)


def opinion_1k_builder(batch_size: int, device: str) -> ProblemDists:

    p0_std = 0.25
    pT_std = 3.0

    # ----- p0 -----
    mu0 = torch.zeros(1000)
    covar0 = p0_std * torch.eye(1000)

    # Start with kind-of polarized opinions.
    covar0[0, 0] = 4.0

    # ----- pT -----
    muT = torch.zeros(1000)
    # Want to finish with more homogenous opinions.
    covarT = pT_std * torch.eye(1000)

    dist = td.MultivariateNormal(muT, covarT)
    pT = Sampler(dist, batch_size, device)

    dist = td.MultivariateNormal(mu0, covar0)
    p0 = Sampler(dist, batch_size, device)

    return ProblemDists(p0, pT)
