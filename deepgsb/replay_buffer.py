from typing import Tuple

import torch
from easydict import EasyDict as edict

from options import Options


class Buffer:
    def __init__(self, opt: Options, direction: str):
        self.opt = opt
        self.max_samples = opt.buffer_size
        self.direction = direction

        self.nx = opt.x_dim

        self.it = 0
        self.n_samples = 0
        self.samples = torch.empty(self.max_samples, self.opt.interval, 3 * self.nx, device="cpu")

    def __len__(self) -> int:
        return self.n_samples

    def append(self, datas: edict) -> None:
        xs, zs, dws = datas.xs, datas.zs, datas.ws
        assert xs.shape == zs.shape == dws.shape

        it, max_samples, batch = self.it, self.max_samples, xs.shape[0]
        sample = torch.cat([xs, zs, dws], dim=-1).detach().cpu()
        self.samples[it : it + batch] = sample[0 : min(batch, max_samples - it), ...]
        if batch > max_samples - it:
            _it = batch - (max_samples - it)
            self.samples[0:_it] = sample[-_it:, ...]
            assert ((it + batch) % max_samples) == _it

        self.it = (it + batch) % max_samples
        self.n_samples = min(self.n_samples + batch, max_samples)

    def clear(self) -> None:
        self.samples = torch.empty_like(self.samples)
        self.n_samples = self.it = 0

    def sample_traj(self, batch_x: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        samp_x_idx = torch.randint(self.n_samples, (batch_x,))
        samples = self.samples[samp_x_idx].to(self.opt.device)

        xs = samples[..., 0 * self.nx : 1 * self.nx].detach()  # (batch_x, T, nx)
        zs = samples[..., 1 * self.nx : 2 * self.nx].detach()  # (batch_x, T, nx)
        ws = samples[..., 2 * self.nx : 3 * self.nx].detach()  # (batch_x, T, nx)

        return xs, zs, ws
