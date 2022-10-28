import torch
import torch.nn as nn

from options import Options
from .util import SiLU, timestep_embedding, ResNet_FC


class OpinionYImpl(torch.nn.Module):
    def __init__(self, data_dim: int, time_embed_dim: int, hid: int, out_hid: int):
        super().__init__()

        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(data_dim, hid, num_res_blocks=5)

        self.out_module = nn.Sequential(
            nn.Linear(hid + hid, out_hid),
            SiLU(),
            nn.Linear(out_hid, out_hid),
            SiLU(),
            nn.Linear(out_hid, 1),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)

        t_out = t_out.expand(x_out.shape)

        out = self.out_module(torch.cat([x_out, t_out], dim=-1))

        return out


class OpinionY(torch.nn.Module):
    def __init__(self, data_dim: int = 1000, hid: int = 128, out_hid: int = 128, time_embed_dim: int = 128):
        super(OpinionY,self).__init__()

        self.time_embed_dim = time_embed_dim
        self.yt_impl = OpinionYImpl(data_dim, time_embed_dim, hid, out_hid)
        self.yt_impl = torch.jit.script(self.yt_impl)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x:torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        t_emb  = timestep_embedding(t, self.time_embed_dim)

        out = self.yt_impl(x, t_emb)

        return out

class OpinionZImpl(torch.nn.Module):
    def __init__(self, data_dim: int, time_embed_dim: int, hid: int):
        super().__init__()

        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(data_dim, hid, num_res_blocks=5)

        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(x_out+t_out)
        return out


class OpinionZ(torch.nn.Module):
    def __init__(self, data_dim=1000, hidden_dim=256, time_embed_dim=128):
        super(OpinionZ,self).__init__()

        self.time_embed_dim = time_embed_dim
        self.z_impl = OpinionZImpl(data_dim, time_embed_dim, hidden_dim)
        self.z_impl = torch.jit.script(self.z_impl)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t = t[None]

        t_emb = timestep_embedding(t, self.time_embed_dim)
        out = self.z_impl(x, t_emb)

        return out

def build_opinion_net_policy(opt: Options, YorZ: str) -> torch.nn.Module:
    assert opt.x_dim == 1000
    # 2nets: {"hid":200, "out_hid":400, "time_embed_dim":256}
    return {
        "Y": OpinionY,
        "Z": OpinionZ,
    }.get(YorZ)()
