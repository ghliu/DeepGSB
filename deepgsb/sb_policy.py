import logging
from typing import Tuple, TypeVar

import torch
from torch_ema import ExponentialMovingAverage

from mfg import MFGPolicy
from mfg.sde import SimpleSDE
from models import build_opinion_net_policy, build_toy_net_policy
from options import Options

from . import util

log = logging.getLogger(__file__)

_TorchModule = TypeVar("_TorchModule", bound=torch.nn.Module)


def build(opt: Options, dyn: SimpleSDE, direction: str) -> MFGPolicy:
    log.info("build {} SB model...".format(direction))

    # ------ build SB policy ------
    Ynet = _build_net(opt, "Y")

    if opt.sb_param == "critic":
        policy = SB_paramY(opt, direction, dyn, Ynet).to(opt.device)
    elif opt.sb_param == "actor-critic":
        Znet = _build_net(opt, "Z")
        policy = SB_paramYZ(opt, direction, dyn, Ynet, Znet).to(opt.device)
    else:
        raise RuntimeError(f"unknown sb net type {opt.sb_param}")

    log.info("# param in SBPolicy = {}".format(util.count_parameters(policy)))

    return policy


def _build_net(opt: Options, YorZ: str) -> torch.nn.Module:
    assert YorZ in ["Y", "Z"]

    if opt.policy_net == "toy":
        net = build_toy_net_policy(opt, YorZ)
    elif opt.policy_net == "opinion_net":
        net = build_opinion_net_policy(opt, YorZ)
    else:
        raise RuntimeError()
    return net


def _freeze(net: _TorchModule) -> _TorchModule:
    for p in net.parameters():
        p.requires_grad = False
    return net


def _activate(net: _TorchModule) -> _TorchModule:
    for p in net.parameters():
        p.requires_grad = True
    return net


class SchrodingerBridgeModel(MFGPolicy):
    def __init__(self, opt: Options, direction: str, dyn: SimpleSDE, use_t_idx: bool = True):
        super(SchrodingerBridgeModel, self).__init__(direction, dyn)
        self.opt = opt
        self.use_t_idx = use_t_idx
        self.g = opt.diffusion_std

    def _preprocessing(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # make sure t.shape = [batch]
        t = t.squeeze()
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        assert t.dim() == 1 and t.shape[0] == x.shape[0]

        if self.use_t_idx:
            t = t / self.opt.T * self.opt.interval
        return x, t

    def compute_value(self, x: torch.Tensor, t: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        raise NotImplementedError()

    def compute_policy(self, x, t) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, x, t):  # set default calling to policy
        return self.compute_policy(x, t)

    def freeze(self):
        self.eval()
        self.zero_grad()

    def activate(self):
        self.train()


class SB_paramY(SchrodingerBridgeModel):
    def __init__(self, opt: Options, direction: str, dyn: SimpleSDE, Ynet: torch.nn.Module):
        super(SB_paramY, self).__init__(opt, direction, dyn, use_t_idx=True)
        self.Ynet = Ynet
        self.emaY = ExponentialMovingAverage(self.Ynet.parameters(), decay=opt.ema)
        self.param = "critic"

    def compute_value(self, x: torch.Tensor, t: torch.Tensor, use_ema: bool = False) -> torch.Tensor:
        x, t = self._preprocessing(x, t)
        if use_ema:
            with self.emaY.average_parameters():
                return self._standardizeYnet(self.Ynet(x, t).squeeze())
        return self._standardizeYnet(self.Ynet(x, t).squeeze())

    def compute_value_grad(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x, t = self._preprocessing(x, t)
        requires_grad = x.requires_grad
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self._standardizeYnet(self.Ynet(x, t))
            out = torch.autograd.grad(y.sum(), x, create_graph=self.training)[0]
        x.requires_grad_(requires_grad)  # restore original setup
        if not self.training:
            self.zero_grad()  # out = out.detach()
        return out

    def compute_policy(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Z = g * nabla Y
        return self.g * self.compute_value_grad(x, t)

    def _standardizeYnet(self, out: torch.Tensor) -> torch.Tensor:
        # standardize the Ynet output
        return out / self.g

    def freeze(self) -> None:
        self.Ynet = _freeze(self.Ynet)
        super(SB_paramY, self).freeze()

    def activate(self) -> None:
        self.Ynet = _activate(self.Ynet)
        super(SB_paramY, self).activate()

    def get_ema(self) -> ExponentialMovingAverage:
        return self.emaY

    def update_ema(self) -> None:
        self.emaY.update()

    def state_dict(self):
        return {
            "Ynet": self.Ynet.state_dict(),
            "emaY": self.emaY.state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        self.Ynet.load_state_dict(state_dict["Ynet"])
        self.emaY.load_state_dict(state_dict["emaY"])


class SB_paramYZ(SB_paramY):
    def __init__(self, opt: Options, direction: str, dyn: SimpleSDE, Ynet: torch.nn.Module, Znet: torch.nn.Module):
        super(SB_paramYZ, self).__init__(opt, direction, dyn, Ynet)
        self.Znet = Znet
        self.emaZ = ExponentialMovingAverage(self.Znet.parameters(), decay=opt.ema)
        self.param = "actor-critic"

    def compute_policy(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x, t = self._preprocessing(x, t)
        return self.Znet(x, t)

    def freeze(self) -> None:
        self.Znet = _freeze(self.Znet)
        super(SB_paramYZ, self).freeze()

    def activate(self, only_Ynet: bool = False) -> None:
        self.Znet = _activate(self.Znet) if not only_Ynet else _freeze(self.Znet)
        super(SB_paramYZ, self).activate()

    def get_ema(self) -> ExponentialMovingAverage:
        return self.emaZ

    def update_ema(self, only_Ynet: bool = False) -> None:
        if not only_Ynet:
            self.emaZ.update()
        super(SB_paramYZ, self).update_ema()

    def state_dict(self):
        sdict = {
            "Znet": self.Znet.state_dict(),
            "emaZ": self.emaZ.state_dict(),
        }
        sdict.update(super(SB_paramYZ, self).state_dict())
        return sdict

    def load_state_dict(self, state_dict):
        self.Znet.load_state_dict(state_dict["Znet"])
        self.emaZ.load_state_dict(state_dict["emaZ"])
        super(SB_paramYZ, self).load_state_dict(state_dict)
