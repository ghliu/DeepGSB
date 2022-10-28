import argparse
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

import configs


@dataclass
class Options:
    problem_name: str
    seed: int
    gpu: int
    load: Optional[str]
    dir: str
    group: str
    name: str
    log_fn: Optional[str]
    log_tb: bool
    cpu: bool
    t0: float
    T: float
    interval: int
    policy_net: str
    diffusion_std: float
    train_bs_x: int
    num_stage: int
    num_itr: int
    samp_bs: int
    samp_method: str
    rb_bs_x: int
    MF_cost: float
    lr: float
    lr_y: Optional[float]
    lr_gamma: float
    lr_step: int
    l2_norm: float
    optimizer: str
    noise_type: str
    ema: float
    snapshot_freq: int
    ckpt_freq: int
    sb_param: str
    use_rb_loss: bool
    multistep_td: bool
    buffer_size: int
    weighted_loss: bool
    x_dim: int
    device: str
    ckpt_path: str
    eval_path: str
    log_dir: str
    # Additional options set in problem config.
    weights: Optional[Dict[str, float]] = None


def set():
    # fmt: off
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-name",   type=str)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--gpu",            type=int,   default=0,        help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,     help="load the checkpoints")
    parser.add_argument("--dir",            type=str,   default=None,     help="directory name to save the experiments under results/")
    parser.add_argument("--group",          type=str,   default='0',      help="father node of directionary for saving checkpoint")
    parser.add_argument("--name",           type=str,   default=None,     help="son node of directionary for saving checkpoint")
    parser.add_argument("--log-fn",         type=str,   default=None,     help="name of tensorboard logging")
    parser.add_argument("--log-tb",         action="store_true",          help="logging with tensorboard")
    parser.add_argument("--cpu",            action="store_true",          help="use cpu device")

    # --------------- DeepGSB & MFG ---------------
    parser.add_argument("--t0",             type=float, default=0.0,      help="time integral start time")
    parser.add_argument("--T",              type=float, default=1.0,      help="time integral end time")
    parser.add_argument("--interval",       type=int,   default=100,      help="number of interval")
    parser.add_argument("--policy-net",     type=str,                     help="model class of policy network")
    parser.add_argument("--diffusion-std",  type=float, default=1.0,      help="diffusion scalar in SDE")
    parser.add_argument("--sb-param",       type=str,  choices=['actor-critic', 'critic'])
    parser.add_argument("--MF-cost",        type=float, default=0.0,      help="coefficient of MF cost")

    # --------------- training & sampling ---------------
    parser.add_argument("--train-bs-x",     type=int,                     help="batch size for sampling data")
    parser.add_argument("--num-stage",      type=int,                     help="number of stage")
    parser.add_argument("--num-itr",        type=int,                     help="number of training iterations (for each stage)")
    parser.add_argument("--samp-bs",        type=int,                     help="batch size for all trajectory sampling purposes")
    parser.add_argument("--samp-method",    type=str,  default='jacobi',  choices=['jacobi','gauss']) # gauss seidel
    parser.add_argument("--rb-bs-x",        type=int,                     help="batch size when sampling from replay buffer")
    parser.add_argument("--use-rb-loss",    action="store_true",          help="whether or not to use the replay buffer loss")
    parser.add_argument("--multistep-td",   action="store_true",          help="whether or not to use the multi-step TD loss")
    parser.add_argument("--buffer-size",    type=int,  default=20000,     help="the maximum size of replay buffer")
    parser.add_argument("--weighted-loss",  action="store_true",          help="whether or not to reweight the combined loss")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--lr",             type=float,                   help="learning rate for Znet")
    parser.add_argument("--lr-y",           type=float, default=None,     help="learning rate for Ynet")
    parser.add_argument("--lr-gamma",       type=float, default=1.0,      help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,     help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0,      help="weight decay rate")
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--noise-type",     type=str,   default='gaussian', choices=['gaussian','rademacher'], help='choose noise type to approximate Trace term')
    parser.add_argument("--ema",            type=float, default=0.99)

    # ---------------- evaluation ----------------
    parser.add_argument("--snapshot-freq",  type=int,   default=1,        help="snapshot frequency w.r.t stages")
    parser.add_argument("--ckpt-freq",      type=int,   default=1,        help="checkpoint saving frequency w.r.t stages")

    # fmt: on

    problem_name = parser.parse_args().problem_name
    sb_param = parser.parse_args().sb_param

    parser.set_defaults(**configs.get_default(problem_name, sb_param))
    opt = parser.parse_args()
    # ========= seed & torch setup =========
    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # torch.autograd.set_detect_anomaly(True)

    # ========= auto setup & path handle =========
    opt.device = "cuda:" + str(opt.gpu)

    if opt.name is None:
        opt.name = opt.dir

    opt.ckpt_path = os.path.join("checkpoint", opt.group, opt.name)
    os.makedirs(opt.ckpt_path, exist_ok=True)
    if opt.snapshot_freq:
        opt.eval_path = os.path.join("results", opt.dir)
        os.makedirs(os.path.join(opt.eval_path, "forward"), exist_ok=True)
        os.makedirs(os.path.join(opt.eval_path, "backward"), exist_ok=True)
        os.makedirs(os.path.join(opt.eval_path, "logp"), exist_ok=True)
        if "opinion" in opt.problem_name:
            os.makedirs(
                os.path.join(opt.eval_path, "directional_sim_forward"), exist_ok=True
            )
            os.makedirs(
                os.path.join(opt.eval_path, "directional_sim_backward"), exist_ok=True
            )

    if opt.log_tb:
        opt.log_dir = os.path.join(
            "runs", opt.dir
        )  # if opt.log_fn is not None else None
        if os.path.exists(opt.log_dir):
            shutil.rmtree(opt.log_dir)  # remove folder & its files

    opt = Options(**vars(opt))
    return opt
