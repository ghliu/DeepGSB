import logging
from typing import TYPE_CHECKING, Iterable, Tuple

import termcolor
import torch
from easydict import EasyDict as edict

if TYPE_CHECKING:
    from deepgsb.deepgsb import DeepGSB
    from options import Options

log = logging.getLogger(__file__)


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_stage(opt, stage):
    """ Determine what metrics to evaluate for the current stage,
    if metrics is None, use the frequency in opt to decide it.
    """
    match = lambda freq: (freq>0 and stage%freq==0)
    return [match(opt.snapshot_freq), match(opt.ckpt_freq)]

def get_time(sec: float) -> Tuple[int, int, float]:
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h, m, s

def flatten_dim01(x: torch.Tensor) -> torch.Tensor:
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

def unflatten_dim01(x: torch.Tensor, dim01) -> torch.Tensor:
    # (dim0x1, *dim2) --> (dim0, dim1, *dim2)
    return x.reshape(*dim01, *x.shape[1:])

def restore_checkpoint(opt: "Options", runner: "DeepGSB", load_name: str) -> None:
    assert load_name is not None
    log.info("#loading checkpoint {}...".format(load_name))

    full_keys = ['z_f','optimizer_f','z_b','optimizer_b']

    with torch.cuda.device(opt.gpu):
        checkpoint = torch.load(load_name,map_location=opt.device)
        ckpt_keys=[*checkpoint.keys()]

        for k in ckpt_keys:
            thing = getattr(runner,k)
            if hasattr(thing, "load_state_dict"):
                getattr(runner,k).load_state_dict(checkpoint[k])
            else:
                setattr(runner, k, checkpoint[k])

    if len(full_keys) != len(ckpt_keys):
        missing_keys = { k for k in set(full_keys) - set(ckpt_keys) }
        extra_keys = {k for k in set(ckpt_keys) - set(full_keys)}

        if len(missing_keys) > 0:
            log.warning("Does not load model for {}, check is it correct".format(missing_keys))
        else:
            log.warning("Loaded extra keys not found in full_keys: {}".format(extra_keys))

    else:
        log.info('#successfully loaded all the modules')

    # runner.ema_f.copy_to()
    # runner.ema_b.copy_to()
    # print(green('#loading form ema shadow parameter for polices'))
    log.info("#######summary of checkpoint##########")

def save_checkpoint(opt: "Options", runner: "DeepGSB", keys: Iterable[str], stage_it: int) -> None:
    checkpoint = {}
    fn = opt.ckpt_path + "/stage_{0:04}.npz".format(stage_it)
    with torch.cuda.device(opt.gpu):
        for k in keys:
            variable = getattr(runner, k)
            if hasattr(variable, "state_dict"):
                checkpoint[k] = variable.state_dict()
            else:
                checkpoint[k] = variable

        torch.save(checkpoint, fn)
    print(green("checkpoint saved: {}".format(fn)))

def get_loss_str(loss) -> str:
    if isinstance(loss, edict):
        return ' '.join([ f'({key})' + f'{val.item():+2.3f}' for key, val in loss.items()])
    else:
        return f'{loss.item():+.4f}'


