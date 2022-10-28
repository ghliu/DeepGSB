from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pathlib
import sys

import colored_traceback.always
import torch

import options
from git_utils import log_git_info
from deepgsb import DeepGSB
from mfg import MFG

from rich.console import Console
from rich.logging import RichHandler
from options import Options

def setup_logger(log_dir: pathlib.Path) -> None:
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = open(log_dir / "log.txt", "w")
    file_console = Console(file=log_file, width=150)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[RichHandler(), RichHandler(console=file_console)],
    )

def run(opt: Options):
    log = logging.getLogger(__name__)
    log.info("=======================================================")
    log.info("        Deep Generalized Schrodinger Bridge ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))

    mfg = MFG(opt)
    deepgsb = DeepGSB(opt, mfg)
    deepgsb.train(opt)

def main():
    print("setting configurations...")
    opt = options.set()

    run_dir = pathlib.Path("results") / opt.dir
    setup_logger(run_dir)
    log_git_info(run_dir)

    if not opt.cpu:
        with torch.cuda.device(opt.gpu):
            run(opt)
    else:
        run(opt)


if __name__ == "__main__":
    main()
