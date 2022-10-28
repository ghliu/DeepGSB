from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pathlib
import pickle
import sys
import os
import argparse

import ipdb
import torch
import numpy as np

from mfg import MFG
from deepgsb import DeepGSB

from rich.logging import RichHandler

import matplotlib.pyplot as plt

import imageio
from mfg.plotting import *

from mfg.opinion_lib import est_directional_similarity

from ipdb import set_trace as debug

def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        force=True,
        handlers=[RichHandler(),],
    )

def restore_ckpt_option(opt):
    assert opt.load is not None
    ckpt_path = pathlib.Path(opt.load)
    assert ckpt_path.exists()

    options_pkl_path = ckpt_path.parent / "options.pkl"
    assert options_pkl_path.exists()

    # Load options pkl and overwrite the load.
    with open(options_pkl_path, "rb") as f:
        ckpt_options = pickle.load(f)
    ckpt_options.load = opt.load

    return ckpt_options

def build_steps(direction, interval, total_steps=100):
    steps = np.linspace(0, interval-1, total_steps).astype(int)
    if direction == "backward":
        steps = np.flip(steps)
    return steps

def get_title(opt, direction):
    return {
        "GMM":          "GMM",
        "Stunnel":      "S-tunnel",
        "Vneck":        "V-neck",
        "opinion":      "Opinion",
        "opinion_1k":   "Opinion 1k",
    }.get(opt.problem_name) + f" ({direction} policy)"

@torch.no_grad()
def plot_directional_sim(opt, xs, ax) -> None:

    n_est = 5000
    directional_sim = est_directional_similarity(xs, n_est)
    assert directional_sim.shape == (n_est, )

    directional_sim = to_numpy(directional_sim)

    bins = 15
    _, _, patches = ax.hist(directional_sim, bins=bins, )

    colors = plt.cm.coolwarm(np.linspace(1.0, 0.0, bins))

    for c, p in zip(colors, patches):
        plt.setp(p, 'facecolor', c)

    ymax = 1000 if opt.x_dim == 2 else 2000
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

@torch.no_grad()
def make_gif(opt, policy_f, policy_b, mfg, gif_name=None, plot_dim=[0,1]):

    file_path = os.path.join(".tmp", opt.group, opt.name)
    os.makedirs(file_path, exist_ok=True)

    xs_f, xs_b, xs_f_np, xs_b_np = sample_traj(opt, mfg, mfg.ts, policy_f, policy_b, plot_dim)

    xlims = get_lims(opt)
    ylims = get_ylims(opt)

    filenames = []
    for xs, xs_np, policy in zip([xs_f, xs_b], [xs_f_np, xs_b_np], [policy_f, policy_b]):

        if "opinion" in opt.problem_name and policy.direction == "backward":
            # skip backward opinion due to the mean-field drift
            continue

        y_mesher = get_func_mesher(opt, mfg.ts, 200, policy.compute_value) if opt.x_dim == 2 else None

        colors = get_colors(xs_np.shape[1])
        title = get_title(opt, policy.direction)
        # title = "Polarize (before apply DeepGSB)"
        # title = "Depolarize (after apply DeepGSB)"

        steps = build_steps(policy.direction, xs_np.shape[1], total_steps=100)
        for step in steps:
            # prepare plotting
            fig = plt.figure(figsize=(3,3), constrained_layout=True)
            ax = fig.subplots(1, 1)

            # plot policy and value
            plot_obs(opt, ax, zorder=0)
            ax.scatter(xs_np[:,step,0], xs_np[:,step,1], s=1.5, color=colors[step], alpha=0.5, zorder=1)
            if y_mesher is not None:
                cp = ax.contour(*y_mesher(step), levels=10, cmap="copper", linewidths=1, zorder=2)
                ax.clabel(cp, inline=True, fontsize=6)
            setup_ax(ax, title, xlims, ylims, title_fontsize=12)

            if "opinion" in opt.problem_name:
                axins = ax.inset_axes([0.59, 0.01, 0.4, 0.4])
                plot_directional_sim(opt, xs[:,step], axins)
                axins.text(
                    0.5, 0.9, r"$t$=" + f"{step/xs.shape[1]:0.2f}" + r"$T$",
                    transform=axins.transAxes, fontsize=7, ha='center', va='center'
                )

            # save fig
            filename = f"{file_path}/{policy.direction}_{str(step).zfill(3)}.png"
            filenames.append(filename)
            plt.savefig(filename)
            plt.close(fig)

    # build gif
    images = list(map(lambda filename: imageio.imread(filename), filenames))
    imageio.mimsave(f'{gif_name or opt.problem_name}.gif', images, duration=0.04) # modify the frame duration as needed

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

def run(ckpt_options, gif_name=None):
    mfg = MFG(ckpt_options)
    deepgsb = DeepGSB(ckpt_options, mfg, save_opt=False)
    make_gif(ckpt_options, deepgsb.z_f, deepgsb.z_b, mfg, gif_name=gif_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load",   type=str)
    parser.add_argument("--name",   type=str, default=None)
    arg = parser.parse_args()

    setup_logger()
    log = logging.getLogger(__name__)
    log.info("Command used:\n{}".format(" ".join(sys.argv)))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    ckpt_options = restore_ckpt_option(arg)

    if not ckpt_options.cpu:
        with torch.cuda.device(ckpt_options.gpu):
            run(ckpt_options, gif_name=arg.name)
    else:
        run(ckpt_options, gif_name=arg.name)

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
