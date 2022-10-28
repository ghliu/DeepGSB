from __future__ import absolute_import, division, print_function, unicode_literals

import os

import torch
import numpy as np

from . import util
from . import opinion_lib
from .state_cost import gmm_obstacle_cfg , vneck_obstacle_cfg, stunnel_obstacle_cfg

import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from matplotlib.patches import Circle, Ellipse

from ipdb import set_trace as debug


def to_numpy(t):
    return t.detach().cpu().numpy()

def get_lims(opt):
    return {
        "GMM":          [-16.25, 16.25],
        "Stunnel":      [-15, 15],
        "Vneck":        [-10, 10],
        "opinion":      [-10, 10],
        "opinion_1k":   [-10, 10],
    }.get(opt.problem_name)

def get_ylims(opt):
    return {
        "GMM":          [-16.25, 16.25],
        "Stunnel":      [-10, 10],
        "Vneck":        [-5, 5],
        "opinion":      [-10, 10],
        "opinion_1k":   [-10, 10],
    }.get(opt.problem_name)

def get_colors(n_snapshot):
    # assert n_snapshot % 2 == 1
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["b","r"])
    colors = cm1(np.linspace(0.0, 1.0, n_snapshot))
    return colors

def create_mesh(opt, n_grid, lims, convert_to_numpy=True):
    import warnings

    _x = torch.linspace(*(lims+[n_grid]))

    # Suppress warning about indexing arg becoming required.
    with warnings.catch_warnings():
        X, Y = torch.meshgrid(_x, _x)

    xs = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(opt.device)
    return [to_numpy(X), to_numpy(Y), xs] if convert_to_numpy else [X, Y, xs]

def get_func_mesher(opt, ts, grid_n, func, out_dim=1):
    if func is None: return None
    lims = get_lims(opt)
    X1, X2, XS = create_mesh(opt, grid_n, lims)
    out_shape = [grid_n,grid_n] if out_dim==1 else [grid_n,grid_n,out_dim]

    def mesher(idx):
        # print(func, ts[idx])
        arg_xs = XS.detach()
        arg_ts = ts[idx].repeat(grid_n**2).detach()
        fn_out = func(arg_xs, arg_ts)
        return X1, X2, to_numpy(fn_out.reshape(*out_shape))

    return mesher

def plot_obs(opt, ax, scale=1., zorder=0):
    if opt.problem_name == 'GMM':
        centers, radius = gmm_obstacle_cfg()
        for c in centers:
            circle = Circle(xy=np.array(c), radius=radius, zorder=zorder)

            ax.add_artist(circle)
            circle.set_clip_box(ax.bbox)
            circle.set_facecolor("darkgray")
            circle.set_edgecolor(None)

    elif opt.problem_name == 'Vneck':
        c_sq, coef = vneck_obstacle_cfg()
        x = np.linspace(-6,6,100)
        y1 = np.sqrt(c_sq + coef * np.square(x))
        y2 = np.ones_like(x) * y1[0]

        ax.fill_between(x, y1, y2, color="darkgray", edgecolor=None, zorder=zorder)
        ax.fill_between(x, -y1, -y2, color="darkgray", edgecolor=None, zorder=zorder)

    elif opt.problem_name == 'Stunnel':
        a, b, cc, centers = stunnel_obstacle_cfg()
        for c in centers:
            elp = Ellipse(
                xy=np.array(c)*scale, width=2*np.sqrt(cc/a)*scale, height=2*np.sqrt(cc/b)*scale, zorder=zorder
            )

            ax.add_artist(elp)
            elp.set_clip_box(ax.bbox)
            elp.set_facecolor("darkgray")
            elp.set_edgecolor(None)

def setup_ax(ax, title, xlims, ylims, title_fontsize=18):
    ax.axis('equal')
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)

def plot_traj_snapshot(opt, xs, axes, sample_steps, titles, y_mesher=None):

    n_snapshot = len(axes)
    assert len(sample_steps) == len(titles) == n_snapshot

    if sample_steps is None:
        sample_steps = np.linspace(0, xs.shape[1]-1, n_snapshot).astype(int)

    xlims = get_lims(opt)
    ylims = get_ylims(opt)

    colors = get_colors(n_snapshot)

    for ax, step, title, color in zip(axes, sample_steps, titles, colors):
        plot_obs(opt, ax, zorder=0)

        ax.scatter(xs[:,step,0],xs[:,step,1], s=1.5, color=color, alpha=0.5, zorder=1)
        if y_mesher is not None:
            cp = ax.contour(*y_mesher(step), levels=10, cmap="copper", linewidths=1, zorder=2)
            ax.clabel(cp, inline=True, fontsize=6)
        setup_ax(ax, title, xlims, ylims)

def plot_directional_sim(opt, ax, stage, xs_term) -> None:
    n_est = 5000
    directional_sim = opinion_lib.est_directional_similarity(xs_term, n_est)
    assert directional_sim.shape == (n_est, )

    directional_sim = to_numpy(directional_sim)

    bins = 100
    ax.hist(directional_sim, bins=bins)
    ax.set(xlabel="Directional Similarity", title="Stage={:3}".format(stage), xlim=(0., 1.))


def get_fig_axes_steps(interval, n_snapshot=5, ax_length_in: float = 4):
    n_row, n_col = 1, n_snapshot
    figsize = (n_col*ax_length_in, n_row*ax_length_in)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axes = fig.subplots(n_row, n_col)
    steps = np.linspace(0, interval-1, n_snapshot).astype(int)

    return fig, axes, steps

@torch.no_grad()
def sample_traj(opt, mfg, ts, policy_f, policy_b, plot_dim):
    xs_f, _, _, _ = mfg.sde.sample_traj(ts, policy_f)
    xs_b, _, _, _ = mfg.sde.sample_traj(ts, policy_b)
    util.assert_zero_grads(policy_f)
    util.assert_zero_grads(policy_b)

    if opt.x_dim > 2:
        xs_f, xs_b = util.proj_pca(xs_f, xs_b, reverse=False)

    xs_f_np, xs_b_np = to_numpy(xs_f[..., plot_dim]), to_numpy(xs_b[..., plot_dim])

    return xs_f, xs_b, xs_f_np, xs_b_np

@torch.no_grad()
def snapshot(opt, policy_f, policy_b, mfg, stage, plot_logp=False, plot_dim=[0,1]):

    # sample forward & backward trajs
    ts = mfg.ts

    xs_f, xs_b, xs_f_np, xs_b_np = sample_traj(opt, mfg, ts, policy_f, policy_b, plot_dim)

    interval = len(ts)

    for xs, policy in zip([xs_f_np, xs_b_np], [policy_f, policy_b]):

        # prepare plotting
        titles = [r'$t$ = 0', r'$t$ = 0.25$T$', r'$t$ = 0.50$T$', r'$t$ = 0.75$T$', r'$t = T$']
        fig, axes, sample_steps = get_fig_axes_steps(interval, n_snapshot=len(titles))
        assert len(titles) == len(axes) == len(sample_steps)

        # plot policy and value
        y_mesher = get_func_mesher(opt, ts, 200, policy.compute_value) if opt.x_dim == 2 else None
        plot_traj_snapshot(
            opt, xs, axes, sample_steps, titles, y_mesher=y_mesher,
        )

        plt.savefig(os.path.join('results', opt.dir, policy.direction, f'stage{stage}.pdf'))
        plt.close(fig)

    if "opinion" in opt.problem_name:
        for xs_term, policy in zip([xs_f[:,-1], xs_b[:,0]], [policy_f, policy_b]):
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            plot_directional_sim(opt, ax, stage, xs_term)
            plt.savefig(os.path.join(
                'results', opt.dir, f"directional_sim_{policy.direction}", f'stage{stage}.pdf'
            ))
            plt.close(fig)

    if plot_logp:
        assert opt.x_dim == 2
        def logp_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            y_f = policy_f.compute_value(x, t)
            y_b = policy_b.compute_value(x, t)
            return y_f + y_b

        # prepare plotting
        titles = [r'$t$ = 0', r'$t$ = 0.25$T$', r'$t$ = 0.50$T$', r'$t$ = 0.75$T$', r'$t = T$']
        fig, axes, sample_steps = get_fig_axes_steps(interval, n_snapshot=len(titles))
        assert len(titles) == len(axes) == len(sample_steps)

        # plot logp
        logp_mesher = get_func_mesher(opt, ts, 200, logp_fn)
        plot_traj_snapshot(
            opt, xs_f_np, axes, sample_steps, titles, y_mesher=logp_mesher,
        )

        plt.savefig(os.path.join('results', opt.dir, "logp", f'stage{stage}.pdf'))
        plt.close(fig)

    return xs_f, xs_b
