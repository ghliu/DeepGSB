import copy
import gc
import logging
import os
import pickle
import time
from typing import Dict, Optional, Tuple

import torch
from easydict import EasyDict as edict
from torch.optim import SGD, Adagrad, Adam, AdamW, RMSprop, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from mfg import MFG, MFGPolicy
from options import Options

from . import eval_metrics, loss_lib, sb_policy, util
from .replay_buffer import Buffer

log = logging.getLogger(__file__)

OptSchedPair = Tuple[torch.optim.Optimizer, lr_scheduler._LRScheduler]


def build_optimizer_sched(opt: Options, policy) -> OptSchedPair:
    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr,
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    if policy.param == "actor-critic":
        optimizer = optim_name([
            {'params': policy.Znet.parameters()}, # use original optim_dict
            {'params': policy.Ynet.parameters(), 'lr': opt.lr_y},
        ], **optim_dict)
    elif policy.param == "critic":
        optimizer = optim_name(policy.parameters(), **optim_dict)
    else:
        raise ValueError(f"Expected either actor-critic or critic, got {policy.param}")

    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    else:
        sched = None

    return optimizer, sched

def get_grad_loss_norm(opt: Options) -> str:
    problem_name: str = opt.problem_name

    if problem_name in ["GMM", "opinion", "opinion_1k"]:
        # Use L1 on GMM.
        return "l1"
    if problem_name in []:
        # Use L2 loss for ??
        return "l2"

    # Otherwise, default to Huber loss.
    return "huber"

class DeepGSB:
    def __init__(self, opt: Options, mfg: MFG, save_opt: bool = True):
        super(DeepGSB, self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path + "/options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        self.start_time = time.time()

        self.mfg = mfg

        # build forward (z_f) and backward (z_b) policies
        self.z_f = sb_policy.build(opt, mfg.sde, 'forward')  # p0 -> pT
        self.z_b = sb_policy.build(opt, mfg.sde, 'backward') # p0 -> pT
        if mfg.uses_mf_drift():
            mfg.initialize_mf_drift(self.z_f)

        self.optimizer_f, self.sched_f = build_optimizer_sched(opt, self.z_f)
        self.optimizer_b, self.sched_b = build_optimizer_sched(opt, self.z_b)

        self.buffer_f = Buffer(opt, 'forward') if opt.use_rb_loss else None
        self.buffer_b = Buffer(opt, 'backward') if opt.use_rb_loss else None

        self.it_f = self.it_b = 0
        if opt.log_tb: # tensorboard related things
            self.writer=SummaryWriter(log_dir=opt.log_dir)

        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)

    @property
    def is_critic_param(self) -> bool:
        return self.z_f.param == self.z_b.param == "critic"

    @property
    def is_actor_critic_param(self) -> bool:
        return self.z_f.param == self.z_b.param == "actor-critic"

    def get_count(self, direction: str) -> int:
        return self.it_f if direction == "forward" else self.it_b

    def update_count(self, direction: str) -> int:
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()

    def get_optimizer_sched(self, z: MFGPolicy) -> OptSchedPair:
        if z == self.z_f:
            return self.optimizer_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.sched_b
        else:
            raise RuntimeError()

    @torch.no_grad()
    def sample_train_data(self, opt: Options, train_direction: str) -> edict:
        policy_opt, policy_impt = {
            'forward': [self.z_f, self.z_b],  # train forward,   sample from backward
            'backward': [self.z_b, self.z_f],  # train backward, sample from forward
        }.get(train_direction)

        # prepare training data
        train_ts = self.mfg.ts.detach()

        # update mf_drift if we need it and we're sampling forward traj
        update_mf_drift = (self.mfg.uses_mf_drift() and policy_impt.direction == 'forward')

        ema, ema_impt = policy_opt.get_ema(), policy_impt.get_ema()
        with ema.average_parameters(), ema_impt.average_parameters():
            policy_impt.freeze()
            policy_opt.freeze()

            xs, zs, ws, _ = self.mfg.sample_traj(policy_impt, update_mf_drift=update_mf_drift)
            train_xs = xs.detach().cpu(); del xs
            train_zs = zs.detach().cpu(); del zs
            train_ws = ws.detach().cpu(); del ws

        log.info('generate train data from [sampling]!')

        assert train_xs.shape[0] == opt.samp_bs
        assert train_xs.shape[1] == len(train_ts)
        assert train_xs.shape == train_zs.shape
        gc.collect()

        return edict(
            xs=train_xs, zs=train_zs, ws=train_ws, ts=train_ts
        )

    def train_stage(self, opt: Options, stage: int, train_direction: str, datas: Optional[edict]=None) -> None:
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forwad,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(train_direction)

        buffer_impt, buffer_opt = {
            'forward':  [self.buffer_f, self.buffer_b],
            'backward': [self.buffer_b, self.buffer_f],
        }.get(policy_impt.direction)

        if datas is None:
            datas = self.sample_train_data(opt, train_direction)

        # Compute the cost and statistical distance for the forward / backward trajectories
        if opt.log_tb:
            t1 = time.time()
            self.log_validate_metrics(opt, train_direction, datas)
            log.info("Done logging validate metrics! Took {:.1f} s!".format(time.time() - t1))

        # update buffers
        if opt.use_rb_loss:
            buffer_impt.append(datas)

        self.train_ep(opt, stage, train_direction, datas, policy_opt, policy_impt, buffer_opt, buffer_impt)

    def train_ep(
        self,
        opt: Options,
        stage: int,
        direction: str,
        datas: edict,
        policy: MFGPolicy,
        policy_impt: MFGPolicy,
        buffer_opt: Optional[Buffer],
        buffer_impt: Optional[Buffer],
    ) -> None:
        train_xs, train_zs, train_ws, train_ts = datas.xs, datas.zs, datas.ws, datas.ts

        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        optimizer, sched  = self.get_optimizer_sched(policy)
        optimizer_impt, _ = self.get_optimizer_sched(policy_impt)

        policy.activate() # activate Y (and Z)
        policy_impt.freeze() # freeze Y_impt (and Z_impt)

        if stage>0 and opt.use_rb_loss: assert len(buffer_opt)>0 and len(buffer_impt)>0

        mfg = self.mfg
        samp_direction = policy_impt.direction
        for it in range(opt.num_itr):
            step = self.update_count(direction)

            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,))

            dim01 = [opt.train_bs_x, opt.interval]

            # -------- build sample --------
            ts      = train_ts.detach()
            xs      = train_xs[samp_x_idx].to(opt.device)
            zs_impt = train_zs[samp_x_idx].to(opt.device)
            dw      = train_ws[samp_x_idx].to(opt.device)

            if mfg.uses_xs_all():
                samp_x_idx2 = torch.randint(opt.samp_bs,  (opt.train_bs_x,))
                xs_all = train_xs[samp_x_idx2].to(opt.device)
            else:
                xs_all = None

            optimizer.zero_grad()
            optimizer_impt.zero_grad()

            # -------- compute KL loss --------
            loss_kl, zs, kl, _ = loss_lib.compute_kl_loss(
                opt, dim01, mfg, samp_direction,
                ts.detach(), xs.detach(), zs_impt.detach(),
                policy, return_all=True
            )

            # -------- compute bsde TD loss --------
            loss_bsde_td = loss_lib.compute_bsde_td_loss(
                opt, mfg, samp_direction,
                ts.detach(), xs.detach(), zs.detach(), dw.detach(), kl.detach(),
                policy, policy_impt, xs_all
            )

            # -------- compute boundary loss --------
            loss_boundary = loss_lib.compute_boundary_loss(
                opt, mfg, ts.detach(), xs.detach(), policy_impt, policy,
            )

            # -------- compute mismatch loss between Z and \nabla_x Y --------
            loss_grad = torch.Tensor([0.0])
            if self.is_actor_critic_param:
                norm = get_grad_loss_norm(opt)
                loss_grad = loss_lib.compute_grad_loss(opt, ts.detach(), xs.detach(), mfg.sde, policy, norm)

            # -------- compute replay buffer loss ---------
            loss_bsde_td_rb = torch.Tensor([0.0])
            if opt.use_rb_loss:
                loss_bsde_td_rb = loss_lib.compute_bsde_td_loss_from_buffer(
                    opt, mfg, buffer_impt, ts.detach(), policy, policy_impt, xs_all
                )

            # -------- compute loss and backprop --------
            if self.is_critic_param:
                if opt.weighted_loss:
                    w_kl, w_nkl = opt.weights['kl'], opt.weights['non-kl']
                    loss = w_kl * loss_kl + w_nkl * (loss_bsde_td + loss_boundary + loss_bsde_td_rb)
                else:
                    loss = loss_kl + loss_bsde_td + loss_boundary + loss_bsde_td_rb

            elif self.is_actor_critic_param:
                if opt.weighted_loss:
                    w_kl, w_nkl = opt.weights['kl'], opt.weights['non-kl']
                    loss = w_kl * loss_kl + w_nkl * (loss_boundary + loss_bsde_td + loss_grad + loss_bsde_td_rb)
                else:
                    loss = loss_kl + loss_boundary + loss_bsde_td + loss_grad + loss_bsde_td_rb
            else:
                raise RuntimeError("")

            assert not torch.isnan(loss)
            loss.backward()

            optimizer.step()
            policy.update_ema()

            if sched is not None: sched.step()

            # -------- logging --------
            loss = edict(
                kl=loss_kl, grad=loss_grad,
                boundary=loss_boundary, bsde_td=loss_bsde_td,
            )
            if it % 20 == 0:
                self.log_train(opt, it, stage, loss, optimizer, direction)

    def train(self, opt: Options) -> None:
        self.evaluate(opt, 0)

        for stage in range(opt.num_stage):
            if opt.samp_method == 'jacobi':
                datas1 = self.sample_train_data(opt, 'forward')
                datas2 = self.sample_train_data(opt, 'backward')
                self.train_stage(opt, stage, 'forward', datas=datas1)
                self.train_stage(opt, stage, 'backward', datas=datas2)

            elif opt.samp_method == 'gauss':
                self.train_stage(opt, stage, 'forward')
                self.train_stage(opt, stage, 'backward')

            t1 = time.time()
            self.evaluate(opt, stage+1)
            log.info("Finished evaluate! Took {:.2f}s.".format(time.time() - t1))

        if opt.log_tb: self.writer.close()

    @torch.no_grad()
    def evaluate(self, opt: Options, stage: int) -> None:
        snapshot, ckpt = util.evaluate_stage(opt, stage)
        if snapshot:
            self.z_f.freeze(); self.z_b.freeze()
            self.mfg.save_snapshot(self.z_f, self.z_b, stage)

        if ckpt and stage > 0:
            keys = ['z_f','optimizer_f','z_b','optimizer_b', "it_f", "it_b"]
            util.save_checkpoint(opt, self, keys, stage)

    def compute_validate_metrics(self, opt: Options, train_direction: str, datas: edict) -> Dict[str, torch.Tensor]:
        # Sample direction is opposite of train_direction.
        xs, zs, ts = datas.xs, datas.zs, datas.ts

        b, T, nx = xs.shape
        assert zs.shape == (b, T, nx)
        assert ts.shape == (T,)

        mfg = self.mfg
        dt = mfg.dt

        metrics = {}

        # Compute "polarization" via the "condition number" of the covariance matrix.
        if "opinion" in mfg.problem_name:
            metrics.update(
                eval_metrics.compute_conv_l1_metrics(mfg, xs, train_direction)
            )

        # Compute Wasserstein distance between the x0 / xT and p0 / pT.
        metrics.update(
            eval_metrics.compute_sinkhorn_metrics(opt, mfg, xs, train_direction)
        )

        # Compute the state + control cost.
        if mfg.uses_xs_all():
            xs_all = xs.to(opt.device)
            xs_all = xs_all[torch.randperm(opt.samp_bs)]
        else:
            xs_all = None

        est_mf_cost, logp = eval_metrics.compute_est_mf_cost(
            opt, self, xs, ts, dt, xs_all, return_logp=True
        )
        s_cost, mf_cost = eval_metrics.compute_state_cost(opt, mfg, xs, ts, xs_all, logp, dt)
        del logp, xs_all
        control_cost = eval_metrics.compute_control_cost(opt, zs, dt)

        mean_s_cost, mean_control_cost, mean_mf_cost = s_cost.mean(), control_cost.mean(), mf_cost.mean()

        mean_nonmf_cost = mean_s_cost + mean_control_cost
        mean_total_cost = mean_nonmf_cost + mfg.mf_coeff * mean_mf_cost

        metrics["est_mf_cost"] = est_mf_cost
        metrics["state_cost"] = mean_s_cost
        metrics["nonmf_cost"] = mean_nonmf_cost
        metrics["mf_cost"] = mean_mf_cost
        metrics["control_cost"] = mean_control_cost
        metrics["total_cost"] = mean_total_cost

        return metrics

    @torch.no_grad()
    def log_validate_metrics(self, opt: Options, direction: str, datas: edict) -> None:

        def tag(name: str) -> str:
            return os.path.join(f"{direction}-loss", name)
        step = self.get_count(direction)

        metrics = self.compute_validate_metrics(opt, direction, datas)

        # Log all metrics.
        for key in metrics:
            self.writer.add_scalar(tag(key), metrics[key], global_step=step)
        del metrics

    def log_train(
        self, opt: Options, it: int, stage: int, loss: edict, optimizer: torch.optim.Optimizer, direction: str
    ) -> None:
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        log.info("[SB {0}] stage {1}/{2} | itr {3}/{4} | lr {5} | loss {6} | time {7}"
            .format(
                "fwd" if direction=="forward" else "bwd",
                str(1+stage).zfill(2),
                opt.num_stage,
                str(1+it).zfill(3),
                opt.num_itr,
                "{:.2e}".format(lr),
                util.get_loss_str(loss),
                "{0}:{1:02d}:{2:05.2f}".format(*time_elapsed),
        ))

        step = self.get_count(direction)
        if opt.log_tb:
            assert isinstance(loss, edict)
            for key, val in loss.items():
                # assert val > 0 # for taking log
                self.writer.add_scalar(
                    os.path.join(f'{direction}-loss', f'{key}'), val.detach(), global_step=step
                )

            # Also log the current stage.
            self.writer.add_scalar(os.path.join(f"{direction}-loss", "stage"), stage, global_step=step)

            # Log the LR.
            self.writer.add_scalar(os.path.join(f"{direction}-opt", "lr"), lr, global_step=step)
