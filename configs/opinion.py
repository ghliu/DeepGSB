from easydict import EasyDict as edict


def _common():
    config = edict()
    config.problem_name = "opinion"
    config.x_dim = 2

    # sde
    config.seed = 42
    config.t0 = 0.0
    config.T = 3.0
    config.interval = 300
    config.diffusion_std = 0.1

    # training
    config.num_stage = 40
    config.num_itr = 100
    config.train_bs_x = 128
    config.rb_bs_x = 128

    # sampling & evaluation
    config.samp_bs = 2000
    config.snapshot_freq = 1
    # config.ckpt_freq = 2

    # optimization
    config.optimizer = "AdamW"

    return config


def opinion_actor_critic():
    config = _common()

    # paramatrization
    config.sb_param = "actor-critic"
    config.policy_net = "toy"

    # optimization
    coeff = 3.
    config.lr = coeff * 5e-4
    config.lr_y = 1e-3
    config.lr_gamma = 0.999

    # tuning
    config.num_stage = 40
    config.multistep_td = True
    config.use_rb_loss = True

    config.samp_method = "gauss"

    config.weights = {'kl': 0.8, 'non-kl': 0.05}

    return config


def opinion_critic():
    config = _common()

    # paramatrization
    config.sb_param = "critic"
    config.policy_net = "toy"

    # optimization
    config.lr = 5e-4
    config.lr_gamma = 0.999

    # tuning
    config.num_stage = 40
    config.multistep_td = True
    config.use_rb_loss = True

    config.samp_method = "gauss"

    config.weights = {'kl': 0.6, 'non-kl': 0.002}

    return config