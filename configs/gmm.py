from easydict import EasyDict as edict

def _common():
  config = edict()
  config.problem_name = 'GMM'
  config.x_dim = 2

  # sde
  config.seed = 42
  config.t0 = 0.0
  config.T = 1.0
  config.interval = 100
  config.diffusion_std = 1.0

  # training
  config.num_itr   = 250
  config.train_bs_x = 128
  config.rb_bs_x    = 128

  # sampling & evaluation
  config.samp_bs = 5000
  config.snapshot_freq = 1
  # config.ckpt_freq = 2

  # optimization
  config.optimizer = 'AdamW'

  return config

def gmm_actor_critic():
  config = _common()

  # paramatrization
  config.sb_param = 'actor-critic'
  config.policy_net = 'toy'

  # optimization
  config.lr = 5e-4
  config.lr_y = 1e-3
  config.lr_gamma = 0.999

  # tuning
  config.num_stage = 40
  config.multistep_td = True
  config.samp_method = 'gauss'

  return config

def gmm_critic():
  config = _common()

  # paramatrization
  config.sb_param = 'critic'
  config.policy_net = 'toy'

  # optimization
  config.lr = 5e-4
  config.lr_gamma = 0.999

  # tuning
  config.num_stage = 40
  config.multistep_td = True
  config.samp_method = 'gauss'

  return config
