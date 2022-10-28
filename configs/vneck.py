from easydict import EasyDict as edict

def _common():
  config = edict()
  config.problem_name = 'Vneck'
  config.x_dim = 2

  # sde
  config.seed = 42
  config.t0 = 0.0
  config.T = 2.0
  config.interval = 200
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

def vneck_actor_critic():
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
  config.use_rb_loss = True

  config.samp_method = 'jacobi'

  return config

def vneck_critic():
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
  config.use_rb_loss = True

  # gauss converges faster but with slightly worse W2
  config.samp_method = 'jacobi'

  return config
