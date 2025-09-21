import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer, PTSDBuffer2	
from envs import make_prey_env
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger
import os
torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	env = make_prey_env(cfg)
	agent = TDMPC2(cfg)
	buffer = PTSDBuffer2(cfg)
	if os.path.exists(cfg.checkpoint):
		agent.load(cfg.checkpoint)
	else:
		print(colored('Training from scratch...', 'yellow', attrs=['bold']))
	trainer = OnlineTrainer(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=buffer,
		logger=Logger(cfg),
	)
	trainer.train()


if __name__ == '__main__':
	train()