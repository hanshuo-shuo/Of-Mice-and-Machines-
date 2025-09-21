import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')
import copy
import hydra
from cellworld_game.video import save_video_output
import numpy as np
import torch
from termcolor import colored
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_prey_env
from tdmpc2 import TDMPC2
import pandas as pd
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name 
		`model_size`: model size, must be one of `[1, 3, 5]` 
		`checkpoint`: path to model checkpoint to load, has to be absolute path
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`seed`: random seed (default: 2755)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=cellworld_evade model_size=1 checkpoint=/path/to/fianl.pt
		$ python evaluate.py task=cellworld_evade model_size=5 checkpoint=/path/to/fianl-317M.pt
	```
	"""
	# assert torch.cuda.is_available()
	# assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	# set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.model_size}', 'blue', attrs=['bold']))
	# print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	# Make environment
	env = make_prey_env(cfg)
	# Load agent
	print(colored('Loading agent...', 'yellow', attrs=['bold']))
	agent = TDMPC2(cfg)
	print(os.getcwd())
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	scores = []
	tasks = [cfg.task]
	for task_idx, task in enumerate(tasks):
		task_idx = None
		ep_rewards, ep_successes = [], []
		# Q_var_list, ep_list= [], []
		obs_list = []
		action_list = []
		reward_list = []
		done_list = []
		next_obs_list = []
		variance_penalty_list = []
		for i in tqdm(range(cfg.eval_episodes), desc='Evaluating episodes'):
			save_video_output(env.model, "/Users/hanshuo/Documents/project/RL/Of-Mice-and-Machines/VP-TDMPC-2/video")
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			while not done:
				action = agent.act(obs, t0=t==0, task=task_idx)
				# ep_list.append(i+1)
				copied_obs = copy.deepcopy(obs.numpy())
				new_obs, reward, done, info = env.step(action)
				obs_list.append(copied_obs)
				action_list.append(action.numpy())
				# save reward as float, now it is tensor
				reward_list.append(reward.item())
				done_list.append(done)
				next_obs_list.append(copy.deepcopy(new_obs.numpy()))
				obs = new_obs
				ep_reward += reward
				t += 1
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		data = {
			"obs": obs_list,
			"action": action_list,
			"reward": reward_list,
			"done": done_list,
			"next_obs": next_obs_list,
		}
		# data = {
		# 	"ep": ep_list,
		# 	"Q_var": Q_var_list
		# }
		# df = pd.DataFrame(data)
		# print(obs_list)
		# df.to_csv("/Users/hanshuo/Documents/project/RL/tlppo_cellworld_evade/030_vp5.csv", index=False)
		print(colored(f'  {task:<22}' \
			f'\tR: {ep_rewards:.01f}  ' \
			f'\tS: {ep_successes:.02f}', 'yellow'))

if __name__ == '__main__':
	evaluate()