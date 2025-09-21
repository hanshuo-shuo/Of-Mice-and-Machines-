import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel


class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        """Initialize TD-MPC2 agent with configuration"""
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize world model and optimizers
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': []}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        
        self.model.eval()
        self.scale = RunningScale(cfg)
        
        # Increase iterations for large action spaces
        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)  
        
        # Calculate discount factor based on episode length
        self.discount = self._get_discount(cfg.episode_length)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict["model"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
            float: Variance penalty value
        """
        # Encode observation into latent space
        new_obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(new_obs, task)
        
        # # Calculate variance penalty
        # variance_penalty = self.variance_penalty(z, task=task)
        # print(variance_penalty)

        # Plan and return action
        a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """
        Estimate value of a trajectory.
        
        Args:
            z (torch.Tensor): Initial latent state
            actions (torch.Tensor): Sequence of actions
            task (torch.Tensor): Task index
            
        Returns:
            torch.Tensor: Estimated value of the trajectory
        """
        # Regular value estimation for the trajectory
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G = G + discount * reward
            discount = discount * self.discount
        
        # Get final value using policy actions
        pi_actions = self.model.pi(z, task)[1]
        final_q = self.model.Q(z, pi_actions, task, return_type='min')
        final_value = G + discount * final_q
        
        return final_value

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        """
        Plan a sequence of actions using the learned world model.
        
        Args:
            z (torch.Tensor): Initial latent state
            t0 (bool): Whether this is the first timestep
            eval_mode (bool): Whether to use mean action vs sampling
            task (torch.Tensor): Task index
            
        Returns:
            torch.Tensor: Planned action to execute
        """
        # Sample policy trajectories if needed
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon - 1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        Q_values_iteration = []
        
        # Iterate MPPI
        for i in range(self.cfg.iterations):
            # Sample actions
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                                                  torch.randn(self.cfg.horizon,
                                                          self.cfg.num_samples - self.cfg.num_pi_trajs,
                                                          self.cfg.action_dim, device=std.device)) \
                .clamp(0, 1)

            # Compute elite actions using pre-computed variance penalty
            value = self._estimate_value(z, actions, task)
            value = value.nan_to_num_(0)

            # Get Q-values for current state
            current_state_values = self.model.Q(z, actions[0], task, return_type='avg')
            current_state_variance = torch.var(current_state_values)
            Q_values_iteration.append(current_state_variance.item())

            # Select elite actions
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(
                torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)

        # Select final action
        score = score.squeeze(1).cpu().numpy()
        selected_idx = np.random.choice(np.arange(score.shape[0]), p=score)
        selected_actions = elite_actions[:, selected_idx]
        self._prev_mean = mean

        # Add noise in training mode
        a, std = selected_actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device) * 0.0
            
        return a.clamp_(0, 1)

    def update_pi(self, zs, task):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def variance_penalty(self, next_z, task):
        """
        Calculate variance penalty based on Q-values across action space.
        
        Args:
            next_z (torch.Tensor): Next latent state
            task (torch.Tensor): Task index
            
        Returns:
            torch.Tensor: Variance penalty value
        """
        batch_size = 1
        grid_size = 30
        
        # Create 2D action grid
        x = torch.linspace(0, 1, grid_size, device=self.device)
        y = torch.linspace(0, 1, grid_size, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Create action samples with random third dimension
        grid_actions = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.rand(xx.flatten().shape, device=self.device)
        ], dim=1)

        # Expand latent state to match grid size
        current_z = next_z
        z_expanded = current_z.unsqueeze(1).expand(-1, grid_size*grid_size, -1)
        z_flat = z_expanded.reshape(-1, current_z.shape[-1])
        grid_actions_flat = grid_actions.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)

        # Get Q-values and compute variance
        grid_q_values = self.model.Q(z_flat, grid_actions_flat, task, return_type='all')
        grid_q_values = torch.stack([math.two_hot_inv(q, self.cfg) for q in grid_q_values])
        grid_q_values = grid_q_values.view(self.cfg.num_q, batch_size, grid_size*grid_size)
        variance_penalty = torch.var(grid_q_values, dim=(0, 2))

        return variance_penalty

    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        """
        Note on the third dimension of action space:
        The action space consists of (x, y) coordinates and a third dimension that can be sampled from:
        1. Waiting/peeking space (third dim = 1): For careful local exploration and uncertainty assessment
        2. Moving space (third dim = 0): For regular movement actions
        
        During our experiments, we found that sampling from the waiting/peeking space 
        led to better performance. This suggests that the variance in Q-values during 
        these small, careful movements provides valuable information about state uncertainty.
        
        Possible explanations:
        1. High variance during peeking indicates critical decision points
        2. Helps identify states where even small actions lead to significant value changes
        3. Provides a natural mechanism for local exploration
        4. Allows the agent to better assess risk before committing to larger movements
        
        This dual-space sampling approach and its effects on performance present 
        an interesting direction for future research.
        """
        horizon = next_z.shape[0]      # horizon length
        batch_size = next_z.shape[1]   # batch size
        grid_size = 20                 # size of action grid, higher is better but slower
        grid_area = grid_size * grid_size  # total number of grid points
        
        # Create grid of actions for variance computation
        x = torch.linspace(0, 1, grid_size, device=self.device)
        y = torch.linspace(0, 1, grid_size, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Can sample actions from either space:
        # Waiting/peeking space (third dim = 1)
        grid_actions = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.ones_like(xx.flatten())
        ], dim=1)
        
        # # Moving space (third dim = 0)
        # grid_actions = torch.stack([
        #     xx.flatten(),
        #     yy.flatten(),
        #     torch.zeros_like(xx.flatten())
        # ], dim=1)

        # # # hybrid action space the third dim is between any number between 0 and 1
        # grid_actions = torch.stack([
        #     xx.flatten(),
        #     yy.flatten(),
        #     torch.rand(xx.flatten().shape, device=self.device)
        # ], dim=1)
        
        # Process each timestep
        td_targets = []
        for t in range(horizon):
            # Get current state
            current_z = next_z[t]  # [batch_size, latent_dim]
            
            # Expand z to match grid_actions shape
            z_expanded = current_z.unsqueeze(1).expand(-1, grid_area, -1)  # [batch_size, grid_area, latent_dim]
            z_flat = z_expanded.reshape(-1, current_z.shape[-1])    # [batch_size*grid_area, latent_dim]
            grid_actions_flat = grid_actions.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 3)  # [batch_size*grid_area, 3]

            # Get Q-values for all grid actions
            grid_q_values = self.model.Q(z_flat, grid_actions_flat, task, return_type='all')
            
            # Reshape and compute variance
            grid_q_values = torch.stack([math.two_hot_inv(q, self.cfg) for q in grid_q_values])  # [num_q, batch_size*grid_area]
            grid_q_values = grid_q_values.view(self.cfg.num_q, batch_size, grid_area)  # [num_q, batch_size, grid_area]
            variance_penalty = torch.var(grid_q_values, dim=(0, 2))  # [batch_size]
            
            # Regular TD target computation for this timestep
            pi = self.model.pi(current_z, task)[1]
            q_value = self.model.Q(current_z, pi, task, return_type='min', target=True)

            # clip the variance_penalty
            variance_penalty = torch.clamp(variance_penalty, 0, 1000)
            td_target = reward[t] + self.discount * (q_value - self.cfg.penalty_coefficient * variance_penalty.unsqueeze(1))
            td_targets.append(td_target)
        
        # # Debug information: print the range of variance_penalty for each batch
        # variance_min, variance_max = variance_penalty.min().item(), variance_penalty.max().item()
        # print(f"Batch variance_penalty range: min={variance_min}, max={variance_max}")
        
        # Stack all timesteps
        return torch.stack(td_targets, dim=0)  # [horizon, batch_size, 1]

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
        # Sample batch from buffer
        obs, action, reward, task = buffer.sample()

        # Compute TD targets
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Latent rollout
        zs = torch.empty(self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho ** t
            zs[t + 1] = z

        # Get predictions
        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)

        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho ** t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho ** t
                
        # Average losses
        consistency_loss *= (1 / self.cfg.horizon)
        reward_loss *= (1 / self.cfg.horizon)
        value_loss *= (1 / (self.cfg.horizon * self.cfg.num_q))
        
        # Combine losses
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy and target Q-functions
        pi_loss = self.update_pi(zs.detach(), task)
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }