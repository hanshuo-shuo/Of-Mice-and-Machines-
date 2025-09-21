from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class CellworldCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CellworldCallback, self).__init__(verbose)
        self.stats_windows_size = 0
        self.current_survival = 0.0
        self.metrics = {
            'rewards': [0],
            'captures': [0], 
            'captured': [0],
            'survival': [0],
            'finished': [0],
            'truncated': [0]
        }
        self.agents = {}

    def _on_training_start(self):
        self.stats_windows_size = self.model._stats_window_size

    def _on_step(self):
        for env_id, info in enumerate(self.locals["infos"]):
            if 'terminal_observation' not in info:
                continue

            # Update metrics
            for metric, values in self.metrics.items():
                if metric == 'rewards':
                    value = info["reward"]
                elif metric == 'captures':
                    value = info["captures"]
                elif metric == 'captured':
                    value = 1 if info["captures"] > 0 else 0
                elif metric == 'survival':
                    value = info["survived"]
                elif metric in ['finished', 'truncated']:
                    value = 1 if info["TimeLimit.truncated"] else 0
                    if metric == 'finished':
                        value = not value

                values.append(value)
                if len(values) > self.stats_windows_size:
                    values.pop(0)

            # Update logging metrics
            self.current_survival = safe_mean(self.metrics['survival'])
            self._log_metrics(info)

        return True

    def _log_metrics(self, info):
        # Log main metrics
        metrics_to_log = {
            'avg_captures': safe_mean(self.metrics['captures']),
            'survival_rate': self.current_survival,
            'ep_finished': sum(self.metrics['finished']),
            'ep_truncated': sum(self.metrics['truncated']),
            'ep_captured': sum(self.metrics['captured']),
            'reward': safe_mean(self.metrics['rewards'])
        }

        for metric_name, value in metrics_to_log.items():
            self.logger.record(f'cellworld/{metric_name}', value)

        # Log agent-specific metrics
        for agent_name, agent_stats in info["agents"].items():
            if agent_name not in self.agents:
                self.agents[agent_name] = {}
                
            for stat, value in agent_stats.items():
                if stat not in self.agents[agent_name]:
                    self.agents[agent_name][stat] = []
                    
                stat_values = self.agents[agent_name][stat]
                stat_values.append(value)
                
                if len(stat_values) > self.stats_windows_size:
                    stat_values.pop(0)
                    
                self.logger.record(
                    f'cellworld/{agent_name}_{stat}', 
                    safe_mean(stat_values)
                )