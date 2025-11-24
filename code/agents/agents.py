import time
import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from utils.utils import load_config

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.n_calls = 0
        
    def _on_step(self):
        # Get reward from the last step
        reward = self.locals['rewards'][0]
        self.current_episode_rewards.append(reward)
        self.rewards.append(reward)
        self.n_calls += 1
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(sum(self.current_episode_rewards))
            self.current_episode_rewards = []
        
        return True

def train_rl_models(env, seed, timesteps=None):
    """Train RL models with reward tracking"""
    config = load_config()
    
    if timesteps is None:
        timesteps = config['timesteps']
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    vec_env = DummyVecEnv([lambda: env])
    models = {}
    reward_callbacks = {}
    
    models["PPO"] = PPO(
        "MlpPolicy", vec_env, verbose=0, seed=seed,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        n_steps=1024,
        batch_size=128,
        ent_coef=0.01, vf_coef=0.5
    )
    
    models["DQN"] = DQN(
        "MlpPolicy", vec_env, verbose=0, seed=seed,
        learning_rate=1e-4, gamma=0.99, buffer_size=100000,
        learning_starts=1000, batch_size=64,
        exploration_fraction=0.3, exploration_final_eps=0.05
    )
    
    models["A2C"] = A2C(
        "MlpPolicy", vec_env, verbose=0, seed=seed,
        learning_rate=7e-4, gamma=0.99, n_steps=512,
        ent_coef=0.01, vf_coef=0.5
    )
    
    for name, model in models.items():
        callback = RewardCallback()
        reward_callbacks[name] = callback
        
        start_time = time.time()
        model.learn(total_timesteps=timesteps, callback=callback)
        end_time = time.time()
        print(f"      {name}: {end_time - start_time:.1f}s")
    
    return models, reward_callbacks

def evaluate_models_with_detailed_tracking(env, models, n_episodes=None, seed=42):
    """Enhanced evaluation with detailed health metrics tracking"""
    config = load_config()
    
    if n_episodes is None:
        n_episodes = config['eval_episodes']
        
    np.random.seed(seed)
    results = {}
    episode_rewards = {}
    reward_by_timestep = {model_name: [] for model_name in models}
    health_metrics = {}
    
    for model_name, model in models.items():
        ep_rewards = []
        glucose_trajectories = []
        bp_trajectories = []
        rewards_over_time = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0
            episode_rewards_by_step = []
            done = False
            
            # Track health metrics
            episode_glucose = []
            episode_bp = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                episode_rewards_by_step.append(total_reward)  # Cumulative reward at each step
                
                # Track health metrics
                if 'glucose' in info:
                    episode_glucose.append(info['glucose'])
                if 'sysbp' in info:
                    episode_bp.append(info['sysbp'])
            
            ep_rewards.append(total_reward)
            rewards_over_time.append(episode_rewards_by_step)
            
            if episode_glucose:
                glucose_trajectories.append(episode_glucose)
            if episode_bp:
                bp_trajectories.append(episode_bp)
        
        results[model_name] = np.mean(ep_rewards)
        episode_rewards[model_name] = ep_rewards
        reward_by_timestep[model_name] = rewards_over_time
        
        health_metrics[model_name] = {
            'glucose_trajectories': glucose_trajectories,
            'bp_trajectories': bp_trajectories
        }
    
    return results, episode_rewards, reward_by_timestep, health_metrics