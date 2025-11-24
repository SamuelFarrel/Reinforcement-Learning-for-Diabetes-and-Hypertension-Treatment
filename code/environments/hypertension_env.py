import numpy as np
import gymnasium as gym
from utils.utils import load_config

class HypertensionEnv(gym.Env):
    """
    Custom Gym environment for simulating blood pressure (BP) management in hypertensive patients.

    Action Space:
        - Discrete(7): Each action represents a different intervention intensity:
            0: Strongest BP-lowering intervention
            1: Strong BP-lowering intervention
            2: Moderate BP-lowering intervention
            3: No intervention
            4: Mild BP-raising intervention
            5: Moderate BP-raising intervention
            6: Strongest BP-raising intervention

    Reward Function:
        - +2.0 if systolic BP (sysbp) is within the target range (110-130 mmHg).
        - Otherwise, negative reward proportional to the distance from the target center (120 mmHg): reward = -|sysbp - 120| / 25.
        - Additional bonus (+0.3) for stable BP (change < 12 mmHg from previous step).

    State Transition Distribution:
        - The next BP is determined by:
            - A base effect associated with the chosen action.
            - Stochastic noise sampled from N(0, 4) added to the action effect.
            - Additional noise N(0, 3) added to the BP update.
            - If a risk model is used, action effects are scaled based on predicted patient risk.
        - BP is clipped to [70, 250] mmHg.
        - The episode terminates after a fixed number of steps.
    """
    def __init__(self, data, risk_model=None, use_sl_guidance=False):
        super().__init__()
        self.data = data
        self.risk_model = risk_model
        self.use_sl_guidance = use_sl_guidance
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1] + 1,), dtype=np.float32)
        self.bp_history = []
        self.patient_risk_predictions = None
        self.config = load_config()
        self.max_steps = self.config['max_steps_per_episode']
        
        # Predict risk for all patients before RL starts
        if self.risk_model is not None and self.use_sl_guidance:
            self.patient_risk_predictions = self.risk_model.predict(self.data)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_patient = np.random.randint(0, len(self.data))
        self.current_step = 0
        self.current_sysbp = self.data[self.current_patient][7] * 40 + 120
        self.initial_sysbp = self.current_sysbp
        self.bp_history = [self.current_sysbp]
        
        # Get pre-computed risk for this patient
        if self.patient_risk_predictions is not None:
            self.current_risk = self.patient_risk_predictions[self.current_patient]
        else:
            self.current_risk = 0
        
        state = np.append(self.data[self.current_patient], self.current_sysbp / 200.0)
        return state.astype(np.float32), {}
    
    def step(self, action):
        base_effects = [-22, -16, -10, 0, +7, +14, +20]
        
        stochastic_noise = np.random.normal(0, 4)
        
        # Adjust action intensity based on risk prediction
        if self.use_sl_guidance:
            if self.current_risk == 1:  # High risk patient
                if action in [0, 1, 2]:
                    effect = base_effects[action] * 1.5 + stochastic_noise
                else:
                    effect = base_effects[action] + stochastic_noise
            else:  # Low risk patient
                if action in [0, 1, 2]:
                    effect = base_effects[action] * 0.8 + stochastic_noise
                else:
                    effect = base_effects[action] + stochastic_noise
        else:
            effect = base_effects[action] + stochastic_noise
        
        self.current_sysbp += effect + np.random.normal(0, 3)
        self.current_sysbp = np.clip(self.current_sysbp, 70, 250)
        
        target_range = (110, 130)
        if target_range[0] <= self.current_sysbp <= target_range[1]:
            reward = 2.0
        else:
            target_center = (target_range[0] + target_range[1]) / 2
            distance = abs(self.current_sysbp - target_center)
            reward = -distance / 25.0
        
        if len(self.bp_history) > 0:
            bp_change = abs(self.current_sysbp - self.bp_history[-1])
            if bp_change < 12:
                reward += 0.3
        
        self.bp_history.append(self.current_sysbp)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        state = np.append(self.data[self.current_patient], self.current_sysbp / 200.0)
        
        return state.astype(np.float32), reward, terminated, False, {
            'sysbp': self.current_sysbp,
            'patient_risk': self.current_risk,
            'bp_history': self.bp_history.copy()
        }