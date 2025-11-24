import numpy as np
import gymnasium as gym
from utils.utils import load_config

class DiabetesEnv(gym.Env):
    """
    Custom Gym environment for simulating glucose management in diabetic patients.

    Action Space:
        - Discrete(7): Each action represents a different intervention intensity:
            0: Strongest glucose-lowering intervention
            1: Strong glucose-lowering intervention
            2: Moderate glucose-lowering intervention
            3: No intervention
            4: Mild glucose-raising intervention
            5: Moderate glucose-raising intervention
            6: Strongest glucose-raising intervention

    Reward Function:
        - +2.0 if glucose is within the target range (90-130 mg/dL).
        - Otherwise, negative reward proportional to the distance from the target center (110 mg/dL): reward = -|glucose - 110| / 20.
        - Additional bonus (+0.3) for stable glucose (change < 10 mg/dL from previous step).

    State Transition Distribution:
        - The next glucose level is determined by:
            - A base effect associated with the chosen action.
            - Stochastic noise sampled from N(0, 3) added to the action effect.
            - Additional noise N(0, 2) added to the glucose update.
            - If a risk model is used, action effects are scaled based on predicted patient risk.
        - Glucose is clipped to [40, 300] mg/dL.
        - The episode terminates after a fixed number of steps.
    """
    def __init__(self, data, risk_model=None, use_sl_guidance=False):
        super().__init__()
        self.data = data
        self.risk_model = risk_model
        self.use_sl_guidance = use_sl_guidance
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1] + 1,), dtype=np.float32)
        self.glucose_history = []
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
        self.current_glucose = self.data[self.current_patient][1] * 50 + 100
        self.initial_glucose = self.current_glucose
        self.glucose_history = [self.current_glucose]
        
        # Get pre-computed risk for this patient
        if self.patient_risk_predictions is not None:
            self.current_risk = self.patient_risk_predictions[self.current_patient]
        else:
            self.current_risk = 0
        
        state = np.append(self.data[self.current_patient], self.current_glucose / 200.0)
        return state.astype(np.float32), {}
    
    def step(self, action):
        base_effects = [-25, -18, -12, 0, +8, +15, +22]
        
        stochastic_noise = np.random.normal(0, 3)
        
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

        self.current_glucose += effect + np.random.normal(0, 2)  
        self.current_glucose = np.clip(self.current_glucose, 40, 300)
        
        target_range = (90, 130)
        if target_range[0] <= self.current_glucose <= target_range[1]:
            reward = 2.0
        else:
            target_center = (target_range[0] + target_range[1]) / 2
            distance = abs(self.current_glucose - target_center)
            reward = -distance / 20.0
        
        if len(self.glucose_history) > 0:
            glucose_change = abs(self.current_glucose - self.glucose_history[-1])
            if glucose_change < 10:
                reward += 0.3
        
        self.glucose_history.append(self.current_glucose)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        state = np.append(self.data[self.current_patient], self.current_glucose / 200.0)
        
        return state.astype(np.float32), reward, terminated, False, {
            'glucose': self.current_glucose,
            'patient_risk': self.current_risk,
            'glucose_history': self.glucose_history.copy()
        }