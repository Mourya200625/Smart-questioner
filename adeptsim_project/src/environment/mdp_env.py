import gymnasium as gym
from gymnasium import spaces
import numpy as np

DOMAINS = [
    "Demographic & Role Constraints",
    "Core Values & Priorities",
    "Risk Tolerance & Stress Tendencies",
    "Social & Relational Dynamics",
    "Resource & Temporal Boundaries"
]

class AdeptSimEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, backend, max_steps: int = 5, embed_dim: int = 768):
        super().__init__()
        self.backend = backend
        self.max_steps = max_steps
        self.embed_dim = embed_dim

        # 5 trait domains (0-4) + 1 Terminal Stop Action (5)
        self.STOP_ACTION = 5
        self.action_space = spaces.Discrete(6)

        # State = [context_embedding (768), trait_mask (5), step_progress (1)]
        state_dim = self.embed_dim + 5 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.p_known = {}
        self.trait_mask = np.ones(5, dtype=np.float32)
        self.scenario_data = None
        self.context_emb = None

    def reset(self, scenario_data: dict, seed=None, options=None):
        super().reset(seed=seed)
        self.scenario_data = scenario_data
        self.current_step = 0
        self.p_known = {}
        self.trait_mask = np.ones(5, dtype=np.float32)  # 1 = unqueried, 0 = queried
        self.context_emb = self.backend.get_embedding(scenario_data["context"])
        return self._get_obs(), {}

    def _get_obs(self):
        step_feat = np.array([self.current_step / float(self.max_steps)], dtype=np.float32)
        return np.concatenate([self.context_emb, self.trait_mask, step_feat]).astype(np.float32)

    def step(self, action: int):
        self.current_step += 1
        terminated = False
        reward = 0.0
        info = {
            "is_terminal": False,
            "choice_consistency": 0.0,
            "query_count": 0,
            "resolved_critical": False
        }

        # Terminal Action: Stop & Simulate
        if action == self.STOP_ACTION or self.current_step >= self.max_steps:
            terminated = True
            info["is_terminal"] = True
            
            simulated_choice = self.backend.simulate_choice(
                self.scenario_data["context"], self.p_known, temperature=0.1
            )
            gold = self.scenario_data["gold_choice"].lower()
            sim = simulated_choice.lower()
            
            is_match = (gold in sim) or (sim in gold)
            if is_match:
                reward += 1.0
                info["choice_consistency"] = 1.0
            
            info["simulated_choice"] = simulated_choice
            info["p_known"] = self.p_known
            return self._get_obs(), reward, terminated, False, info

        # Step Action: Query trait domain (0 to 4)
        domain_idx = action
        info["query_count"] = 1

        # Redundant query penalty
        if self.trait_mask[domain_idx] == 0.0:
            reward -= 0.05
            return self._get_obs(), reward, terminated, False, info

        domain_name = DOMAINS[domain_idx]
        trait_val, is_critical = self.backend.query_user_proxy(
            self.scenario_data["full_profile"], domain_name
        )
        self.p_known[domain_name] = trait_val
        self.trait_mask[domain_idx] = 0.0

        # Step reward computation
        reward -= 0.05
        if is_critical:
            reward += 0.5
            info["resolved_critical"] = True

        return self._get_obs(), reward, terminated, False, info