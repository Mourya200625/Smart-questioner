import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim: int = 6):
        super().__init__()

        # Ensure state_dim is an integer if a tuple/shape was passed
        if isinstance(state_dim, (tuple, list)):
            state_dim = state_dim[0]

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class DoubleDQNAgent:
    def __init__(self, state_dim: int, action_dim: int = 6, lr: float = 1e-3, gamma: float = 0.95):
        self.action_dim = action_dim
        self.gamma = gamma
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=2000)

    def select_action(self, state: np.ndarray, trait_mask: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            # Random valid action: unqueried traits (where mask==1) or Stop Action (5)
            valid_actions = [i for i in range(5) if trait_mask[i] == 1.0] + [5]
            return random.choice(valid_actions)

        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = self.q_net(s_tensor).squeeze(0).numpy()
            
            # Mask out already queried domains by setting their Q-value to -infinity
            for i in range(5):
                if trait_mask[i] == 0.0:
                    q_vals[i] = -1e9
            return int(np.argmax(q_vals))

    def update(self, batch_size: int = 16):
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, batch_size)
        s, a, r, ns, d = zip(*batch)

        s_t = torch.tensor(np.array(s), dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        ns_t = torch.tensor(np.array(ns), dtype=torch.float32)
        d_t = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        curr_q = self.q_net(s_t).gather(1, a_t)

        with torch.no_grad():
            # Double DQN: action selected by q_net, evaluated by target_net
            next_actions = self.q_net(ns_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns_t).gather(1, next_actions)
            target_q = r_t + (1.0 - d_t) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())