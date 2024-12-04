from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FinancialLiteracyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, size: int = 5):
        super().__init__()
        
        # Grid properties
        self.size = size
        self.window_size = 512  # The size of the PyGame window
        self.render_mode = render_mode
        
        # Initial values
        self.initial_balance = 50
        self.target_balance = 100
        
        # Spaces
        # Actions: up, down, left, right, invest, save
        self.action_space = spaces.Discrete(6)
        
        # Observation space: position (2), balance (1), literacy_score (1)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([
                self.size - 1,  # max x position
                self.size - 1,  # max y position
                float('inf'),   # max balance
                100.0          # max literacy score
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action to direction mapping
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1])   # down
        }
        
        # Initialize grid elements
        self._init_grid()

    def _init_grid(self):
        self.grid = np.zeros((self.size, self.size), dtype=str)
        # Place elements (emoji represents different financial elements)
        self.grid[0, 4] = "🏦"  # Bank
        self.grid[1, 1] = "💵"  # Income
        self.grid[3, 2] = "💵"  # Income
        self.grid[2, 2] = "📈"  # Investment
        self.grid[1, 3] = "🛍️"  # Expense
        self.grid[3, 1] = "🛍️"  # Expense
        self.grid[2, 4] = "💡"  # Financial tip

    def _get_obs(self):
        return np.array([
            self.position[0],
            self.position[1],
            self.balance,
            self.literacy_score
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "balance": float(self.balance),
            "literacy_score": float(self.literacy_score),
            "distance_to_bank": np.linalg.norm(
                self.position - np.array([0, 4]), ord=1  # Manhattan distance to bank
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize agent state
        self.position = np.array([4, 0])  # Start at bottom-left
        self.balance = self.initial_balance
        self.literacy_score = 0.0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        terminated = truncated = False
        reward = 0
        
        # Handle movement actions (0-3)
        if action < 4:
            direction = self._action_to_direction[action]
            new_position = np.clip(
                self.position + direction,
                0,
                self.size - 1
            )
            self.position = new_position
        
        # Handle special actions and cell effects
        cell_type = self.grid[self.position[0], self.position[1]]
        
        if cell_type == "💵":  # Income
            self.balance += 20
            reward += 5
        elif cell_type == "🛍️":  # Expense
            self.balance -= 15
            reward -= 5
        elif cell_type == "📈" and action == 4:  # Investment action
            success = self.np_random.random() < 0.7
            if success:
                self.balance *= 1.2
                reward += 5
            else:
                self.balance *= 0.8
                reward -= 5
        elif cell_type == "💡":  # Financial tip
            self.literacy_score = min(100.0, self.literacy_score + 10.0)
            reward += 2
        elif cell_type == "🏦" and action == 5:  # Save action at bank
            if self.balance >= self.target_balance:
                reward += 10
                terminated = True
        
        # Apply literacy score bonus to reward
        reward *= (1 + self.literacy_score / 100)
        
        # Check termination conditions
        if self.balance <= 0:
            terminated = True
            reward -= 10
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def render(self):
        return self.grid.copy()

    def close(self):
        pass


# Register the environment
gym.register(
    id='FinancialLiteracy-v0',
    entry_point='financial_literacy_env:FinancialLiteracyEnv',
)