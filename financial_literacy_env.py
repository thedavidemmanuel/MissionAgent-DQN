# financial_literacy_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum

class CellType(IntEnum):
    """Defines the different types of cells in our financial grid"""
    EMPTY = 0
    BANK = 1
    INCOME = 2
    INVESTMENT = 3
    EXPENSE = 4
    LITERACY_TIP = 5

class FinancialLiteracyEnv(gym.Env):
    """
    A custom environment for teaching financial literacy through reinforcement learning.
    
    The agent navigates a 5x5 grid representing different financial decisions. The goal
    is to reach the bank with a sufficient balance while making smart financial choices
    along the way.
    """
    
    def __init__(self, grid_size=5):
        super().__init__()
        
        # Environment settings
        self.grid_size = grid_size
        self.initial_balance = 50.0
        self.target_balance = 100.0
        self.max_steps = 100
        
        # Initialize state variables
        self.balance = self.initial_balance
        self.literacy_score = 0.0
        self.position = np.array([grid_size-1, 0])
        self.steps = 0
        
        # Action space: up, down, left, right, invest, save
        self.action_space = spaces.Discrete(6)
        
        # Observation space: position (2), grid (25), balance (1), literacy_score (1)
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0]*25 + [0, 0]),
            high=np.array([grid_size-1, grid_size-1] + [5]*25 + [float('inf'), 100]),
            dtype=np.float32
        )
        
        # Initialize episode tracking
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset agent state
        self.position = np.array([self.grid_size-1, 0])
        self.balance = float(self.initial_balance)
        self.literacy_score = 0.0
        self.steps = 0
        self.current_episode_reward = 0
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._init_grid()
        
        return self._get_observation(), {}
    
    def _init_grid(self):
        """Initialize the grid with financial elements"""
        # Place bank (goal) in top-right corner
        self.grid[0, self.grid_size-1] = CellType.BANK
        
        # Place income sources
        self.grid[1, 1] = CellType.INCOME
        self.grid[3, 2] = CellType.INCOME
        self.grid[2, 3] = CellType.INCOME
        
        # Place investment opportunities
        self.grid[1, 3] = CellType.INVESTMENT
        self.grid[3, 1] = CellType.INVESTMENT
        
        # Place expenses
        self.grid[2, 2] = CellType.EXPENSE
        self.grid[1, 4] = CellType.EXPENSE
        
        # Place financial literacy tips
        self.grid[4, 1] = CellType.LITERACY_TIP
        self.grid[2, 0] = CellType.LITERACY_TIP
    
    def step(self, action):
        """Execute one time step within the environment"""
        self.steps += 1
        prev_position = self.position.copy()
        
        # Handle movement actions
        if action < 4:
            self._move(action)
        # Handle financial actions
        else:
            reward = self._handle_financial_action(action)
        
        # Get current cell type
        current_cell = self.grid[tuple(self.position)]
        
        # Calculate reward based on cell type and action
        reward = self._calculate_reward(current_cell)
        self.current_episode_reward += reward
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Win condition: reach bank with target balance
        if current_cell == CellType.BANK and self.balance >= self.target_balance:
            terminated = True
            reward += 10.0
        
        # Lose condition: out of money or too many steps
        if self.balance <= 0 or self.steps >= self.max_steps:
            terminated = True
            reward -= 5.0
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _move(self, action):
        """Handle movement actions"""
        if action == 0:  # up
            self.position[0] = max(0, self.position[0] - 1)
        elif action == 1:  # down
            self.position[0] = min(self.grid_size - 1, self.position[0] + 1)
        elif action == 2:  # left
            self.position[1] = max(0, self.position[1] - 1)
        elif action == 3:  # right
            self.position[1] = min(self.grid_size - 1, self.position[1] + 1)
    
    def _handle_financial_action(self, action):
        """Handle investment and saving actions"""
        current_cell = self.grid[tuple(self.position)]
        reward = 0
        
        if action == 4:  # invest
            if current_cell == CellType.INVESTMENT:
                # Investment success rate increases with literacy score
                success_rate = 0.7 + (self.literacy_score / 200)
                if self.np_random.random() < success_rate:
                    gain = self.balance * 0.5
                    self.balance += gain
                    reward += 5.0
                else:
                    loss = self.balance * 0.3
                    self.balance -= loss
                    reward -= 5.0
                    
        elif action == 5:  # save
            if current_cell == CellType.BANK:
                reward += 2.0
        
        return reward
    
    def _calculate_reward(self, cell_type):
        """Calculate reward based on current cell type"""
        reward = 0
        
        if cell_type == CellType.INCOME:
            self.balance += 20.0
            reward += 5.0
        
        elif cell_type == CellType.EXPENSE:
            expense = 15.0
            self.balance -= expense
            reward -= 5.0
        
        elif cell_type == CellType.LITERACY_TIP:
            self.literacy_score = min(100, self.literacy_score + 10)
            reward += 2.0
        
        return reward
    
    def _get_observation(self):
        """Return the current observation state"""
        # Flatten grid and combine with position and financial info
        return np.concatenate([
            self.position,
            self.grid.flatten(),
            [self.balance],
            [self.literacy_score]
        ]).astype(np.float32)