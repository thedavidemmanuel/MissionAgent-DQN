# train.py

import os
import numpy as np
import pygame
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time

from financial_literacy_env import FinancialLiteracyEnv, CellType

class VisualCallback(BaseCallback):
    """
    Custom callback for visualizing training progress in real-time.
    Shows the agent's current position, financial state, and learning progress.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pygame_initialized = False
        
        # Visual settings
        self.cell_size = 120
        self.info_height = 150
        
        # Training metrics
        self.episode_count = 0
        self.best_reward = -float('inf')
        self.current_reward = 0
        
        # Color scheme for different cell types
        self.colors = {
            CellType.EMPTY: (255, 255, 255),    # White
            CellType.BANK: (0, 255, 0),         # Green
            CellType.INCOME: (0, 255, 255),     # Cyan
            CellType.INVESTMENT: (255, 165, 0),  # Orange
            CellType.EXPENSE: (255, 0, 0),      # Red
            CellType.LITERACY_TIP: (147, 112, 219)  # Purple
        }
        
        # Cell labels for clarity
        self.labels = {
            CellType.BANK: "BANK",
            CellType.INCOME: "INCOME",
            CellType.INVESTMENT: "INVEST",
            CellType.EXPENSE: "EXPENSE",
            CellType.LITERACY_TIP: "TIP"
        }

    def _init_pygame(self, env):
        """Initialize the Pygame display"""
        pygame.init()
        # Access the unwrapped environment to get grid_size
        base_env = env.unwrapped
        width = base_env.grid_size * self.cell_size
        height = width + self.info_height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Financial Literacy Training")
        self.font = pygame.font.Font(None, 36)
        self.pygame_initialized = True

    def render_env(self, env):
        """Render the current state of the environment"""
        # Get the unwrapped environment
        base_env = env.unwrapped
        
        # Clear screen
        self.screen.fill((240, 240, 240))
        
        # Draw grid cells
        for i in range(base_env.grid_size):
            for j in range(base_env.grid_size):
                rect = pygame.Rect(
                    j * self.cell_size, 
                    i * self.cell_size,
                    self.cell_size - 2, 
                    self.cell_size - 2
                )
                
                # Draw cell background
                cell_type = base_env.grid[i, j]
                pygame.draw.rect(self.screen, self.colors[cell_type], rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
                
                # Draw cell label if it has one
                if cell_type in self.labels:
                    label = self.font.render(
                        self.labels[cell_type], 
                        True, 
                        (0, 0, 0)
                    )
                    label_rect = label.get_rect(center=rect.center)
                    self.screen.blit(label, label_rect)
        
        # Draw agent
        agent_pos = base_env.position
        pygame.draw.circle(
            self.screen,
            (255, 255, 0),  # Yellow
            (int(agent_pos[1] * self.cell_size + self.cell_size/2),
             int(agent_pos[0] * self.cell_size + self.cell_size/2)),
            self.cell_size//3
        )
        
        # Draw information panel
        self._draw_info_panel(base_env)
        
        pygame.display.flip()

    def _draw_info_panel(self, env):
        """Draw the information panel showing current training status"""
        panel_y = env.grid_size * self.cell_size
        
        # Training progress
        episode_text = f"Episode: {self.episode_count}"
        text = self.font.render(episode_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 10))
        
        # Financial state
        balance_text = f"Balance: ${env.balance:.2f}"
        text = self.font.render(balance_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 50))
        
        # Literacy score
        literacy_text = f"Financial Literacy: {env.literacy_score:.1f}"
        text = self.font.render(literacy_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 90))
        
        # Best reward
        reward_text = f"Best Reward: {self.best_reward:.1f}"
        text = self.font.render(reward_text, True, (0, 0, 0))
        self.screen.blit(text, (300, panel_y + 10))

    def _on_step(self):
        """Update visualization after each step"""
        try:
            if not self.pygame_initialized:
                self._init_pygame(self.training_env.envs[0])
            
            # Update metrics
            base_env = self.training_env.envs[0].unwrapped
            self.current_reward += base_env.current_episode_reward
            
            # Update visualization every few steps
            if self.n_calls % 10 == 0:
                self.render_env(self.training_env.envs[0])
                
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                
                time.sleep(0.1)  # Small delay for visibility
            
            # Check if episode is done
            if self.locals.get('dones')[0]:
                self.episode_count += 1
                self.best_reward = max(self.best_reward, self.current_reward)
                self.current_reward = 0
            
            return True
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            return True  # Continue training even if visualization fails

def train():
    """Execute the main training loop"""
    print("\nFinancial Literacy Agent Training")
    print("================================")
    
    # Create and wrap the environment
    env = FinancialLiteracyEnv()
    env = Monitor(env)
    
    # Initialize the DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.001,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=nn.ReLU
        ),
        verbose=1
    )
    
    # Create visualization callback
    viz_callback = VisualCallback()
    
    try:
        print("\nStarting training...")
        print("Controls:")
        print("  - Close window to stop training")
        print("  - Training progress shown in real-time")
        print("  - Model saves automatically when training ends")
        
        # Train the agent
        model.learn(
            total_timesteps=500000,
            callback=viz_callback,
            progress_bar=True
        )
        
        # Save the trained model
        model.save("financial_literacy_model")
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save("financial_literacy_model_interrupted")
    except Exception as e:
        print(f"\nError during training: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    train()