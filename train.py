import os
import numpy as np
import pygame
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import time
import logging

from financial_literacy_env import FinancialLiteracyEnv, CellType

# Configure logging
logging.basicConfig(
    filename="training_log.txt",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class VisualCallback(BaseCallback):
    """
    Custom callback for visualizing training progress in real-time.
    Shows the agent's current position, financial state, and learning progress.
    """
    def __init__(self, verbose=0, render_interval=50):
        super().__init__(verbose)
        self.pygame_initialized = False
        self.render_interval = render_interval

        # Visual settings
        self.cell_size = 100
        self.info_height = 150

        # Training metrics
        self.episode_count = 0
        self.best_reward = -float('inf')
        self.current_reward = 0

        # Color scheme for different cell types
        self.colors = {
            CellType.EMPTY: (255, 255, 255),
            CellType.BANK: (0, 255, 0),
            CellType.INCOME: (0, 255, 255),
            CellType.INVESTMENT: (255, 165, 0),
            CellType.EXPENSE: (255, 0, 0),
            CellType.LITERACY_TIP: (147, 112, 219),
        }

        # Text labels and reward descriptions
        self.labels = {
            CellType.BANK: ("BANK", "+2 reward"),
            CellType.INCOME: ("INCOME", "+$20"),
            CellType.INVESTMENT: ("INVEST", "50%-150% return"),
            CellType.EXPENSE: ("EXPENSE", "-$15"),
            CellType.LITERACY_TIP: ("TIP", "+10 knowledge"),
        }

    def _init_pygame(self, env):
        """Initialize the Pygame display."""
        pygame.init()
        base_env = env.unwrapped
        width = base_env.grid_size * self.cell_size
        height = width + self.info_height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Financial Literacy Training")
        self.label_font = pygame.font.Font(None, max(16, self.cell_size // 6))  # Small font for labels
        self.desc_font = pygame.font.Font(None, max(12, self.cell_size // 8))  # Smaller font for descriptions
        self.info_font = pygame.font.Font(None, 36)
        self.pygame_initialized = True

    def render_env(self, env):
        """Render the current state of the environment."""
        base_env = env.unwrapped
        self.screen.fill((240, 240, 240))

        # Draw grid cells
        for i in range(base_env.grid_size):
            for j in range(base_env.grid_size):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size - 2,
                    self.cell_size - 2,
                )
                cell_type = base_env.grid[i, j]
                pygame.draw.rect(self.screen, self.colors[cell_type], rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)

                # Add text labels and descriptions
                if cell_type in self.labels:
                    label, desc = self.labels[cell_type]

                    # Render label (e.g., "BANK")
                    label_text = self.label_font.render(label, True, (0, 0, 0))
                    label_rect = label_text.get_rect(
                        center=(rect.centerx, rect.centery - self.cell_size // 8)
                    )
                    self.screen.blit(label_text, label_rect)

                    # Render reward/description (e.g., "+2 reward")
                    desc_text = self.desc_font.render(desc, True, (0, 0, 0))
                    desc_rect = desc_text.get_rect(
                        center=(rect.centerx, rect.centery + self.cell_size // 8)
                    )
                    self.screen.blit(desc_text, desc_rect)

        # Draw agent
        agent_pos = base_env.position
        pygame.draw.circle(
            self.screen,
            (255, 255, 0),  # Yellow
            (int(agent_pos[1] * self.cell_size + self.cell_size / 2),
             int(agent_pos[0] * self.cell_size + self.cell_size / 2)),
            self.cell_size // 3,
        )

        # Draw information panel
        self._draw_info_panel(base_env)
        pygame.display.flip()

    def _draw_info_panel(self, env):
        """Draw the information panel."""
        panel_y = env.grid_size * self.cell_size
        episode_text = f"Episode: {self.episode_count}"
        text = self.info_font.render(episode_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 10))
        balance_text = f"Balance: ${env.balance:.2f}"
        text = self.info_font.render(balance_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 50))
        literacy_text = f"Literacy: {env.literacy_score:.1f}"
        text = self.info_font.render(literacy_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 90))
        reward_text = f"Best Reward: {self.best_reward:.1f}"
        text = self.info_font.render(reward_text, True, (0, 0, 0))
        self.screen.blit(text, (300, panel_y + 10))

    def _on_step(self):
        """Update visualization after each step."""
        try:
            if not self.pygame_initialized:
                self._init_pygame(self.training_env.envs[0])
            
            if self.n_calls % self.render_interval == 0:
                self.render_env(self.training_env.envs[0])
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
            
            base_env = self.training_env.envs[0].unwrapped
            self.current_reward += base_env.current_episode_reward
            if self.locals.get('dones')[0]:
                self.episode_count += 1
                self.best_reward = max(self.best_reward, self.current_reward)
                self.current_reward = 0
                logging.info(f"Episode {self.episode_count}: Best Reward {self.best_reward:.1f}")
            return True
        except Exception as e:
            logging.error(f"Error in visualization: {e}")
            return True

def train():
    """Train the financial literacy agent."""
    print("\nStarting Financial Literacy Training...")
    env = FinancialLiteracyEnv()
    env = Monitor(env)
    
    # Hyperparameters
    hyperparams = {
        "learning_rate": 0.001,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy_kwargs": dict(net_arch=[128, 128], activation_fn=nn.ReLU),
        "verbose": 1,
    }
    
    model = DQN("MlpPolicy", env, **hyperparams)
    viz_callback = VisualCallback(render_interval=100)
    
    try:
        print("\nTraining the agent...")
        model.learn(
            total_timesteps=500000,
            callback=viz_callback,
            progress_bar=True,
        )
        model.save("financial_literacy_model")
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted! Saving current model...")
        model.save("financial_literacy_model_interrupted")
    except Exception as e:
        logging.error(f"Training error: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    train()
