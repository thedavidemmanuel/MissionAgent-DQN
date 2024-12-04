import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import pygame
import time
from datetime import datetime

from financial_literacy_env import FinancialLiteracyEnv

class VisualizedFinancialTraining:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.CELL_SIZE = 100
        self.GRID_SIZE = 5
        self.WINDOW_SIZE = self.GRID_SIZE * self.CELL_SIZE
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE + 50))
        pygame.display.set_caption("Financial Literacy Training Visualization")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.font = pygame.font.Font(None, 36)

        # Initialize environment and training components
        self.env = FinancialLiteracyEnv()
        self.setup_training()

    def setup_training(self):
        state_size = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n

        # Neural Network
        model = Sequential([
            Flatten(input_shape=(1,) + (state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(num_actions, activation='linear')
        ])

        # Training components
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = EpsGreedyQPolicy(eps=0.1)
        
        # Create DQN agent
        self.dqn = DQNAgent(
            model=model,
            nb_actions=num_actions,
            memory=memory,
            nb_steps_warmup=100,
            target_model_update=1e-2,
            policy=policy
        )
        
        self.dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    def draw_grid(self):
        self.screen.fill(self.WHITE)
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.BLACK, 
                           (i * self.CELL_SIZE, 0), 
                           (i * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE))
            pygame.draw.line(self.screen, self.BLACK, 
                           (0, i * self.CELL_SIZE), 
                           (self.GRID_SIZE * self.CELL_SIZE, i * self.CELL_SIZE))

    def draw_state(self, env_state, episode, step, reward):
        # Draw grid elements
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                cell_center = (j * self.CELL_SIZE + self.CELL_SIZE//2, 
                             i * self.CELL_SIZE + self.CELL_SIZE//2)
                if env_state.grid[i,j] != "":
                    text = self.font.render(env_state.grid[i,j], True, self.BLACK)
                    text_rect = text.get_rect(center=cell_center)
                    self.screen.blit(text, text_rect)

        # Draw agent
        agent_pos = (env_state.position[1] * self.CELL_SIZE + self.CELL_SIZE//2, 
                    env_state.position[0] * self.CELL_SIZE + self.CELL_SIZE//2)
        pygame.draw.circle(self.screen, self.BLUE, agent_pos, self.CELL_SIZE//3)

        # Draw status
        status_text = f"Episode: {episode} | Step: {step} | Balance: ${env_state.balance:.2f} | Reward: {reward:.2f}"
        status = self.font.render(status_text, True, self.BLACK)
        self.screen.blit(status, (10, self.WINDOW_SIZE + 10))

    def train(self):
        # Set up TensorBoard logging
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = f'logs/training_{current_time}'
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        episode = 0
        max_episodes = 100
        steps_per_episode = 50

        while episode < max_episodes:
            observation, info = self.env.reset()
            episode_reward = 0

            for step in range(steps_per_episode):
                # Get action from DQN
                action = self.dqn.forward(observation)
                
                # Take action
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward

                # Visualize
                self.draw_grid()
                self.draw_state(self.env, episode, step, reward)
                pygame.display.flip()

                # Train DQN
                self.dqn.backward(reward, terminated)

                if terminated or truncated:
                    break

                observation = next_observation
                
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                time.sleep(0.1)  # Slow down visualization

            episode += 1
            print(f"Episode: {episode}, Total Reward: {episode_reward}")

            # Save model periodically
            if episode % 10 == 0:
                self.dqn.save_weights(f'models/financial_dqn_weights_{episode}.h5f', overwrite=True)

        # Save final model
        self.dqn.save_weights('models/financial_dqn_weights_final.h5f', overwrite=True)

if __name__ == "__main__":
    trainer = VisualizedFinancialTraining()
    trainer.train()