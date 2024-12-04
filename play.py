import pygame
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from financial_literacy_env import FinancialLiteracyEnv

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 100
GRID_SIZE = 5
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
pygame.display.set_caption("Financial Literacy Agent")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Initialize environment and agent
env = FinancialLiteracyEnv()
state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

# Rebuild the model (same architecture as training)
model = Sequential([
    Flatten(input_shape=(1,) + (state_size,)),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(num_actions, activation='linear')
])

# Configure and create the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, policy=policy)
dqn.compile(optimizer='adam', metrics=['mae'])

# Load trained weights
dqn.load_weights('financial_dqn_weights.h5f')

def draw_grid():
    # Draw grid lines
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(SCREEN, BLACK, (i * CELL_SIZE, 0), 
                        (i * CELL_SIZE, GRID_SIZE * CELL_SIZE))
        pygame.draw.line(SCREEN, BLACK, (0, i * CELL_SIZE), 
                        (GRID_SIZE * CELL_SIZE, i * CELL_SIZE))

def draw_state(env, font):
    # Draw grid elements
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            cell_center = (j * CELL_SIZE + CELL_SIZE//2, 
                         i * CELL_SIZE + CELL_SIZE//2)
            if env.grid[i,j] != "":
                text = font.render(env.grid[i,j], True, BLACK)
                text_rect = text.get_rect(center=cell_center)
                SCREEN.blit(text, text_rect)

    # Draw agent
    agent_pos = (env.position[1] * CELL_SIZE + CELL_SIZE//2, 
                env.position[0] * CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(SCREEN, BLUE, agent_pos, CELL_SIZE//3)

    # Draw status
    status_text = f"Balance: ${env.balance:.2f} | Literacy: {env.literacy_score}"
    status = font.render(status_text, True, BLACK)
    SCREEN.blit(status, (10, WINDOW_SIZE + 10))

def main():
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    observation, info = env.reset()
    done = truncated = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not (done or truncated):
            state = np.array([observation])
            action = dqn.forward(state)
            observation, reward, done, truncated, info = env.step(action)

            # Draw everything
            SCREEN.fill(WHITE)
            draw_grid()
            draw_state(env, font)
            pygame.display.flip()

            if done or truncated:
                pygame.time.wait(2000)  # Pause to show final state
                observation, info = env.reset()
                done = truncated = False

        clock.tick(2)  # Control simulation speed

if __name__ == "__main__":
    main()