import os
import numpy as np
import pygame
from stable_baselines3 import DQN
from financial_literacy_env import FinancialLiteracyEnv, CellType

class FinancialSimulation:
    """
    Enhanced simulation of the trained financial advisor agent with detailed
    visualization and adaptive exploration.
    """
    def __init__(self, screen_width=1024, screen_height=900):
        pygame.init()
        pygame.display.set_caption("Financial Advisor Simulation - Learning Visualization")
        
        self.env = FinancialLiteracyEnv()
        self.cell_size = 100  # Increased cell size for better text and visuals
        self.grid_display_size = self.cell_size * self.env.grid_size
        self.info_height = 300
        
        self.screen = pygame.display.set_mode((
            max(screen_width, self.grid_display_size + 200),
            self.grid_display_size + self.info_height
        ))
        
        self.title_font = pygame.font.Font(None, 48)
        self.info_font = pygame.font.Font(None, 36)
        self.detail_font = pygame.font.Font(None, 24)
        
        self.colors = {
            CellType.EMPTY: (240, 240, 240),
            CellType.BANK: (0, 200, 0),
            CellType.INCOME: (0, 191, 255),
            CellType.INVESTMENT: (255, 140, 0),
            CellType.EXPENSE: (220, 20, 60),
            CellType.LITERACY_TIP: (147, 112, 219)
        }
        
        self.labels = {
            CellType.BANK: ("BANK", "+2 reward"),
            CellType.INCOME: ("INCOME", "+$20"),
            CellType.INVESTMENT: ("INVEST", "50%-150% return"),
            CellType.EXPENSE: ("EXPENSE", "-$15"),
            CellType.LITERACY_TIP: ("LEARN", "+10 knowledge")
        }
        
        self.episode = 1
        self.steps = 0
        self.total_reward = 0
        self.action_history = []
        self.current_action = None
        self.action_description = ""
        self.decision_explanation = ""
        self.paused = False
        self.visit_count = np.zeros((self.env.grid_size, self.env.grid_size))
        
        self.model = self._load_model()

    def _load_model(self):
        model_path = None
        possible_names = ["financial_literacy_model.zip", "financial_literacy_model_interrupted.zip"]
        
        for name in possible_names:
            if os.path.exists(name):
                model_path = name
                break
        
        if model_path is None:
            raise FileNotFoundError("No trained model found. Please run train.py first.")
        
        print(f"Loading model from: {model_path}")
        return DQN.load(model_path)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        obs, _ = self.env.reset()
        
        print("\nFinancial Advisor Simulation - Learning Session")
        print("==========================================")
        print("Controls:")
        print("  SPACE  - Pause/Resume simulation")
        print("  R      - Reset episode")
        print("  Q/ESC  - Exit simulation")
        print("  D      - Toggle detailed analysis")
        
        show_details = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print("Simulation", "paused" if self.paused else "resumed")
                    elif event.key == pygame.K_r:
                        obs, _ = self.env.reset()
                        self.episode += 1
                        self.steps = 0
                        self.total_reward = 0
                        self.action_history.clear()
                        self.visit_count.fill(0)
                        print("\nEpisode reset")
                    elif event.key == pygame.K_d:
                        show_details = not show_details
            
            if not self.paused:
                exploration_prob = 0.1
                if np.random.rand() < exploration_prob:
                    action = self.env.action_space.sample()
                    self.current_action = "Randomly chosen action for exploration"
                else:
                    action, _states = self.model.predict(obs, deterministic=True)
                    self.current_action = f"Action {action} (predicted)"
                
                prev_pos = self.env.position.copy()
                prev_balance = self.env.balance
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.steps += 1
                self.total_reward += reward
                self.visit_count[tuple(self.env.position)] += 1
                
                self.action_description = self._analyze_action(
                    action, prev_pos, self.env.position, prev_balance, self.env.balance, reward
                )
                
                self.action_history.append({
                    'action': action,
                    'reward': reward,
                    'balance': self.env.balance,
                    'position': self.env.position.copy()
                })
                
                if terminated or truncated:
                    self._handle_episode_completion()
                    obs, _ = self.env.reset()
                    self.episode += 1
                    self.steps = 0
                    self.total_reward = 0
                    self.action_history.clear()
                    self.visit_count.fill(0)
            
            self._render(show_details)
            clock.tick(2)
        
        pygame.quit()

    def _handle_episode_completion(self):
        """Provide detailed feedback on episode completion."""
        print(f"\nEpisode {self.episode} Summary:")
        print(f"Steps taken: {self.steps}")
        print(f"Final Balance: ${self.env.balance:.2f}")
        print(f"Target Balance: ${self.env.target_balance:.2f}")
        print(f"Financial Literacy: {self.env.literacy_score:.1f}")
        print(f"Total Reward: {self.total_reward:.1f}")
        
        if self.env.balance >= self.env.target_balance:
            print("SUCCESS: Financial goals achieved!")
        else:
            print("NOTE: Keep trying to reach the target balance.")

    def _analyze_action(self, action, prev_pos, new_pos, prev_balance, new_balance, reward):
        if action < 4:
            directions = ["Up", "Down", "Left", "Right"]
            movement = directions[action]
            cell_type = self.env.grid[tuple(new_pos)]
            if cell_type != CellType.EMPTY:
                return f"Moved {movement} towards {self.labels[cell_type][0]}"
            return f"Moved {movement} (exploring)"
        if action == 4:
            if new_balance > prev_balance:
                return f"Investment successful: +${new_balance - prev_balance:.2f}"
            elif new_balance < prev_balance:
                return f"Investment loss: -${prev_balance - new_balance:.2f}"
            return "Investment (no change)"
        if action == 5:
            return f"Saved successfully: +{reward:.1f} reward" if reward > 0 else "Attempted to save (invalid location)"
        return "Unknown action"

    def _render(self, show_details):
        self.screen.fill((240, 240, 240))
        self._draw_heatmap()
        self._draw_grid()
        self._draw_agent()
        self._draw_enhanced_info_panel(show_details)
        pygame.display.flip()

    def _draw_grid(self):
        """Draw the grid with cell labels and point effects."""
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                rect = pygame.Rect(
                    j * self.cell_size, 
                    i * self.cell_size,
                    self.cell_size - 2, 
                    self.cell_size - 2
                )
                cell_type = self.env.grid[i, j]
                pygame.draw.rect(self.screen, self.colors[cell_type], rect)
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)  # Border
                
                if cell_type in self.labels:
                    label, desc = self.labels[cell_type]
                    
                    # Dynamic font scaling
                    label_font = pygame.font.Font(None, max(16, self.cell_size // 6))
                    desc_font = pygame.font.Font(None, max(12, self.cell_size // 8))
                    
                    label_text = label_font.render(label, True, (0, 0, 0))
                    label_rect = label_text.get_rect(
                        center=(rect.centerx, rect.centery - self.cell_size // 8)
                    )
                    self.screen.blit(label_text, label_rect)
                    
                    desc_text = desc_font.render(desc, True, (0, 0, 0))
                    desc_rect = desc_text.get_rect(
                        center=(rect.centerx, rect.centery + self.cell_size // 8)
                    )
                    self.screen.blit(desc_text, desc_rect)

    def _draw_agent(self):
        pos = self.env.position
        center = (
            int(pos[1] * self.cell_size + self.cell_size / 2),
            int(pos[0] * self.cell_size + self.cell_size / 2)
        )
        pygame.draw.circle(self.screen, (255, 215, 0), center, self.cell_size // 3)

    def _draw_heatmap(self):
        max_visits = np.max(self.visit_count)
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                if max_visits > 0:
                    intensity = 255 - int((self.visit_count[i, j] / max_visits) * 255)
                else:
                    intensity = 255
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size - 2,
                    self.cell_size - 2
                )
                pygame.draw.rect(self.screen, (intensity, intensity, intensity), rect)

    def _draw_enhanced_info_panel(self, show_details):
        panel_y = self.grid_display_size
        episode_text = f"Episode: {self.episode}  Steps: {self.steps}"
        text = self.info_font.render(episode_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 10))
        
        balance_text = f"Balance: ${self.env.balance:.2f} / ${self.env.target_balance:.2f}"
        text = self.info_font.render(balance_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 50))
        
        literacy_text = f"Financial Literacy: {self.env.literacy_score:.1f}"
        text = self.info_font.render(literacy_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 90))
        
        if show_details:
            self._draw_detailed_stats(panel_y + 170)
        
        if self.paused:
            pause_text = "PAUSED - Press SPACE to continue"
            text = self.title_font.render(pause_text, True, (255, 0, 0))
            text_rect = text.get_rect(center=(self.screen.get_width() // 2, panel_y + 250))
            self.screen.blit(text, text_rect)

    def _draw_detailed_stats(self, y_pos):
        stats_text = [
            f"Total Reward: {self.total_reward:.1f}",
            f"Average Reward: {self.total_reward / max(1, self.steps):.2f}",
            f"Action Count: Move ({sum(1 for a in self.action_history if a['action'] < 4)}), "
            f"Invest ({sum(1 for a in self.action_history if a['action'] == 4)}), "
            f"Save ({sum(1 for a in self.action_history if a['action'] == 5)})"
        ]
        for i, text in enumerate(stats_text):
            surface = self.detail_font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, y_pos + i * 25))

def main():
    try:
        print("\nStarting Financial Education Simulation...")
        print("----------------------------------------")
        simulation = FinancialSimulation()
        simulation.run()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run train.py first to create a trained model.")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")
    finally:
        pygame.quit()
        print("\nSimulation ended. Thank you for learning about financial decision-making!")

if __name__ == "__main__":
    main()
