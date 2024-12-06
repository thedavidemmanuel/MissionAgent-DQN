# play.py

import os
import numpy as np
import pygame
from stable_baselines3 import DQN
import time
from financial_literacy_env import FinancialLiteracyEnv, CellType

class FinancialSimulation:
    """
    Enhanced simulation of the trained financial advisor agent with detailed
    visualization and educational feedback about decision-making processes.
    
    This simulation provides a visual interface to watch the trained agent navigate
    through financial decisions, showing its learning outcomes and decision-making
    process in real-time.
    """
    def __init__(self, screen_width=1024, screen_height=900):
        # Initialize Pygame with a larger window for more detailed information
        pygame.init()
        pygame.display.set_caption("Financial Advisor Simulation - Learning Visualization")
        
        # Calculate dimensions based on grid
        self.env = FinancialLiteracyEnv()
        self.cell_size = 80  # Base cell size for the grid
        self.grid_display_size = self.cell_size * self.env.grid_size
        self.info_height = 300  # Height for information panel
        
        # Create display window with room for grid and info panel
        self.screen = pygame.display.set_mode((
            max(screen_width, self.grid_display_size + 200),
            self.grid_display_size + self.info_height
        ))
        
        # Initialize fonts for different text elements
        self.title_font = pygame.font.Font(None, 48)
        self.info_font = pygame.font.Font(None, 36)
        self.detail_font = pygame.font.Font(None, 24)
        
        # Define color scheme for different cell types
        self.colors = {
            CellType.EMPTY: (240, 240, 240),    # Light gray for empty cells
            CellType.BANK: (0, 200, 0),         # Soft green for bank
            CellType.INCOME: (0, 191, 255),     # Sky blue for income
            CellType.INVESTMENT: (255, 140, 0),  # Orange for investments
            CellType.EXPENSE: (220, 20, 60),    # Red for expenses
            CellType.LITERACY_TIP: (147, 112, 219)  # Purple for learning
        }
        
        # Define informative labels and descriptions for each cell type
        self.labels = {
            CellType.BANK: ("BANK", "Safe savings (+2 reward)"),
            CellType.INCOME: ("INCOME", "+$20 guaranteed"),
            CellType.INVESTMENT: ("INVEST", "50-150% return, risk based on literacy"),
            CellType.EXPENSE: ("EXPENSE", "-$15 cost"),
            CellType.LITERACY_TIP: ("LEARN", "+10 knowledge, better decisions")
        }
        
        # Initialize performance tracking metrics
        self.episode = 1
        self.steps = 0
        self.total_reward = 0
        self.action_history = []
        self.current_action = None
        self.action_description = ""
        self.decision_explanation = ""
        self.paused = False
        
        # Load the trained agent
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained model with error handling"""
        model_path = None
        possible_names = [
            "financial_literacy_model.zip",
            "financial_literacy_model_interrupted.zip"
        ]
        
        # Search for available model files
        for name in possible_names:
            if os.path.exists(name):
                model_path = name
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "No trained model found. Please run train.py first."
            )
        
        print(f"Loading model from: {model_path}")
        return DQN.load(model_path)

    def run(self):
        """Run the simulation with enhanced feedback"""
        clock = pygame.time.Clock()
        running = True
        obs, _ = self.env.reset()
        
        # Display simulation instructions
        print("\nFinancial Advisor Simulation - Learning Session")
        print("==========================================")
        print("Controls:")
        print("  SPACE  - Pause/Resume simulation")
        print("  R      - Reset episode")
        print("  Q/ESC  - Exit simulation")
        print("  D      - Toggle detailed analysis")
        
        show_details = False
        
        while running:
            # Handle user input events
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
                        print("\nEpisode reset")
                    elif event.key == pygame.K_d:
                        show_details = not show_details
            
            if not self.paused:
                # Get the agent's next action
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Store current state for comparison
                prev_pos = self.env.position.copy()
                prev_balance = self.env.balance
                
                # Execute action and get feedback
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Update simulation metrics
                self.steps += 1
                self.total_reward += reward
                self.current_action = action
                
                # Generate explanation of the action
                self.action_description = self._analyze_action(
                    action, prev_pos, self.env.position,
                    prev_balance, self.env.balance, reward
                )
                
                # Record action for history
                self.action_history.append({
                    'action': action,
                    'reward': reward,
                    'balance': self.env.balance,
                    'position': self.env.position.copy()
                })
                
                # Handle episode completion
                if terminated or truncated:
                    self._handle_episode_completion()
                    obs, _ = self.env.reset()
                    self.episode += 1
                    self.steps = 0
                    self.total_reward = 0
                    self.action_history.clear()
            
            # Update the display
            self._render(show_details)
            
            # Control simulation speed
            clock.tick(2)  # 2 FPS for clear visualization
        
        pygame.quit()

    def _analyze_action(self, action, prev_pos, new_pos, prev_balance, new_balance, reward):
        """Provide detailed analysis of the agent's decision"""
        if action < 4:  # Movement actions
            directions = ["Up", "Down", "Left", "Right"]
            movement = directions[action]
            cell_type = self.env.grid[tuple(new_pos)]
            
            if cell_type != CellType.EMPTY:
                return f"Moved {movement} towards {self.labels[cell_type][0]}"
            return f"Moved {movement} (exploring)"
        
        # Financial actions
        if action == 4:  # Invest
            if new_balance > prev_balance:
                return f"Investment successful: +${new_balance - prev_balance:.2f}"
            elif new_balance < prev_balance:
                return f"Investment loss: -${prev_balance - new_balance:.2f}"
            return "Investment (no change)"
        
        if action == 5:  # Save
            if reward > 0:
                return f"Saved successfully: +{reward:.1f} reward"
            return "Attempted to save (invalid location)"

    def _handle_episode_completion(self):
        """Provide detailed feedback on episode completion"""
        print(f"\nEpisode {self.episode} Summary:")
        print(f"Steps taken: {self.steps}")
        print(f"Final Balance: ${self.env.balance:.2f}")
        print(f"Target Balance: ${self.env.target_balance:.2f}")
        print(f"Financial Literacy: {self.env.literacy_score:.1f}")
        print(f"Total Reward: {self.total_reward:.1f}")
        
        if self.env.balance >= self.env.target_balance:
            print("SUCCESS: Financial goals achieved!")
        else:
            print("Note: Keep trying to reach the target balance")

    def _render(self, show_details):
        """Enhanced rendering with detailed visualization"""
        self.screen.fill((240, 240, 240))  # Light gray background
        
        self._draw_grid()
        self._draw_agent()
        self._draw_enhanced_info_panel(show_details)
        
        pygame.display.flip()

    def _draw_grid(self):
        """Draw the grid with enhanced visual feedback"""
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
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
                
                if cell_type in self.labels:
                    label, desc = self.labels[cell_type]
                    
                    label_text = self.detail_font.render(label, True, (0, 0, 0))
                    label_rect = label_text.get_rect(
                        centerx=rect.centerx,
                        centery=rect.centery - 10
                    )
                    self.screen.blit(label_text, label_rect)
                    
                    desc_text = self.detail_font.render(desc, True, (0, 0, 0))
                    desc_rect = desc_text.get_rect(
                        centerx=rect.centerx,
                        centery=rect.centery + 10
                    )
                    self.screen.blit(desc_text, desc_rect)

    def _draw_agent(self):
        """Draw the agent with movement indication"""
        pos = self.env.position
        center = (
            int(pos[1] * self.cell_size + self.cell_size/2),
            int(pos[0] * self.cell_size + self.cell_size/2)
        )
        
        # Draw agent circle
        pygame.draw.circle(
            self.screen,
            (255, 215, 0),  # Golden yellow
            center,
            self.cell_size//3
        )
        
        # Draw movement direction indicator
        if self.current_action is not None and self.current_action < 4:
            direction_color = (0, 0, 0)  # Black
            direction_length = self.cell_size//4
            
            if self.current_action == 0:  # Up
                end_pos = (center[0], center[1] - direction_length)
            elif self.current_action == 1:  # Down
                end_pos = (center[0], center[1] + direction_length)
            elif self.current_action == 2:  # Left
                end_pos = (center[0] - direction_length, center[1])
            else:  # Right
                end_pos = (center[0] + direction_length, center[1])
            
            pygame.draw.line(self.screen, direction_color, center, end_pos, 3)

    def _draw_enhanced_info_panel(self, show_details):
        """Draw enhanced information panel with detailed metrics"""
        panel_y = self.grid_display_size
        
        # Draw episode info
        episode_text = f"Episode: {self.episode}  Steps: {self.steps}"
        text = self.info_font.render(episode_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 10))
        
        # Draw financial state
        balance_text = f"Balance: ${self.env.balance:.2f} / ${self.env.target_balance:.2f}"
        text = self.info_font.render(balance_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 50))
        
        # Draw literacy score with progress bar
        literacy_text = f"Financial Literacy: {self.env.literacy_score:.1f}"
        text = self.info_font.render(literacy_text, True, (0, 0, 0))
        self.screen.blit(text, (10, panel_y + 90))
        
        # Draw literacy progress bar
        bar_width = 200
        bar_height = 20
        progress = self.env.literacy_score / 100.0
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (220, panel_y + 90, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 255, 0),
                        (220, panel_y + 90, bar_width * progress, bar_height))
        
        # Draw current action description
        if self.action_description:
            action_text = f"Action: {self.action_description}"
            text = self.info_font.render(action_text, True, (0, 0, 0))
            self.screen.blit(text, (10, panel_y + 130))
        
        # Draw detailed stats if enabled
        if show_details:
            self._draw_detailed_stats(panel_y + 170)
        
        # Draw pause indicator if needed
        if self.paused:
            pause_text = "PAUSED - Press SPACE to continue"
            text = self.title_font.render(pause_text, True, (255, 0, 0))
            text_rect = text.get_rect(
                center=(self.screen.get_width()//2, panel_y + 250)
            )
            self.screen.blit(text, text_rect)

    def _draw_detailed_stats(self, y_pos):
        """Draw detailed performance statistics"""
        stats_text = [
            f"Total Reward: {self.total_reward:.1f}",
            f"Average Reward: {self.total_reward/max(1, self.steps):.2f}",
            f"Action Count: Move ({sum(1 for a in self.action_history if a['action'] < 4)}), "
            f"Invest ({sum(1 for a in self.action_history if a['action'] == 4)}), "
            f"Save ({sum(1 for a in self.action_history if a['action'] == 5)})"
        ]
        
        for i, text in enumerate(stats_text):
            surface = self.detail_font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, y_pos + i * 25))


def main():
    """Main entry point for running the simulation"""
    try:
        print("\nStarting Financial Education Simulation...")
        print("----------------------------------------")
        print("This simulation demonstrates the agent's learned financial decision-making.")
        print("\nKey Features:")
        print("- Watch the agent navigate financial opportunities and risks")
        print("- See real-time decision analysis")
        print("- Track financial literacy progress")
        print("- Monitor balance growth towards goals")
        
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