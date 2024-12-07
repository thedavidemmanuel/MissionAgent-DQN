````markdown
# Financial Literacy Grid DQN

## Project Description

The **Financial Literacy Grid DQN** project is a reinforcement learning simulation designed to teach an AI agent how to make smart financial decisions. The agent represents an individual navigating a grid-based world that simulates various financial opportunities and challenges, such as earning income, making investments, incurring expenses, and learning financial tips. The primary goal is to help the agent achieve its savings target while balancing risks and rewards through effective decision-making.

This project serves as a unique educational tool for understanding financial principles through reinforcement learning. It incorporates financial literacy scoring, risk management, and dynamic decision-making in a gamified environment.

---

## Key Features

- **Grid-Based Financial World**: A 5x5 grid where each cell represents a financial situation (e.g., bank, income, expenses, etc.).
- **Dynamic Investment System**: Investment opportunities with varying risks and returns based on financial literacy.
- **Financial Literacy Scoring**: Improves the agent's ability to make better decisions.
- **Real-Time Visualization**: See the agent's learning journey unfold with visual feedback.
- **Reward System**: Based on sound financial principles to encourage responsible decisions.

---

## Demo

- 📹 [Simulation Video Demo](https://youtu.be/your_video_link)
- 📂 [Extended Learning Session](https://drive.google.com/file/your_link)

---

## Requirements

The following are required to run the project:

```plaintext
Python 3.12
PyTorch
Stable-Baselines3
Pygame
Gymnasium
NumPy
```
````

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/financial_literacy_DQN.git
   cd financial_literacy_DQN
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Environment Overview

### Grid Elements

- **🏦 Bank (Goal)**: The safe destination to save money and achieve the target balance.
- **💵 Income**: Provides financial resources to boost balance.
- **📈 Investment**: Offers growth opportunities with varying risks.
- **🛍️ Expenses**: Deducts money from the balance, representing spending challenges.
- **💡 Financial Tips**: Increases the agent's financial literacy score, improving decision outcomes.

### Agent Actions

- **Movement**: The agent can move Up, Down, Left, or Right to explore the grid.
- **Financial Actions**: The agent can choose to Invest or Save when on relevant cells.

---

## Reward System

The reward system is designed to reinforce good financial behaviors:

- **+10 points**: Reaching the bank with the savings goal.
- **+5 points**: Collecting income or making successful investments.
- **-5 points**: Incurring expenses or failing an investment.
- **+2 points**: Learning financial tips.
- **Small penalties**: For excessive or inefficient movement to encourage planning.

---

## Running the Project

### Training the Agent

Run the following command to train the agent:

```bash
python train.py
```

This process will:

- Initialize the financial literacy environment.
- Train the agent using the DQN algorithm.
- Display training progress in real-time.
- Save the trained model for future use.

### Running the Simulation

Once the agent is trained, run the simulation with:

```bash
python play.py
```

The simulation provides a visual representation of:

- The agent's position on the grid.
- Financial opportunities and challenges in the environment.
- Real-time updates on balance, literacy score, and rewards.
- Decision-making outcomes.

---

## Simulation Controls

- **SPACE**: Pause or resume the simulation.
- **R**: Reset the simulation.
- **ESC**: Exit the simulation.

---

## Learning Outcomes

Through this project, the agent learns to:

- Build savings by identifying and capitalizing on income opportunities.
- Manage risks through informed investments.
- Avoid unnecessary expenses.
- Improve financial literacy and make better long-term decisions.
- Plan efficient paths to financial goals while balancing immediate challenges.

---

## Performance Metrics

The agent's performance is evaluated using the following metrics:

- **Final Balance**: The total savings at the end of the episode.
- **Steps Taken**: Efficiency in reaching the financial goal.
- **Literacy Score**: The agent's understanding of financial principles.
- **Investment Success Rate**: Ratio of successful to failed investments.
- **Total Reward**: Accumulated rewards throughout the episode.

---

## Project Structure

```plaintext
financial_literacy_DQN/
├── financial_literacy_env.py  # Environment definition
├── train.py                   # Training script for the DQN agent
├── play.py                    # Simulation and visualization script
├── requirements.txt           # Project dependencies
└── README.md                  # Documentation (this file)
```

---

## Contributing

Contributions are welcome! If you'd like to improve the project or add new features, please open an issue or submit a pull request. Make sure to review the contributing guidelines first.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

### Acknowledgments

Special thanks to the open-source community for the libraries used in this project, including:

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Pygame](https://www.pygame.org/)
- [Gymnasium](https://gymnasium.farama.org/)

---

Feel free to reach out with questions or suggestions for improvement! 🎉

```

---
```
