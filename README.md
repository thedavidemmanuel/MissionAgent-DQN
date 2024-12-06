# Financial Literacy Grid DQN

## Project Description

This project implements a reinforcement learning environment where an AI agent learns to make smart financial decisions in a grid-based world. The agent represents a person learning to manage their money, navigating through different financial opportunities and challenges while trying to reach their savings goal.

The environment simulates real financial choices through a simple grid where different cells represent various financial situations like income sources, investment opportunities, and spending decisions. The agent must learn to balance risk and reward while maintaining a healthy bank balance.

### Key Features

- 5x5 grid environment representing financial decision-making landscape
- Dynamic interaction between risk and reward through investment mechanics
- Financial literacy scoring that influences decision outcomes
- Real-time visualization of the agent's learning journey
- Clear reward structure based on sound financial principles

## Project Demo

- [Simulation Video Demo](https://youtu.be/your_video_link)
- [Extended Learning Session](https://drive.google.com/file/your_link)

## Requirements

```
Python 3.12
PyTorch
Stable-Baselines3
Pygame
Gymnasium
NumPy
```

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

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Details

### Grid Elements

- 🏦 Bank (Goal): Safe destination for savings
- 💵 Income: Provides financial resources
- 📈 Investment: Opportunities for growth with some risk
- 🛍️ Expenses: Represents spending challenges
- 💡 Financial Tips: Improves decision-making ability

### Agent Actions

- Movement: Up, Down, Left, Right
- Financial Actions: Invest, Save

### Reward Structure

- +10 points for reaching savings goal at bank
- +5 points for collecting income or successful investments
- -5 points for expenses or failed investments
- +2 points for learning financial tips
- Small movement penalty to encourage efficient paths

## Running the Project

### Training the Agent

```bash
python train.py
```

This will:

- Initialize the financial literacy environment
- Train the DQN agent
- Show training progress
- Save the trained model

### Running the Simulation

```bash
python play.py
```

The visualization shows:

- Agent's current position
- Financial opportunities and challenges
- Current balance and literacy score
- Decision outcomes and rewards

### Controls

- SPACE: Pause/Resume simulation
- R: Reset simulation
- ESC: Exit

## Learning Outcomes

Through this simulation, the agent learns to:

- Build savings through smart financial choices
- Manage risk through strategic investments
- Avoid unnecessary expenses
- Improve financial literacy over time
- Plan efficient paths to financial goals

## Performance Metrics

The agent's success is measured by:

- Final balance achieved
- Number of steps to reach goal
- Financial literacy score
- Investment success rate
- Overall reward accumulation

## Project Structure

```
financial_literacy_DQN/
├── financial_literacy_env.py   # Environment definition
├── train.py                   # Training script
├── play.py                    # Visualization script
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
