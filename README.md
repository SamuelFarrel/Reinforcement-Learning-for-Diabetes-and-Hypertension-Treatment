# Overview

This project presents a hybrid **Reinforcement Learning (supported with Supervised Learning)** framework for personalized treatment optimization in **diabetes** and **hypertension** management. The system integrates supervised learning models with deep RL agents to build adaptive treatment strategies that adjust dynamically to individual patient characteristics and health trajectories.

---

# Key Features

- **Dual-Disease Support**  
  Supports treatment optimization for both Type 2 Diabetes and Hypertension.

- **Hybrid Learning Architecture**  
  Reinforcement learning agents are guided using supervised learning predictions to improve stability, convergence, and clinical performance.

- **Multiple RL Algorithms**  
  Implements and compares:
  - Proximal Policy Optimization (PPO)
  - Deep Q-Network (DQN)
  - Advantage Actor-Critic (A2C)

- **Supervised Learning Component**  
  Random Forest classifier with feature engineering, class imbalance handling, and hyperparameter tuning.

- **Patient Simulation Environments**  
  Custom-built diabetes and hypertension environments with health dynamics and reward shaping.

- **Comprehensive Evaluation**  
  Includes reward curves, clinical outcome trajectories, and SL vs non-SL comparisons.

---

# Project Context

This project is the **Final Project for Machine and Reinforcement Learning course** at Fasilkom UI.  
The goal of this project is to evaluate reinforcement learning models for personalized diabetes and hypertension treatment, and to investigate how supervised learning guidance can improve their clinical performance.

---

# Detailed Documentation

For full methodology, algorithm design, results, and analysis, see:

📄 **Paper.pdf**

It includes:
- Literature review  
- RL & SL methodology  
- Diabetes & hypertension feature engineering  
- Environment design  
- Experimental configuration  
- Results, comparisons, visualizations  
- Discussion and limitations  

---

# Key Components

## Environments
- `DiabetesEnv` - models glucose regulation and insulin/diet response  
- `HypertensionEnv` - models blood pressure dynamics and medication effects  

## RL Agents
- **PPO** - stable baseline with strong performance  
- **DQN** - improved significantly with SL guidance  
- **A2C** - actor–critic baseline for comparison  

## Supervised Learning
- **Random Forest Classifier** for:
  - treatment risk scoring  
  - early policy guidance  
  - improved RL stability and reward trajectory  

---

# Installation & Setup

## Prerequisites
- Python **3.10+**
- Conda (Anaconda or Miniconda)

## Clone the repository
```bash
git clone <repository-url>
cd Reinforecement_learning-for_Diabetes-and-Hypertension-Treatment
```

## Create & Activate Environment

Run the following commands:
```bash
conda env create -f code/requirements.yml
conda activate project-m4
```

The environment includes:
- PyTorch
- Stable-Baselines3
- Scikit-learn
- Gymnasium
- NumPy, Pandas
- Matplotlib, Seaborn

---

# Data Requirements

Place these dataset files inside `code/data/`:

- diabetes_train.csv  
- diabetes_test.csv  
- hypertension_train.csv  
- hypertension_test.csv  

---

# Running the Project

## Quick Start
```bash
cd code  
python xprmt.py  
```

---

# Configuration

Example configuration (`code/config.yml`):
```python
# Training seeds for reproducibility
seeds: [42, 123, 456, 789, 1011, 1234, 2345, 3456, 4567, 5678]  

# Environment parameters
n_patients: 500  
max_steps_per_episode: 50

# Agent training parameters
timesteps: 100000  
eval_episodes: 30  
```
---

# Experiment Workflow

1. **Data Pre-processing:** Load & preprocess data  
2. **Supervised Learning:** Train supervised learning model (Random Forest)  
3. **Environment Creation:** Create diabetes and hypertension environments  
4. **Reinforcement Learning:** Train RL agents (PPO, DQN, A2C) with and without SL guidance  
5. **Evaluation:** Evaluate reward progression and clinical outcomes  
6. **Visualization:** Generate visualizations  
7. **Comparison:** Compare SL-guided vs standard RL  
---

# Output Files (code/results/)

## Reward Plots
- _diabetes_rewards_with_sl.png_  
- _diabetes_rewards_without_sl.png_  
- _hypertension_rewards_with_sl.png_  
- _hypertension_rewards_without_sl.png_  

## Health Trajectories
- _diabetes_glucose_trajectory.png_  
- _hypertension_bp_trajectory.png_  

## SL vs RL Comparisons
- _diabetes_sl_vs_nosl_comparison.png_  
- _hypertension_sl_vs_nosl_comparison.png_  

## Supervised Learning Performance
- _sl_performance_across_seeds.png_  
- _sl_confusion_matrices.png_  

---

# Project Structure
```
├── code/
│   ├── xprmt.py                    # Experiment script
│   ├── config.yml                  # Experiment parameters configuration
│   ├── requirements.yml            # Conda environment specification
│   ├── agents/
│   │   └── agents.py               # RL agent implementations
│   ├── data/                       # Dataset files
│   ├── environments/
│   │   ├── diabetes_env.py         # Diabetes patient environment
│   │   └── hypertension_env.py     # Hypertension patient environment
│   ├── models/
│   │   └── supervised_models.py    # Supervised learning models
│   ├── utils/
│   │   ├── utils.py                # Utility and helper functions
│   │   └── visualization.py        # Plotting and visualization
│   └── results/                    # Generated results and plots
├── tex/                            # LaTeX source files (paper raw files)
├── paper.pdf                       # Complete research paper
└── README.md                       # This file Deeplearn.pdf  
```
