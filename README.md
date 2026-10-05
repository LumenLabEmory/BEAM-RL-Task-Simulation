# BEAM-RL-Task-Simulation
RL task simulation that will be used for the fMRI portion of the BEAM study.

## Task
Two 2-armed bandits (arms 1 & 2, arms 3 & 4). In each bandit one arm wins 75% of the time and the other 25%.
Each bandit is shown 50 times and the two are randomly interleaved, for 100 trials in total.

## Model
Rescorla-Wagner Q-learning (learning rate α, Q-values start at 0.5) with a softmax choice rule (inverse temperature β).

## Files
- `beam_rl.py`: simulation, likelihood, maximum-likelihood fitting, and parameter recovery functions.
- `Simulation_and_Parameter_Recovery.ipynb`: learning curves for the ten paired (α, β) groups, parameter recovery on
  those groups and on a crossed α × β grid, and a 100- versus 200-trial comparison.
- `outputs/`: figures and recovery tables written by the notebook.
- `Interleaved_75_25.ipynb`: the original simulation notebook, kept for reference.

## Running
```
pip install numpy pandas scipy matplotlib jupyter
jupyter notebook Simulation_and_Parameter_Recovery.ipynb
```
The full notebook fits about 7,000 simulated agents and takes roughly five minutes.
