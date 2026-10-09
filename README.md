# BEAM-RL-Task-Simulation
RL task simulation that will be used for the fMRI portion of the BEAM study.

## Task
Two 2-armed bandits (arms 1 & 2, arms 3 & 4). In each bandit one arm wins 75% of the time and the other 25%.
Each bandit is shown 50 times and the two are randomly interleaved, for 100 trials in total.
A 70/30 version (good arm 70%, bad arm 30%) is also simulated for comparison.

## Model
Rescorla-Wagner Q-learning (learning rate α, Q-values start at 0.5) with a softmax choice rule (inverse temperature β).

## Files
- `beam_rl.py`: simulation, likelihood, maximum-likelihood fitting, and parameter recovery functions.
- `Simulation_and_Parameter_Recovery.ipynb`: learning curves for the ten paired (α, β) groups, parameter recovery on
  those groups and on a crossed α × β grid, and a 100- versus 200-trial comparison.
- `Simulation_and_Parameter_Recovery_70_30.ipynb`: the same analysis with 70/30 arms.
- `Compare_75_25_vs_70_30.ipynb`: side-by-side comparison of the two schedules (run the two notebooks above first).
- `outputs/`: figures and recovery tables for 75/25; `outputs/70_30/` for 70/30; `outputs/comparison/` for the comparison.
- `Interleaved_75_25.ipynb`: the original simulation notebook, kept for reference.

## Running
```
pip install numpy pandas scipy matplotlib jupyter
jupyter notebook Simulation_and_Parameter_Recovery.ipynb
```
The full notebook fits about 7,000 simulated agents and takes roughly five minutes.
