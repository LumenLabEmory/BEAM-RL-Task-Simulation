"""Simulation and parameter recovery for the BEAM RL task.

Task: two 2-armed bandits (pair 1 = arms 1 & 2, pair 2 = arms 3 & 4), each with
a 75% arm and a 25% arm. Each bandit is shown `trials_per_pair` times and the
two bandits are randomly interleaved, so the default task is 50 + 50 = 100
trials.

Model: Rescorla-Wagner Q-learning with a softmax choice rule.
    Q[chosen] <- Q[chosen] + alpha * (outcome - Q[chosen])
    P(choose good arm) = 1 / (1 + exp(-beta * (Q_good - Q_bad)))
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

GOOD_ARM_WIN_CHANCE = 0.75
BAD_ARM_WIN_CHANCE = 0.25
TRIALS_PER_PAIR = 50
Q_START = 0.5

ALPHA_BOUNDS = (0.0, 1.0)
BETA_BOUNDS = (0.0, 20.0)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate_participant(alpha, beta, rng, trials_per_pair=TRIALS_PER_PAIR,
                         p_good=GOOD_ARM_WIN_CHANCE, p_bad=BAD_ARM_WIN_CHANCE):
    """Simulate one agent on the interleaved two-bandit task.

    Returns a dict of arrays, one entry per trial:
        pair            1 or 2 (which bandit was shown)
        pair_trial      1..trials_per_pair (how many times this bandit has been shown)
        choice          arm number 1-4 (arms 1 and 3 are the 75% arms)
        chose_good      1 if the 75% arm was chosen
        outcome         1 = win, 0 = lose
        Q               (n_trials, 4) Q-values after the update on that trial
    """
    n_trials = 2 * trials_per_pair
    pair = np.repeat([1, 2], trials_per_pair)
    rng.shuffle(pair)                      # random interleaving, 50 of each

    Q = np.full(4, Q_START)
    choice = np.empty(n_trials, dtype=int)
    outcome = np.empty(n_trials, dtype=int)
    pair_trial = np.empty(n_trials, dtype=int)
    Q_history = np.empty((n_trials, 4))
    seen = {1: 0, 2: 0}

    for t in range(n_trials):
        good = 0 if pair[t] == 1 else 2    # index of the 75% arm for this bandit
        bad = good + 1
        p_choose_good = 1.0 / (1.0 + np.exp(-beta * (Q[good] - Q[bad])))

        if rng.random() < p_choose_good:
            arm, win_chance = good, p_good
        else:
            arm, win_chance = bad, p_bad

        outcome[t] = rng.random() < win_chance
        Q[arm] += alpha * (outcome[t] - Q[arm])

        seen[pair[t]] += 1
        pair_trial[t] = seen[pair[t]]
        choice[t] = arm + 1
        Q_history[t] = Q

    return {"pair": pair, "pair_trial": pair_trial, "choice": choice,
            "chose_good": (choice % 2 == 1).astype(int), "outcome": outcome,
            "Q": Q_history}


def simulate_group(alphas, betas, rng, trials_per_pair=TRIALS_PER_PAIR):
    """Simulate one agent per (alpha, beta) entry and return a long DataFrame."""
    frames = []
    for agent, (alpha, beta) in enumerate(zip(alphas, betas), start=1):
        sim = simulate_participant(alpha, beta, rng, trials_per_pair)
        n = len(sim["pair"])
        frames.append(pd.DataFrame({
            "agent": agent,
            "trial": np.arange(1, n + 1),
            "pair": sim["pair"],
            "pair_trial": sim["pair_trial"],
            "choice": sim["choice"],
            "chose_good": sim["chose_good"],
            "outcome": sim["outcome"],
            "alpha": alpha,
            "beta": beta,
            "Q_arm1": sim["Q"][:, 0],
            "Q_arm2": sim["Q"][:, 1],
            "Q_arm3": sim["Q"][:, 2],
            "Q_arm4": sim["Q"][:, 3],
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Likelihood and fitting
# --------------------------------------------------------------------------

def negative_log_likelihood(alpha, beta, choice, outcome):
    """Negative log-likelihood of one agent's choices.

    `alpha` and `beta` can be scalars or arrays of the same shape; the result
    has that shape. `choice` holds arm numbers 1-4, `outcome` holds 0/1.
    """
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    Q = np.full(alpha.shape + (4,), Q_START)
    nll = np.zeros(alpha.shape)

    for arm, reward in zip(np.asarray(choice) - 1, outcome):
        other = arm + 1 if arm % 2 == 0 else arm - 1   # other arm of the same bandit
        # -log P(chosen arm) = log(1 + exp(-beta * (Q_chosen - Q_other)))
        nll += np.logaddexp(0.0, -beta * (Q[..., arm] - Q[..., other]))
        Q[..., arm] += alpha * (reward - Q[..., arm])
    return nll


_GRID_ALPHA, _GRID_BETA = np.meshgrid(np.linspace(0.025, 0.975, 20),
                                      np.linspace(0.25, 15.0, 30), indexing="ij")


def fit_participant(choice, outcome):
    """Maximum-likelihood estimate of (alpha, beta) for one agent.

    A coarse grid search picks the starting point, then L-BFGS-B refines it
    inside ALPHA_BOUNDS x BETA_BOUNDS. Returns (alpha_hat, beta_hat, nll).
    """
    grid_nll = negative_log_likelihood(_GRID_ALPHA, _GRID_BETA, choice, outcome)
    start = np.unravel_index(np.argmin(grid_nll), grid_nll.shape)
    x0 = [_GRID_ALPHA[start], _GRID_BETA[start]]

    result = minimize(lambda x: float(negative_log_likelihood(x[0], x[1], choice, outcome)),
                      x0, method="L-BFGS-B", bounds=[ALPHA_BOUNDS, BETA_BOUNDS])
    if result.fun <= grid_nll[start]:
        return result.x[0], result.x[1], result.fun
    return x0[0], x0[1], float(grid_nll[start])


def run_recovery(alphas, betas, rng, trials_per_pair=TRIALS_PER_PAIR):
    """Simulate one agent per (alpha, beta), refit each, return one row per agent."""
    rows = []
    for agent, (alpha, beta) in enumerate(zip(alphas, betas), start=1):
        sim = simulate_participant(alpha, beta, rng, trials_per_pair)
        alpha_hat, beta_hat, nll = fit_participant(sim["choice"], sim["outcome"])
        rows.append({"agent": agent, "alpha_true": alpha, "beta_true": beta,
                     "alpha_hat": alpha_hat, "beta_hat": beta_hat, "nll": nll,
                     "accuracy": sim["chose_good"].mean()})
    return pd.DataFrame(rows)
