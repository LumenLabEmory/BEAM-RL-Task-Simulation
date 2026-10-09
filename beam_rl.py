"""Simulation and parameter recovery for the BEAM RL task.

Task: two 2-armed bandits (pair 1 = arms 1 & 2, pair 2 = arms 3 & 4), each with
a good arm and a bad arm (75% / 25% by default; 70% / 30% is also tested). Each bandit is shown `trials_per_pair` times and the
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
# task we are keeping: 50 trials of each bandit, 100 total.
# 100 trials of each bandit (200 total) is only a comparison in the notebook.

TRIALS_PER_PAIR = 50
Q_START = 0.5  # every arm starts at 0.5, before any outcomes

# search limits used when fitting alpha and beta back from choices.
ALPHA_BOUNDS = (0.0, 1.0)
BETA_BOUNDS = (0.0, 20.0)


# simulation

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
    n_trials = 2 * trials_per_pair  # 50 + 50 = 100, not 100 of each bandit
    # one copy of each bandit for every trial it should appear, then shuffled
    # so the two bandits are mixed together across the 100 trials.
    pair = np.repeat([1, 2], trials_per_pair)
    rng.shuffle(pair)

    Q = np.full(4, Q_START)
    choice = np.empty(n_trials, dtype=int)
    outcome = np.empty(n_trials, dtype=int)
    pair_trial = np.empty(n_trials, dtype=int)
    Q_history = np.empty((n_trials, 4))
    seen = {1: 0, 2: 0}  # how many times each bandit has already been shown

    for t in range(n_trials):
        # Pair 1 offers arms 1 and 2; pair 2 offers arms 3 and 4.
        # The even index (0 or 2) is that bandit's 75% arm.
        good = 0 if pair[t] == 1 else 2
        bad = good + 1
        # Softmax: a larger beta makes the higher-Q arm much more likely.
        p_choose_good = 1.0 / (1.0 + np.exp(-beta * (Q[good] - Q[bad])))

        if rng.random() < p_choose_good:
            arm, win_chance = good, p_good
        else:
            arm, win_chance = bad, p_bad

        outcome[t] = rng.random() < win_chance
        # Update only the chosen arm. The other bandit's Q-values stay put.
        Q[arm] += alpha * (outcome[t] - Q[arm])

        seen[pair[t]] += 1
        pair_trial[t] = seen[pair[t]]  # 1..50 within this bandit, for the learning curves
        choice[t] = arm + 1            # store arm numbers 1-4, not 0-3 indexes
        Q_history[t] = Q

    return {"pair": pair, "pair_trial": pair_trial, "choice": choice,
            "chose_good": (choice % 2 == 1).astype(int), "outcome": outcome,
            "Q": Q_history}


def simulate_group(alphas, betas, rng, trials_per_pair=TRIALS_PER_PAIR,
                   p_good=GOOD_ARM_WIN_CHANCE, p_bad=BAD_ARM_WIN_CHANCE):
    """Simulate one agent per (alpha, beta) entry and return a long DataFrame.

    Pass the same (alpha, beta) many times to get more agents in that bin.
    The notebook does this (10, then 100 agents per group) so the high-alpha
    and high-beta learning curves average out.
    """
    frames = []
    for agent, (alpha, beta) in enumerate(zip(alphas, betas), start=1):
        sim = simulate_participant(alpha, beta, rng, trials_per_pair, p_good, p_bad)
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


# likelihood and fitting

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
        # The other arm of the same bandit: 0<->1 and 2<->3.
        other = arm + 1 if arm % 2 == 0 else arm - 1
        # -log P(chosen arm) = log(1 + exp(-beta * (Q_chosen - Q_other))).
        # logaddexp keeps this stable when beta * Q difference is large.
        nll += np.logaddexp(0.0, -beta * (Q[..., arm] - Q[..., other]))
        # Replay the same update the agent used, so the Q-values stay in step
        # with the choices we are scoring.
        Q[..., arm] += alpha * (reward - Q[..., arm])
    return nll


_GRID_ALPHA, _GRID_BETA = np.meshgrid(np.linspace(0.025, 0.975, 20),
                                      np.linspace(0.25, 15.0, 30), indexing="ij")


def fit_participant(choice, outcome):
    """Maximum-likelihood estimate of (alpha, beta) for one agent.

    This is the fitting half of parameter recovery: given only the choices
    and outcomes, estimate the alpha and beta that generated them.

    A coarse grid search picks the starting point, then L-BFGS-B refines it
    inside ALPHA_BOUNDS x BETA_BOUNDS. Returns (alpha_hat, beta_hat, nll).
    """
    # Score every (alpha, beta) on the grid at once. The likelihood function
    # accepts arrays, so this is one pass rather than a Python loop over cells.
    grid_nll = negative_log_likelihood(_GRID_ALPHA, _GRID_BETA, choice, outcome)
    start = np.unravel_index(np.argmin(grid_nll), grid_nll.shape)
    x0 = [_GRID_ALPHA[start], _GRID_BETA[start]]

    result = minimize(lambda x: float(negative_log_likelihood(x[0], x[1], choice, outcome)),
                      x0, method="L-BFGS-B", bounds=[ALPHA_BOUNDS, BETA_BOUNDS])
    # Keep the grid point if the optimizer walks somewhere worse.
    if result.fun <= grid_nll[start]:
        return result.x[0], result.x[1], result.fun
    return x0[0], x0[1], float(grid_nll[start])


def run_recovery(alphas, betas, rng, trials_per_pair=TRIALS_PER_PAIR,
                 p_good=GOOD_ARM_WIN_CHANCE, p_bad=BAD_ARM_WIN_CHANCE):
    """Simulate one agent per (alpha, beta), refit each, return one row per agent.

    Parameter recovery: generate data from a known alpha and beta, hide those
    values, fit them from choices and outcomes, then compare fitted to true.
    Default length is 50 trials per bandit. Pass trials_per_pair=100 for the
    200-trial comparison. p_good / p_bad set the arms' win chances
    (default 75/25; pass 0.70, 0.30 for the 70/30 version of the task).
    """
    rows = []
    for agent, (alpha, beta) in enumerate(zip(alphas, betas), start=1):
        sim = simulate_participant(alpha, beta, rng, trials_per_pair, p_good, p_bad)
        # Fit sees choices and outcomes only, not the true alpha and beta.
        alpha_hat, beta_hat, nll = fit_participant(sim["choice"], sim["outcome"])
        rows.append({"agent": agent, "alpha_true": alpha, "beta_true": beta,
                     "alpha_hat": alpha_hat, "beta_hat": beta_hat, "nll": nll,
                     "accuracy": sim["chose_good"].mean()})
    return pd.DataFrame(rows)
