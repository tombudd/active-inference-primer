"""
Free Energy Computation
=======================
Core variational free energy (F) and expected free energy (G) functions.

Based on:
    Friston et al. (2019). A free energy principle for a particular physics.
    Da Costa et al. (2020). Active inference on discrete state-spaces.
"""

import numpy as np
from typing import Optional


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = x / temperature
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()


def log_stable(x: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    """Natural log with numerical stability floor."""
    return np.log(np.maximum(x, eps))


def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
    """
    KL[q || p] — divergence from p to q.
    
    KL divergence measures how much information is lost when q is used
    to approximate p. In active inference, this is the 'complexity' term:
    the cost of updating beliefs away from the prior.
    
    Args:
        q: Approximate posterior distribution (agent's beliefs)
        p: Prior distribution (agent's model)
    
    Returns:
        KL divergence (non-negative scalar)
    """
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.sum(q * (log_stable(q) - log_stable(p)))


def variational_free_energy(
    q: np.ndarray,
    likelihood: np.ndarray,
    prior: np.ndarray,
    observation: Optional[int] = None
) -> float:
    """
    Compute variational free energy F.
    
    F = KL[q(x) || p(x)] - E_q[log p(o|x)]
      = Complexity - Accuracy
    
    Minimising F forces the agent to:
    - Stay close to its prior (low complexity)
    - Explain its observations well (high accuracy)
    
    Args:
        q:           Approximate posterior q(x) — shape (n_states,)
        likelihood:  p(o|x) — shape (n_obs, n_states)
        prior:       p(x) — shape (n_states,)
        observation: Index of current observation (None = marginalise)
    
    Returns:
        Scalar free energy value
    """
    complexity = kl_divergence(q, prior)

    if observation is not None:
        log_likelihood = log_stable(likelihood[observation])
        accuracy = float(q @ log_likelihood)
    else:
        # Expected accuracy over observations
        accuracy = float(q @ np.sum(likelihood * log_stable(likelihood), axis=0))

    return complexity - accuracy


def expected_free_energy(
    policy: np.ndarray,
    transition: np.ndarray,
    likelihood: np.ndarray,
    prior_preferences: np.ndarray,
    current_belief: np.ndarray
) -> float:
    """
    Compute expected free energy G(π) for a policy.
    
    G(π) ≈ Pragmatic value + Epistemic value
         = -E_q[log p(o|π)] + KL[q(x|π) || p(x)]
    
    This decomposes into:
    - Pragmatic: how much the policy achieves preferred outcomes
    - Epistemic: how much the policy resolves uncertainty (curiosity/exploration)
    
    Agents that minimise G(π) are both goal-directed AND curious.
    This is the basis for intrinsic motivation in active inference.
    
    Args:
        policy:            Action sequence — shape (n_steps,)
        transition:        p(x'|x,a) — shape (n_states, n_states, n_actions)
        likelihood:        p(o|x) — shape (n_obs, n_states)
        prior_preferences: log p(o) — preferred observations — shape (n_obs,)
        current_belief:    q(x_t) — current state belief — shape (n_states,)
    
    Returns:
        Expected free energy scalar (lower is better)
    """
    belief = current_belief.copy()
    G = 0.0

    for action in policy:
        # Predict next state under this action
        predicted_belief = transition[:, :, action].T @ belief

        # Predicted observations
        predicted_obs = likelihood @ predicted_belief

        # Pragmatic value: E[log p*(o)] — do predictions match preferences?
        pragmatic = -float(predicted_obs @ prior_preferences)

        # Epistemic value: information gain / uncertainty reduction
        # H[p(o|π)] - E_x[H[p(o|x)]]
        # = entropy of predicted obs - expected conditional entropy
        obs_entropy = -float(predicted_obs @ log_stable(predicted_obs))
        cond_entropy = -float(
            predicted_belief @ np.sum(likelihood * log_stable(likelihood), axis=0)
        )
        epistemic = obs_entropy - cond_entropy

        G += pragmatic + epistemic
        belief = predicted_belief

    return G


def policy_posterior(
    policies: list,
    transition: np.ndarray,
    likelihood: np.ndarray,
    prior_preferences: np.ndarray,
    current_belief: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute posterior distribution over policies via softmax over -G(π).
    
    P(π) ∝ exp(-G(π))
    
    This is the active inference decision rule: prefer policies that
    minimise expected free energy.
    
    Args:
        policies:          List of action sequences
        transition:        p(x'|x,a)
        likelihood:        p(o|x)
        prior_preferences: log p*(o)
        current_belief:    q(x_t)
        temperature:       Softmax temperature (precision)
    
    Returns:
        Probability distribution over policies — shape (n_policies,)
    """
    G_values = np.array([
        expected_free_energy(
            policy, transition, likelihood,
            prior_preferences, current_belief
        )
        for policy in policies
    ])

    return softmax(-G_values, temperature=temperature)
