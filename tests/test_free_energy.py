from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.free_energy import (  # noqa: E402
    expected_free_energy,
    kl_divergence,
    log_stable,
    policy_posterior,
    softmax,
    variational_free_energy,
)


def test_log_stable_handles_zero_without_negative_infinity() -> None:
    values = log_stable(np.array([0.0, 1.0]))

    assert np.isfinite(values).all()
    assert values[1] == 0.0


def test_softmax_returns_probability_distribution() -> None:
    probabilities = softmax(np.array([1.0, 2.0, 3.0]))

    assert np.isclose(probabilities.sum(), 1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]


def test_kl_divergence_is_zero_for_matching_distributions() -> None:
    distribution = np.array([0.25, 0.75])

    assert np.isclose(kl_divergence(distribution, distribution), 0.0)


def test_variational_free_energy_rewards_accurate_observation_model() -> None:
    q = np.array([0.9, 0.1])
    prior = np.array([0.5, 0.5])
    likelihood = np.array(
        [
            [0.95, 0.05],
            [0.05, 0.95],
        ]
    )

    observed_f = variational_free_energy(q, likelihood, prior, observation=0)
    mismatched_f = variational_free_energy(q, likelihood, prior, observation=1)

    assert observed_f < mismatched_f


def test_expected_free_energy_prefers_preferred_observation() -> None:
    transition = np.zeros((2, 2, 2))
    transition[:, :, 0] = np.array([[1.0, 0.0], [1.0, 0.0]])
    transition[:, :, 1] = np.array([[0.0, 1.0], [0.0, 1.0]])
    likelihood = np.eye(2)
    preferences = np.log(np.array([0.9, 0.1]))
    belief = np.array([0.5, 0.5])

    preferred_g = expected_free_energy(
        np.array([0]), transition, likelihood, preferences, belief
    )
    dispreferred_g = expected_free_energy(
        np.array([1]), transition, likelihood, preferences, belief
    )

    assert preferred_g < dispreferred_g


def test_policy_posterior_favors_lower_expected_free_energy_policy() -> None:
    transition = np.zeros((2, 2, 2))
    transition[:, :, 0] = np.array([[1.0, 0.0], [1.0, 0.0]])
    transition[:, :, 1] = np.array([[0.0, 1.0], [0.0, 1.0]])
    likelihood = np.eye(2)
    preferences = np.log(np.array([0.9, 0.1]))
    belief = np.array([0.5, 0.5])
    policies = [np.array([0]), np.array([1])]

    posterior = policy_posterior(
        policies, transition, likelihood, preferences, belief
    )

    assert np.isclose(posterior.sum(), 1.0)
    assert posterior[0] > posterior[1]
