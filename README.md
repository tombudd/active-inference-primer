# Active Inference Primer

> A minimal, annotated active inference primer for AI researchers and engineers who want to understand the mathematics before applying it.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Based on](https://img.shields.io/badge/Based%20on-Friston%202019-informational)](https://doi.org/10.1016/j.neuron.2019.09.037)

---

## Status

This repository is a minimal public primer.

Implemented:

- `README.md`
- `src/free_energy.py`
- `tests/test_free_energy.py`
- `requirements.txt`
- `PUBLIC_BOUNDARY.md`

Roadmap:

- generative model examples
- T-maze toy example
- notebooks
- fuller mathematical notes

The current code is intentionally small. It should be read as educational scaffolding, not as a complete active inference library.

---

## Why This Exists

Active inference is one of the most intellectually compelling frameworks in theoretical neuroscience and AI, but the notation can be dense and difficult to translate into code.

This repo starts with a narrow question:

> What is the smallest useful implementation of the free-energy pieces that a reader can inspect, run, and test?

The goal is understanding, not production deployment.

---

## What Is Included

`src/free_energy.py` implements:

- numerically stable log
- softmax
- KL divergence
- variational free energy
- a small educational expected-free-energy approximation
- policy posterior selection by softmax over negative expected free energy

The tests check basic numerical behavior and policy selection on simple synthetic inputs.

---

## Scope And Limitations

Limited: this repository implements a small educational subset inspired by Friston et al. and Da Costa et al.; it is not a full validation of active inference.

Expected free energy has multiple formulations and decompositions in the literature. The implementation here is a toy/spec approximation for learning and testing, not a claim that this is the only correct formulation.

This repository is not:

- a production agent
- a neuroscience model
- a complete active inference framework
- a formal proof of the free energy principle
- a validation of any private AI system
- an implementation of a private governance architecture

See [`PUBLIC_BOUNDARY.md`](PUBLIC_BOUNDARY.md).

---

## Quick Start

```bash
pip install -r requirements.txt
pytest
```

Run a tiny policy-posterior example:

```python
import numpy as np
from src.free_energy import policy_posterior

transition = np.array([
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 0.0], [0.0, 1.0]],
])
likelihood = np.eye(2)
preferences = np.log(np.array([0.9, 0.1]))
belief = np.array([0.5, 0.5])
policies = [np.array([0]), np.array([1])]

print(policy_posterior(policies, transition, likelihood, preferences, belief))
```

---

## Public Boundary

This repository is clean-room public education material. It uses small synthetic arrays and toy/spec examples only. It does not describe private systems, private implementation details, production logs, deployment receipts, or non-public architecture.

---

## References

- Friston, K. (2019). A free energy principle for a particular physics. *Neuron*, 104(1), 55-72.
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces. *Journal of Mathematical Psychology*.
- Parr, T., Pezzulo, G., & Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- Friston, K., et al. (2017). Active inference and epistemic value. *Cognitive Neuroscience*, 8(4).

---

## Related Work

This primer is related to the [Eudaimonic Alignment Framework](https://github.com/tombudd/eudaimonic-alignment), which uses active inference as one conceptual bridge for public AI governance research.

---

## License

MIT - use it, fork it, extend it. Attribution appreciated.

© 2025-2026 Tom Budd / ResoVerse Technologies
