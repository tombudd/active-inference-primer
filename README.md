# Active Inference Primer

> A minimal, annotated reference implementation of the active inference framework (Friston et al.) — written for AI researchers and engineers who want to understand the mathematics before applying it.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Based on](https://img.shields.io/badge/Based%20on-Friston%202019-informational)](https://doi.org/10.1016/j.neuron.2019.09.037)

---

## Why This Exists

Active inference is one of the most intellectually compelling frameworks in theoretical neuroscience and AI — and one of the hardest to actually implement from the papers alone. The notation is dense, the conceptual dependencies are non-obvious, and most code that exists is either too abstract or too tied to specific neuroimaging use cases.

This repo is the implementation I wish had existed when I started. It is:
- **Minimal**: only the core mathematics, no domain-specific baggage
- **Annotated**: every equation has a plain-English explanation alongside it
- **Correct**: validated against Friston et al. (2019) and Da Costa et al. (2020)
- **Extensible**: designed to be the foundation for your own experiments

---

## What Is Active Inference?

Active inference is a unified theory of perception, learning, and action grounded in the **free energy principle**: the claim that all self-organising systems (biological or artificial) act to minimise their variational free energy — a bound on surprise.

In active inference, an agent doesn't have a reward function. It has a **generative model** — a probabilistic model of how its observations are caused — and it acts to minimise the divergence between its model and reality.

This has several striking implications:
- **Perception is inference**: the brain constructs its experience by minimising prediction error
- **Action is inference**: actions are chosen to make predictions come true
- **Learning is model update**: beliefs are updated to reduce free energy over time
- **Curiosity is intrinsic**: agents prefer states that reduce uncertainty about their models

---

## Repository Structure

```
active-inference-primer/
├── notebooks/
│   ├── 01_free_energy_basics.ipynb      # The mathematics of variational free energy
│   ├── 02_generative_model.ipynb        # Building a generative model from scratch
│   ├── 03_perception_as_inference.ipynb # Belief updating via gradient descent on F
│   ├── 04_action_as_inference.ipynb     # Policy selection via expected free energy
│   ├── 05_learning.ipynb                # Parameter learning and model update
│   └── 06_full_agent.ipynb              # Complete active inference agent
├── src/
│   ├── free_energy.py     # Core F computation
│   ├── generative.py      # Generative model classes
│   ├── inference.py       # Belief propagation algorithms
│   ├── agent.py           # Full agent implementation
│   └── utils.py           # Softmax, normalisation, etc.
├── examples/
│   ├── t_maze.py          # Classic T-maze experiment
│   ├── mountain_car.py    # Active inference on a control task
│   └── epistemic_foraging.py  # Information-seeking behaviour
├── tests/
└── docs/
    ├── math_reference.md  # Full mathematical reference
    └── glossary.md        # Key terms defined precisely
```

---

## The Core Mathematics

### Variational Free Energy

For an agent with observations `o` and hidden states `x`, the variational free energy is:

```
F = E_q[log q(x)] - E_q[log p(o, x)]
  = KL[q(x) || p(x)] - E_q[log p(o|x)]
  = Complexity - Accuracy
```

Where:
- `q(x)` is the agent's approximate posterior belief about hidden states
- `p(o, x)` is the generative model (joint distribution over observations and states)
- Minimising F = maximising model evidence while minimising complexity

### Expected Free Energy (Policy Selection)

For choosing actions, the agent evaluates the **expected free energy** of each policy `π`:

```
G(π) = E_q[log q(x|π) - log p(o,x|π)]
     ≈ -E_q[log p(o|π)] + KL[q(x|π) || p(x)]
     = Pragmatic value + Epistemic value
```

Policies are selected by softmax over `-G(π)` — preferring policies that are both useful (pragmatic) and informative (epistemic).

---

## Quick Start

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_free_energy_basics.ipynb
```

Or run the T-maze example directly:

```bash
python examples/t_maze.py
```

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
jupyter>=1.0
```

No deep learning frameworks required — this is pure probabilistic computation.

---

## References

- Friston, K. (2019). A free energy principle for a particular physics. *Neuron*, 104(1), 55-72.
- Da Costa, L., et al. (2020). Active inference on discrete state-spaces. *Journal of Mathematical Psychology*.
- Parr, T., Pezzulo, G., & Friston, K. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.
- Friston, K., et al. (2017). Active inference and epistemic value. *Cognitive Neuroscience*, 8(4).

---

## Related Work

This primer was developed alongside the [Eudaimonic Alignment Framework](https://github.com/tombudd/eudaimonic-alignment), which applies active inference to AI governance by treating eudaimonic flourishing as the generative model's prior.

---

## License

MIT — use it, fork it, extend it. Attribution appreciated.

© 2025–2026 Tom Budd / ResoVerse Technologies
