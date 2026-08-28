# SAC+SimHash: Count-Based Exploration for Sparse-Reward Continuous Control

*Soft Actor-Critic with locality-sensitive hash counting for exploration in continuous control tasks where rewards are sparse.*

**Status:** write-up complete (AAAI format), preprint in preparation for arXiv.

## Overview

Standard Soft Actor-Critic explores through policy entropy alone, which works well when the reward signal is dense but tends to plateau when an agent has to wander through long stretches with little or no feedback before finding a reward. This project adds a count-based exploration bonus on top of SAC: states are hashed into discrete buckets with SimHash, a locality-sensitive hashing scheme, and visit counts per bucket are used to compute a bonus that favors less-visited regions of the state space. This gives SAC a cheap approximation of "have I seen this before" in a continuous, high-dimensional state space, without training a separate density model.

The method is pretrained on the Ant-v4 MuJoCo benchmark, then fine-tuned on a custom choreography task, to test how well locomotion skills learned during pretraining transfer to a task that rewards a specific movement sequence rather than simple forward progress.

## Method

### Base algorithm: SAC
Soft Actor-Critic is an off-policy actor-critic method that optimizes a trade-off between expected return and policy entropy. Training follows the standard SAC setup: a stochastic actor, two critic networks, and a replay buffer.

### Exploration bonus: SimHash counting
Every visited state is hashed into a discrete binary code with SimHash, which maps similar continuous states to the same or nearby codes. The agent keeps a count per hash code and adds a bonus of roughly `1/sqrt(count)` to the reward at each step, encouraging the policy to seek out less-visited regions of state space instead of relying on entropy alone.

### Training pipeline
1. **Pretraining** on Ant-v4 (Gymnasium/MuJoCo) to learn general locomotion.
2. **Fine-tuning** on a custom choreography task with sparse reward tied to a specific movement sequence, starting from the pretrained policy.

## Key Features

- Count-based exploration bonus for continuous state spaces via SimHash, no learned density model required
- Built on SAC for stable, sample-efficient off-policy learning
- Two-stage training: general locomotion pretraining, then task-specific fine-tuning
- Targets sparse-reward continuous control settings where entropy-only exploration plateaus

## Results

_Add learning curves and final numbers here, for example: return or success rate on the choreography task with and without the SimHash bonus, and any ablations on hash dimension or bucket resolution._

## Installation

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

Core dependencies (update to match your actual `requirements.txt`):
- Python 3.10+
- PyTorch
- Gymnasium with the MuJoCo extra
- NumPy

## Usage

```bash
# Pretrain on Ant-v4
python train.py --env Ant-v4 --stage pretrain

# Fine-tune on the choreography task
python train.py --env Choreography-v0 --stage finetune --checkpoint path/to/pretrained.pt
```

_Replace with your actual script names and flags._

## Project Structure

```
.
├── sac/          # SAC actor and critic networks
├── simhash/      # LSH hashing and count-based bonus
├── envs/         # Custom choreography environment
├── configs/      # Training configs
├── train.py
└── requirements.txt
```

_Update to match your actual layout._

## Citation

If you build on this work, please cite the preprint once it is posted:

```bibtex
@article{talwadkar2026sacsimhash,
  title   = {[Paper title]},
  author  = {Talwadkar, Sarthak},
  year    = {2026},
  note    = {Preprint}
}
```

## License

[MIT is a common default for research code; update if you would prefer something else]

## Author

Sarthak Talwadkar
[GitHub](https://github.com/sarthak-talwadkar)
