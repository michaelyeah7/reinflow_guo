# Algorithmic Changes to ReinFlow

This document describes the **algorithmic modifications** made to the original ReinFlow codebase, specifically changes to the stochastic differential equation (SDE) formulation and noise schedule in the flow matching policy optimization.

---

## Overview

The primary algorithmic change is the introduction of a **tunable noise schedule parameter** `noise_level_a` that controls the noise injection in the flow matching SDE. This modification enables systematic exploration of different noise schedules and their impact on policy learning.

---

## 1. Core Algorithmic Change: Noise Schedule Parameterization

### 1.1 Modified File
**File**: `model/flow/ft_ppo/ppoflow.py`

### 1.2 Key Change: Introduction of `noise_level_a`

#### Original Implementation
The original ReinFlow implementation used a fixed or learned noise schedule for the SDE. The noise level `λ_t` was determined by the exploration network or fixed schedulers (e.g., 'vp', 'lin', 'const', 'learn_decay').

#### Modified Implementation
We introduced a **tunable parameter** `noise_level_a` that directly controls the noise schedule according to the FMPO (Flow Matching Policy Optimization) framework:

```python
λ_t = a * (1 - t)
```

where:
- `a` = `noise_level_a` (tunable hyperparameter, default: 0.6)
- `t` = flow matching time in [0, 1)
- `λ_t` = noise level at time `t`

### 1.3 Implementation Details

#### Parameter Addition
**Location**: `PPOFlow.__init__()` method (line 71)
```python
def __init__(self, 
             # ... other parameters ...
             noise_level_a=0.6  # NEW: Noise level coefficient
             ):
    # ...
    # Noise level coefficient: λ_t = a * (1 - t) per FMPO framework
    self.noise_level_a = noise_level_a
```

#### Usage in SDE Formulation

The noise schedule is used in two key methods:

**1. `get_logprobs()` method** (lines 267-284):
```python
# Compute λ_t = a * (1 - t) per FMPO framework (Table 1)
one_minus_t = torch.clamp(1 - t_bc, min=1e-6)
lambda_t = self.noise_level_a * one_minus_t

# Compute drift coefficients per Theorem 4.1
coef_vel = 1.0 + (lambda_t / 2.0) * (t_bc / one_minus_t)
coef_xt = (lambda_t / 2.0) / one_minus_t

# Drift: scaled velocity + contraction term
drift = coef_vel * vt_flat - coef_xt * xt_flat

# Transition std = √(λ_t * dt)
trans_std = torch.sqrt(lambda_t * dt)
```

**2. `get_actions()` method** (lines 394-413):
```python
# Compute λ_t = a * (1 - t) per FMPO framework (Table 1)
one_minus_t = torch.clamp(1 - t_bc, min=1e-6)
lambda_t = self.noise_level_a * one_minus_t

# Compute drift coefficients per Theorem 4.1:
# dZ = [(1 + λ/2 * t/(1-t)) * v - (λ/2 * 1/(1-t)) * Z] dt + √λ dW
coef_vel = 1.0 + (lambda_t / 2.0) * (t_bc / one_minus_t)
coef_xt = (lambda_t / 2.0) / one_minus_t

# Compute drift: scaled velocity + contraction term
drift = coef_vel * vt - coef_xt * xt

# Diffusion std = √(λ_t * dt)
diffusion_std = torch.sqrt(lambda_t * dt)
```

### 1.4 SDE Formulation

The modified SDE follows **Theorem 4.1** from the FMPO framework:

```
dZ_t = [(1 + λ_t/2 * t/(1-t)) * u_θ(Z_t, t) - (λ_t/2 * 1/(1-t)) * Z_t] dt + √λ_t dW_t
```

where:
- `λ_t = noise_level_a * (1 - t)` (NEW: tunable schedule)
- `u_θ(Z_t, t)` = velocity predicted by the policy network
- `dW_t` = Wiener process

In discrete form (Euler-Maruyama):
```
Z_{t+dt} ~ N(Z_t + drift * dt, λ_t * dt)
```

where:
- `drift = (1 + λ/2 * t/(1-t)) * v - (λ/2 * 1/(1-t)) * Z`
- `v` = velocity from policy network

### 1.5 Impact on Algorithm Behavior

**Noise Schedule Properties**:
- **At t=0**: `λ_0 = a * 1 = a` (maximum noise)
- **At t→1**: `λ_1 → a * 0 = 0` (no noise, deterministic)
- **Decay rate**: Linear decay from `a` to `0`

**Effect of `noise_level_a`**:
- **Higher values (e.g., 0.9)**: More exploration, higher variance in early denoising steps
- **Lower values (e.g., 0.1)**: Less exploration, more deterministic behavior
- **Default (0.6)**: Balanced exploration-exploitation trade-off

---

## 2. Supporting Infrastructure Changes

To enable systematic exploration of the `noise_level_a` parameter, we added supporting infrastructure:

### 2.1 Configuration File Updates

Added `noise_level_a` parameter to config files:
- `cfg/gym/finetune/Humanoid-v3/ft_ppo_reflow_mlp.yaml`
- `cfg/gym/finetune/hopper-v2/ft_ppo_reflow_mlp.yaml`
- `cfg/gym/finetune/walker2d-v2/ft_ppo_reflow_mlp.yaml`
- `cfg/gym/finetune/ant-v2/ft_ppo_reflow_mlp.yaml`

**Location**: Under `model:` section
```yaml
model:
  # ... other model parameters ...
  noise_level_a: 0.6  # Tunable noise schedule coefficient
```

### 2.2 Parameter Search Scripts

Created scripts to systematically search the `noise_level_a` parameter space:

- **`script/param_search_noise_level.py`**: Python script for noise level search
- **`script/param_search_noise_level.sh`**: Bash script for noise level search

**Features**:
- Searches `noise_level_a` from 0.1 to 0.9 with step 0.1
- Creates organized log folders: `{ENV}_noise_{LEVEL}_ppo_reflow_mlp_...`
- Creates organized wandb folders: `wandb_offline_{ENV}_noise_{LEVEL}`

### 2.3 Seed Search Scripts

Created scripts for multi-seed experiments to evaluate robustness:
- `script/param_search_seed_antv2.sh`
- `script/param_search_seed_hopperv2.sh`
- `script/param_search_seed_humanoidv3.sh`
- `script/param_search_seed_walker2dv2.sh`

### 2.4 Wandb Directory Support

Added `wandb.dir` parameter to config files to support custom wandb directory paths for better experiment organization.

---

## 3. Mathematical Formulation

### 3.1 Original SDE (Reference)
The original ReinFlow used various noise schedulers, but the SDE form was:
```
dZ_t = [velocity_term] dt + [noise_term] dW_t
```

where the noise term was determined by exploration networks or fixed schedules.

### 3.2 Modified SDE (Current)
The modified SDE explicitly parameterizes the noise schedule:
```
dZ_t = [(1 + λ_t/2 * t/(1-t)) * u_θ(Z_t, t) - (λ_t/2 * 1/(1-t)) * Z_t] dt + √λ_t dW_t
```

with:
```
λ_t = noise_level_a * (1 - t)
```

### 3.3 Log Probability Calculation

The log probability of a trajectory follows the Markov chain:
```
log p(x_0, x_1, ..., x_K) = log p(x_0) + Σ_{i=0}^{K-1} log p(x_{i+1} | x_i)
```

where each transition is:
```
x_{i+1} | x_i ~ N(mean_i, λ_{t_i} * dt)
```

with:
```
mean_i = x_i + [(1 + λ/2 * t/(1-t)) * v_i - (λ/2 * 1/(1-t)) * x_i] * dt
```

---

## 4. Comparison with Original Implementation

| Aspect | Original ReinFlow | Modified Version |
|--------|------------------|------------------|
| **Noise Schedule** | Fixed or learned via exploration network | **Tunable**: `λ_t = a * (1 - t)` |
| **Parameter** | Implicit in exploration network | **Explicit**: `noise_level_a` |
| **Tunability** | Limited, requires network retraining | **Direct**: Single hyperparameter |
| **Interpretability** | Noise schedule learned end-to-end | **Clear**: Linear decay from `a` to `0` |
| **Search Space** | Continuous, high-dimensional | **1D**: Single scalar parameter |

---

## 5. Experimental Setup

### 5.1 Parameter Search Protocol

To systematically explore the `noise_level_a` parameter:

1. **Range**: 0.1 to 0.9 with step 0.1 (9 values)
2. **Environments**: Humanoid-v3, hopper-v2, walker2d-v2
3. **Metrics**: Episode reward, success rate, training stability
4. **Logging**: Separate folders for each noise level

### 5.2 Default Values by Environment

| Environment | Default `noise_level_a` | Rationale |
|------------|------------------------|-----------|
| Humanoid-v3 | 0.6 | Balanced exploration for complex dynamics |
| hopper-v2 | 0.5 | Moderate exploration for simpler task |
| walker2d-v2 | 0.6 | Similar to Humanoid-v3 |
| ant-v2 | 0.4 | Lower exploration for stable learning |

---

## 6. Usage

### 6.1 Override via Command Line

```bash
python script/run.py \
  --config-dir=cfg/gym/finetune/Humanoid-v3 \
  --config-name=ft_ppo_reflow_mlp \
  model.noise_level_a=0.7
```

### 6.2 Run Parameter Search

```bash
# Python script
python script/param_search_noise_level.py

# Bash script
./script/param_search_noise_level.sh
```

---

## 7. Theoretical Justification

The noise schedule `λ_t = a * (1 - t)` follows the **FMPO (Flow Matching Policy Optimization)** framework:

1. **Early steps (t→0)**: High noise (`λ ≈ a`) promotes exploration
2. **Late steps (t→1)**: Low noise (`λ → 0`) enables exploitation
3. **Linear decay**: Simple, interpretable schedule that balances exploration-exploitation

This formulation ensures:
- **Exploration** in early denoising steps when uncertainty is high
- **Exploitation** in late denoising steps when policy confidence increases
- **Tunable trade-off** via single hyperparameter `a`

---

## 8. Files Modified

### Core Algorithm Changes
- `model/flow/ft_ppo/ppoflow.py` - Added `noise_level_a` parameter and modified SDE formulation

### Supporting Infrastructure
- `cfg/gym/finetune/*/ft_ppo_reflow_mlp.yaml` - Added `noise_level_a` and `wandb.dir` parameters
- `script/param_search_noise_level.py` - Parameter search script
- `script/param_search_noise_level.sh` - Parameter search script (bash)
- `script/param_search_seed_*.sh` - Seed search scripts for multiple environments

---

## 9. Future Work

Potential extensions:
- [ ] Adaptive `noise_level_a` scheduling during training
- [ ] Environment-specific automatic tuning
- [ ] Multi-objective optimization over noise schedule
- [ ] Theoretical analysis of optimal `noise_level_a` values

---

## 10. References

- **FMPO Framework**: Flow Matching Policy Optimization (see code comments for Theorem 4.1)
- **Original ReinFlow**: Fine-tuning Flow Matching Policy with Online Reinforcement Learning
- **SDE Formulation**: Based on Theorem 4.1 in the FMPO framework

---

## Contact

For questions about these algorithmic changes, please refer to the code comments in `model/flow/ft_ppo/ppoflow.py` or create an issue in the repository.
