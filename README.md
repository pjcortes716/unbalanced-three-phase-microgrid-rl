# Unbalanced Three-Phase Industrial Microgrid RL Benchmark

A Gymnasium environment coupled to OpenDSS for reinforcement learning energy management in an explicitly unbalanced three-phase industrial microgrid. This private collaboration snapshot contains the current `benchmark_v1` environment, nominal tests, derived development data, and SAC/Optuna infrastructure.

## Main characteristics

- 114-dimensional observation space; 12-dimensional continuous action space.
- 168 hourly steps per episode, with explicit ABC OpenDSS power flow.
- 3 PV plants, 1 BESS, 2 V2G trucks, 2 office EVs, and industrial/building loads.
- Economic energy cost, economic power factor (PF) penalty, and V2G departure-energy deficit penalty.

## Reward

```text
reward = - energy cost - power factor penalty - truck departure deficit penalty
c_unserved = 1 USD/kWh by default
lambda_pf = 1.0 by default
```

Both coefficients are configurable. Energy accounting uses USD and hourly intervals; export earns revenue at the existing purchase/sale tariff. There is no terminal BESS penalty. Preserve the coefficients and record seeds, scenarios, versions, and hashes in comparisons. The changelog's snapshot addendum identifies older statements about the default that no longer match the supplied source.

## Data splits

| Split | Office | Warehouse 1 | Warehouse 2 | Solar |
| --- | ---: | ---: | ---: | ---: |
| `train` | 69 | 55 | 85 | 68 |
| `optuna_validation` | 6 | 6 | 6 | 6 |
| `final_test` | withheld | withheld | withheld | withheld |

Counts are base profiles per source, not synchronized industrial weeks. Train draws each source independently. Validation uses six fixed reference compositions. `final_test` is withheld and not included in this repository snapshot; only its metadata remains. The partition is not strictly chronological.

## Installation

Use Python 3.11, from the repository root:

```bash
conda create -n microgrid_rl python=3.11 -y
conda activate microgrid_rl
pip install -r requirements.txt
```

For CPU-only use, it may be preferable to install PyTorch from its [official CPU index](https://download.pytorch.org/whl/cpu) before installing the remaining requirements:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Quick start

Run this Python example from the repository root. It completes one episode without training:

```python
import numpy as np
from env.Ambiente_industrial import MicrorredIndustrial

env = MicrorredIndustrial(
    split="train",
    c_unserved=1.0,
    lambda_pf=1.0,
    extreme_events=None,
)
try:
    obs, info = env.reset(seed=42)
    assert obs.shape == (114,) and env.action_space.shape == (12,)
    total_reward = 0.0
    for step in range(168):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (114,)
        assert info["electrical_converged"] and np.isfinite(reward)
        total_reward += reward
        if terminated or truncated:
            break
    assert step + 1 == 168 and terminated and not truncated
    print({"steps": step + 1, "return": total_reward})
finally:
    env.close()
```

Zero actions provide an executable example, not a service-optimal controller.

## Testing

```bash
python -B scripts/prepare_benchmark_v1_data.py --verify-only
python -B scripts/run_benchmark_v1_checks.py
```

The nominal runner writes a small JSON summary and log, both ignored by Git. It runs no training or Optuna. Deferred extreme-event tests are absent. Two provenance checks validate distributed derivatives; full reconstruction against the withheld raw sources is outside this snapshot's test coverage.

## Reproducibility and collaboration

Read [BENCHMARK_V1_CHANGELOG.md](BENCHMARK_V1_CHANGELOG.md), [scenario_splits.json](data/scenario_splits.json), and [DATA_PROVENANCE.md](DATA_PROVENANCE.md). The changelog takes precedence over the historical [audit](AUDITORIA_AMBIENTE_TRIFASICO.md). Keep methods, assumptions, limitations, and article changes reviewable through small commits and pull requests.

[The campaign protocol](optuna/OPTUNA_VM_CAMPAIGN.md) fixes 60 trials: 12 warm-up, 48 TPE, at most 100800 steps/trial, separate processes, and validation-only selection. No campaign was launched in preparing this snapshot. Its machine-name placeholder must be configured before a future VM launch.

## Current scope

`final_test` remains retained; extreme events are outside the initial experiments. Conformal prediction and safety projection are outside this repository/paper. Forecasts are synthetic noisy forecasts constructed from the realized future, not trained predictors. Cross-platform bitwise equivalence and learning quality are not established by the nominal tests.

## License

The original software in this repository is made available under the PolyForm Noncommercial License 1.0.0. Commercial use is not permitted under the default license. For commercial licensing inquiries, please contact the copyright holders.

The software is **source-available for non-commercial research and educational use**. The complete [PolyForm Noncommercial License 1.0.0](LICENSE) governs all original software code in this repository, including scripts, tests, and original code examples in documentation. The license text is reproduced unchanged from the [official PolyForm source](https://raw.githubusercontent.com/polyformproject/polyform-licenses/1.0.0/PolyForm-Noncommercial-1.0.0.md).

Required Notice: Copyright (c) 2026 Unbalanced Three-Phase Industrial Microgrid RL Benchmark contributors

Original documentation prose is separately licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en). This grant applies only to original documentation, not to software code, datasets, third-party material, quoted license texts, or upstream notices.

Third-party and derived datasets remain subject to the terms, licenses, and attribution requirements of their respective original sources. Neither the software license nor the documentation license automatically applies to those datasets. Third-party software and notices retain their respective licenses. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for attribution, differing local/upstream notices, and unresolved solar provenance before public release.
