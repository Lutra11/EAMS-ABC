# Code

## Structure

```
code/
├── core/           # EAMS-ABC algorithm implementation
├── baselines/      # Comparison methods (NSGA-II, MOEA/D, MA-NSGA-II, DABC)
└── experiments/    # Experiment runner, analysis, and figure generation
```

## Modules

### core/ — EAMS-ABC Algorithm

| Module | Responsibility |
|--------|---------------|
| `problem.py` | FJSP instance parsing, energy model (processing + standby) |
| `decoder.py` | Bi-level (OS+MS) encoding, forward insertion + reverse compaction, Solution dataclass |
| `initialization.py` | Four-strategy mixed population initialization |
| `criticality.py` | Joint criticality: time slack + processing energy + gap correlation |
| `operators.py` | Four neighborhood operators (N₁–N₄) + stagnation reconstruction |
| `variation.py` | Discrete POX crossover + uniform machine crossover |
| `pareto.py` | Non-dominated sorting, crowding distance, archive management, environmental selection |
| `solver.py` | Main ABC loop, Config dataclass with ablation switches |

### baselines/ — Comparison Methods

| Module | Method |
|--------|--------|
| `nsga2.py` | NSGA-II with tournament selection + crossover; MA-NSGA-II adds local search |
| `moead.py` | MOEA/D with Tchebycheff decomposition |
| `genetic.py` | Shared crossover operator (re-exports from `eams_abc.variation`) |
| `dabc.py` | DABC = Config(guided=False, adaptive=False, rebuild=False, gap=False, elite_exchange=False) |

All baselines share the same encoding, decoding, initialization, and archive framework — only the population update logic differs.

### experiments/ — Experiment Pipeline

| Module | Purpose |
|--------|---------|
| `run_study.py` | Parallel deterministic runner; each run = one JSON file |
| `metrics.py` | HV and IGD+ computation |
| `analyze.py` | Friedman test, Wilcoxon signed-rank, Holm correction |
| `check_correctness.py` | Schedule feasibility verification |
| `generate_figures.py` | All 12 publication figures |
