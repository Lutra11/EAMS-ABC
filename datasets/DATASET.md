# Datasets

## Brandimarte MK Benchmarks

The `brandimarte/` directory contains 15 Flexible Job Shop Scheduling Problem (FJSP) instances (MK01–MK15) originally proposed by Brandimarte (1993).

| Instance | Jobs | Machines | Operations |
|----------|------|----------|------------|
| MK01 | 10 | 6 | 55 |
| MK02 | 10 | 6 | 58 |
| MK03 | 15 | 8 | 150 |
| MK04 | 15 | 8 | 90 |
| MK05 | 15 | 4 | 124 |
| MK06 | 15 | 8 | 150 |
| MK07 | 20 | 5 | 188 |
| MK08 | 20 | 10 | 225 |
| MK09 | 20 | 10 | 240 |
| MK10 | 20 | 15 | 240 |
| MK11 | 20 | 7 | 240 |
| MK12 | 20 | 7 | 240 |
| MK13 | 20 | 8 | 240 |
| MK14 | 20 | 8 | 240 |
| MK15 | 20 | 8 | 240 |

## Usage

- **MK01–MK10**: Used for main comparison and ablation experiments (30 independent runs per method per instance).
- **MK11–MK15**: Held-out validation set, used only after all algorithm development was finalized (30 independent runs per method per instance).
- **MK01, MK06, MK10**: Used for timing, sensitivity, and standby-ratio experiments (10 independent runs per setting).

## File Format

Each `.txt` file follows the standard Brandimarte format:
- Line 1: `n m` (number of jobs, number of machines)
- For each job: operation count, followed by (machine, processing time) pairs for each operation

## Power Parameters

Since the Brandimarte benchmarks do not include power observations, deterministic power parameters are constructed:

- **Independent scenario**: Processing power per machine index; standby power = 0.15 × processing power
- **Speed-dependent scenario**: Processing power inversely proportional to average processing speed; range [2, 6]

See `power_parameters.csv` for the exact values used in each scenario.

## Source

Processing times sourced from the [SchedulingLab](https://github.com/SchedulingLab/fjsp-instances) public repository (version ac4c340231). Power parameters constructed by the authors.

