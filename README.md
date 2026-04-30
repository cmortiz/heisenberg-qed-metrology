# Quantum Sensing with Error Detection: Numerical Simulations

GPU-accelerated Bayesian phase estimation simulations for
"Encoded Quantum Signal Processing for Heisenberg-Limited Metrology."

## Setup

Requires Python 3.10-3.12 and a CUDA-capable GPU (recommended). Install
dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

CPU-only execution works but is significantly slower. The simulations
auto-detect GPU availability and fall back to CPU if needed.

## Simulation Scripts

Three simulation scripts correspond to the three protocols in the paper:

| Script | Protocol | Paper Reference |
|--------|----------|-----------------|
| `scripts/simulation.py` | Bare GHZ | Algorithm 1 (Section IV) |
| `scripts/simulation_error_detected.py` | Error-detected (bit-flip code) | Algorithm 2 (Section VI) |
| `scripts/simulation_combined.py` | Combined (logical GHZ + repetition code) | Algorithm 4 (Section IX) |

### Bare GHZ

```bash
# Usage: simulation.py SEED [GAMMA] [H]
uv run python scripts/simulation.py 42              # noiseless
uv run python scripts/simulation.py 42 0.05         # gamma=5%, homogeneous
uv run python scripts/simulation.py 42 0.05 0.3     # gamma=5%, h=30% heterogeneity
```

### Error-Detected (Bit-Flip Code)

```bash
# Usage: simulation_error_detected.py SEED GAMMA [H] [--L N] [--full-likelihood]
uv run python scripts/simulation_error_detected.py 42 0.05 --L 1        # post-selection, L=1 (N=3)
uv run python scripts/simulation_error_detected.py 42 0.05 --L 2        # post-selection, L=2 (N=5)
uv run python scripts/simulation_error_detected.py 42 0.05 --L 1 --full-likelihood  # full-likelihood
```

### Combined (Logical GHZ + Repetition Code)

```bash
# Usage: simulation_combined.py SEED GAMMA SIGMA_EPSILON [H] [--L N] [--full-likelihood]
uv run python scripts/simulation_combined.py 42 0.05 0.01 --L 1         # post-selection
uv run python scripts/simulation_combined.py 42 0.05 0.01 --L 1 --full-likelihood   # full-likelihood
```

### Inference Modes

Each ED and combined script supports two modes:

- **Post-selection** (default): discard rounds with detected errors. Simple but
  wastes measurement data.
- **Full-likelihood** (`--full-likelihood`): use all rounds with syndrome-adjusted
  likelihoods. More data-efficient; converges in fewer rounds.

### Output

Each simulation run appends rows to a CSV in `scripts/results/`, one row per epsilon
value. Columns include seed, epsilon, gamma, total resources (qubit-time units),
convergence status, and the estimated scaling exponent.

## Reproducing Paper Results

The parameter grids used in the paper:

| Protocol | L | gamma | sigma_eps | Seeds |
|----------|---|-------|-----------|-------|
| Bare GHZ | n/a | 0, 0.01, 0.05, 0.10, 0.15, 0.20 | n/a | 0-39 |
| Bare GHZ (heterogeneous) | n/a | 0.05 (h=0.3), 0.05 (h=0.5), 0.10 (h=0.3) | n/a | 0-39 |
| ED | 1, 2, 3 | 0.01, 0.05, 0.10 | n/a | 0-39 |
| Combined | 1, 2 | 0.01, 0.05, 0.10 | 0.01, 0.05 | 0-39 |

To reproduce, run each (seed, config) combination. For example, all 40 seeds of ED
with L=1, gamma=5%:

```bash
for seed in $(seq 0 39); do
    uv run python scripts/simulation_error_detected.py $seed 0.05 --L 1 &
done
wait
```

Adjust concurrency to match your hardware. On a single GPU, 4-8 concurrent processes
typically achieve good utilization. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` in the
environment to allow JAX to allocate GPU memory on demand:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Full-likelihood mode uses more memory per process; reduce concurrency if you encounter
OOM errors. To pin processes to a specific GPU, set `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/simulation_error_detected.py 42 0.05 --L 1
```

The full suite (bare GHZ + ED + combined, both modes, 40 seeds) totals ~1,680
individual runs. On a modern GPU this takes several hours.

## Analysis and Plotting

After generating (or using the pre-computed) results:

```bash
# Compute scaling exponents for bare GHZ
cd scripts && uv run python analyze_results.py --csv results/bare_ghz.csv

# Unified analysis across all protocols
uv run python analyze_combined.py

# Generate publication figures (output to plots/)
uv run python plot_redesigned.py
```

## Pre-computed Results

The `scripts/results/` directory contains our 40-seed production data:

| File | Rows | Protocol |
|------|------|----------|
| `bare_ghz.csv` | 39,600 | Bare GHZ + product (18 configs x 40 seeds) |
| `ed_postselect.csv` | 21,600 | ED post-selection (9 configs x 40 seeds x 60 epsilon) |
| `ed_full_likelihood.csv` | 21,600 | ED full-likelihood |
| `combined_postselect.csv` | 28,800 | Combined post-selection (12 configs x 40 seeds x 60 epsilon) |
| `combined_full_likelihood.csv` | 28,800 | Combined full-likelihood |

## Key Constants

| Constant | Value | Notes |
|----------|-------|-------|
| `CHECK_FREQUENCY` | 100 | Convergence check interval (post-selection mode) |
| `CHECK_FREQUENCY_FL` | 10 | Convergence check interval (full-likelihood mode) |
| `CONVERGENCE_TOLERANCE` | 1.2 | Converged when posterior width < 1.2 * epsilon |
| `MAX_EXPERIMENTS` | 10,000 / 50,000 | Bare GHZ / ED and combined protocols |

## Implementation Notes

### Adaptive Grid Resolution

The Bayesian posterior grid size is chosen per-protocol to satisfy the Nyquist
constraint: `N_eff * M_max < grid_size / 4`. Bare GHZ uses a fixed 2^16 grid;
ED and combined scripts compute the grid adaptively via `compute_grid_resolution()`,
giving up to 2^20 for combined L=2 (N_total=15).

### Wall-Time vs. Computational Cost

The combined protocol embeds the ED protocol (3 blocks vs 1), so each Bayesian
update is more expensive (larger grid, more qubits). However, combined full-likelihood
runs often complete *faster* in wall time. This is because:

1. **Convergence speed dominates.** Combined L=1 uses 9 qubits, giving ~81 Fisher
   information per round vs ED L=1's ~9. In full-likelihood mode every round
   contributes, so combined converges in roughly 1/9th the Bayesian updates.

2. **GPU insensitivity to grid size.** Both 2^17 and 2^19 grids are small for modern
   GPUs. Per-round GPU time is nearly identical; the bottleneck is Python loop
   overhead, which is the same for both.

The net effect: fewer rounds at similar wall time per round makes combined faster
despite strictly more computation per round. Memory usage also reflects this: ED
accumulates more cache entries (more rounds, more distinct M_evol values).

### Numerical Stability

- Log-space Bayesian updates throughout to avoid underflow
- Integer modular arithmetic for `cos(2n*omega)` on discrete grids to avoid float32
  precision loss at large n
- `sin`/`cos` formulation instead of `tan` to avoid overflow near poles
- Adaptive grid sizing to prevent Nyquist aliasing at high effective frequencies
