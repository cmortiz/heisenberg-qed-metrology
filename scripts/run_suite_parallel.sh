#!/bin/bash
# run_suite_parallel.sh - Parallel orchestrator for the §9 numerical suite.
#
# Identical flag interface and configurations as run_suite.sh, but instead of
# running cells sequentially, generates a job manifest and runs it via GNU
# parallel with:
#   - GPU jobs (ED, combined, bare-Ham, bare-GHZ): 32 concurrent workers
#     (16 per GPU x 2 GPUs), each capped at XLA_PYTHON_CLIENT_MEM_FRACTION=0.06
#     (~6 GB per process; 16 x 6 = 96 GB per GPU, fits a 97 GB card).
#   - CPU jobs (sequential.py): 64 concurrent workers, pure numpy.
#
# Concurrency knobs (override via environment):
#   GPU_JOBS=32  (parallel slots for GPU work)
#   CPU_JOBS=64  (parallel slots for sequential)
#   GPU_MEM_FRAC=0.06
#   N_GPUS=2     (round-robin across CUDA_VISIBLE_DEVICES=0,1,...)
#
# CSV write contention: each script's main() writes the (seed, config) cell's
# 60-row DataFrame ONCE at the end of the cell, under fcntl.LOCK_EX. With 32
# workers and ~2 min per cell, average write rate is ~0.27 writes/sec; lock
# hold time is sub-second. No contention bottleneck.
#
# Logs:
#   scripts/logs/{label}.log     - per-cell stdout+stderr
#   scripts/logs/joblog_gpu.tsv  - parallel's exec log (exit code, runtime)
#   scripts/logs/joblog_cpu.tsv  - same for CPU group
#   scripts/logs/manifest_gpu.tsv, manifest_cpu.tsv - emitted job lists
#
# Usage:
#   bash run_suite_parallel.sh                       # full suite
#   bash run_suite_parallel.sh --seeds N             # override seed count
#   bash run_suite_parallel.sh --quick               # 4 seeds, 1 config per group (smoke)
#   bash run_suite_parallel.sh --skip-bare/--skip-ed/--skip-combined/--skip-bare-ham
#   bash run_suite_parallel.sh --skip-ablation/--skip-sequential/--skip-marg/--skip-sigeps
#   bash run_suite_parallel.sh --dry-run             # emit manifest, do not run
#
# Detached mode (recommended for full runs):
#   tmux new -s suite -d 'bash scripts/run_suite_parallel.sh 2>&1 | tee scripts/logs/orchestrator.log'
#   tmux attach -t suite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUMERICS_DIR="$(dirname "$SCRIPT_DIR")"
cd "$NUMERICS_DIR"

# Defaults
SEEDS=40
QUICK=0
DRY_RUN=0
RUN_BARE=1
RUN_ED=1
RUN_COMBINED=1
RUN_BARE_HAM=1
RUN_ABLATION=1
RUN_SEQUENTIAL=1
RUN_MARG=1
RUN_SIGEPS=1

GPU_JOBS=${GPU_JOBS:-16}
CPU_JOBS=${CPU_JOBS:-64}
GPU_MEM_FRAC=${GPU_MEM_FRAC:-0.10}
N_GPUS=${N_GPUS:-2}
# Shared JAX compilation cache so workers share JIT-compiled kernels and avoid
# autotune contention. Must be on a fast filesystem; ~/.jax_cache typical.
JAX_CACHE_DIR=${JAX_CACHE_DIR:-$NUMERICS_DIR/.jax_cache}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;
    --quick) QUICK=1; SEEDS=4; shift;;
    --dry-run) DRY_RUN=1; shift;;
    --skip-bare) RUN_BARE=0; shift;;
    --skip-ed) RUN_ED=0; shift;;
    --skip-combined) RUN_COMBINED=0; shift;;
    --skip-bare-ham) RUN_BARE_HAM=0; shift;;
    --skip-ablation) RUN_ABLATION=0; shift;;
    --skip-sequential) RUN_SEQUENTIAL=0; shift;;
    --skip-marg) RUN_MARG=0; shift;;
    --skip-sigeps) RUN_SIGEPS=0; shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
GPU_MANIFEST="$LOG_DIR/manifest_gpu.tsv"
CPU_MANIFEST="$LOG_DIR/manifest_cpu.tsv"
GPU_JOBLOG="$LOG_DIR/joblog_gpu.tsv"
CPU_JOBLOG="$LOG_DIR/joblog_cpu.tsv"
> "$GPU_MANIFEST"
> "$CPU_MANIFEST"

# Manifest format: <label>\t<bash command string>
# Each line is consumed by `parallel ... bash -c {2} > logs/{1}.log 2>&1`
# so the {2} substitution is a single shell-quoted arg that bash -c re-parses.
emit_gpu() {
  local label="$1"; shift
  printf '%s\t%s\n' "$label" "uv run python $*" >> "$GPU_MANIFEST"
}
emit_cpu() {
  local label="$1"; shift
  printf '%s\t%s\n' "$label" "uv run python $*" >> "$CPU_MANIFEST"
}

echo "============================================================"
echo "EQSP metrology numerical suite (parallel orchestrator)"
echo "  Seeds:               $SEEDS"
echo "  Quick:               $QUICK"
echo "  Dry-run:             $DRY_RUN"
echo "  GPU jobs:            $GPU_JOBS  (across $N_GPUS GPUs, mem_frac $GPU_MEM_FRAC)"
echo "  CPU jobs:            $CPU_JOBS  (sequential.py only)"
echo "  Bare GHZ depol:      $RUN_BARE"
echo "  Error-det Ham:       $RUN_ED"
echo "  Combined Ham:        $RUN_COMBINED"
echo "  Bare GHZ Ham:        $RUN_BARE_HAM"
echo "  ED/comb ablation:    $RUN_ABLATION"
echo "  Sequential proto:    $RUN_SEQUENTIAL"
echo "  Quenched-vs-marg:    $RUN_MARG"
echo "  σ_ε breakdown scan:  $RUN_SIGEPS"
echo "  Working dir:         $NUMERICS_DIR"
echo "  Log dir:             $LOG_DIR"
echo "============================================================"
echo ""

# -- Bare GHZ (Table I, depolarizing channel) --------------------------------
if [[ $RUN_BARE -eq 1 ]]; then
  declare -a BARE_CONFIGS=(
    "noiseless 0.0 0.0"
    "g01 0.01 0.0"
    "g05 0.05 0.0"
    "g05_h30 0.05 0.3"
    "g10 0.10 0.0"
    "g15 0.15 0.0"
    "g20 0.20 0.0"
  )
  [[ $QUICK -eq 1 ]] && BARE_CONFIGS=("g05 0.05 0.0")
  for cfg in "${BARE_CONFIGS[@]}"; do
    read -r label gamma h <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "bare-ghz_${label}_seed${seed}" \
        "scripts/simulation.py $seed $gamma $h --mode ghz"
    done
  done
fi

# -- Error-detected (Table III) ----------------------------------------------
if [[ $RUN_ED -eq 1 ]]; then
  declare -a ED_CONFIGS=(
    "L1_g01 1 0.01" "L1_g05 1 0.05" "L1_g10 1 0.10"
    "L2_g01 2 0.01" "L2_g05 2 0.05" "L2_g10 2 0.10"
    "L3_g01 3 0.01" "L3_g05 3 0.05" "L3_g10 3 0.10"
  )
  [[ $QUICK -eq 1 ]] && ED_CONFIGS=("L1_g05 1 0.05")
  for cfg in "${ED_CONFIGS[@]}"; do
    read -r label L gamma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "ed-ps_${label}_seed${seed}" \
        "scripts/simulation_error_detected.py $seed $gamma 0.0 --L $L"
      emit_gpu "ed-fl_${label}_seed${seed}" \
        "scripts/simulation_error_detected.py $seed $gamma 0.0 --L $L --full-likelihood"
    done
  done
fi

# -- Combined (Table V) ------------------------------------------------------
if [[ $RUN_COMBINED -eq 1 ]]; then
  declare -a COMB_CONFIGS=(
    "L1_g01_s01 1 0.01 0.01" "L1_g05_s01 1 0.05 0.01" "L1_g10_s01 1 0.10 0.01"
    "L1_g01_s05 1 0.01 0.05" "L1_g05_s05 1 0.05 0.05" "L1_g10_s05 1 0.10 0.05"
    "L2_g01_s01 2 0.01 0.01" "L2_g05_s01 2 0.05 0.01" "L2_g10_s01 2 0.10 0.01"
    "L2_g01_s05 2 0.01 0.05" "L2_g05_s05 2 0.05 0.05" "L2_g10_s05 2 0.10 0.05"
  )
  [[ $QUICK -eq 1 ]] && COMB_CONFIGS=("L1_g05_s01 1 0.05 0.01")
  for cfg in "${COMB_CONFIGS[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "comb-ps_${label}_seed${seed}" \
        "scripts/simulation_combined.py $seed $gamma $sigma 0.0 --L $L"
      emit_gpu "comb-fl_${label}_seed${seed}" \
        "scripts/simulation_combined.py $seed $gamma $sigma 0.0 --L $L --full-likelihood"
    done
  done
fi

# -- Bare GHZ + Hamiltonian (Table IV apples-to-apples + §9 pilot) -----------
if [[ $RUN_BARE_HAM -eq 1 ]]; then
  declare -a BARE_HAM_MATRIX=(
    "L1_g01 1 0.01" "L1_g05 1 0.05" "L1_g10 1 0.10"
    "L2_g01 2 0.01" "L2_g05 2 0.05" "L2_g10 2 0.10"
    "L3_g01 3 0.01" "L3_g05 3 0.05" "L3_g10 3 0.10"
  )
  declare -a BARE_HAM_PILOT=("L2_g030 2 0.30" "L2_g045 2 0.45")
  if [[ $QUICK -eq 1 ]]; then
    BARE_HAM_MATRIX=("L1_g05 1 0.05")
    BARE_HAM_PILOT=()
  fi
  for cfg in "${BARE_HAM_MATRIX[@]}" ${BARE_HAM_PILOT[@]+"${BARE_HAM_PILOT[@]}"}; do
    read -r label L gamma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "bare-ham_${label}_seed${seed}" \
        "scripts/simulation_bare_hamiltonian.py $seed $gamma 0.0 --L $L"
    done
  done
fi

# -- B1: A3/A4 ablation (cond-only ED-FL + combined-FL) ----------------------
if [[ $RUN_ABLATION -eq 1 ]]; then
  declare -a ED_ABL=("L1_g01 1 0.01" "L1_g10 1 0.10" "L3_g01 3 0.01" "L3_g10 3 0.10")
  declare -a COMB_ABL=(
    "L1_g01_s01 1 0.01 0.01" "L1_g10_s01 1 0.10 0.01"
    "L2_g01_s01 2 0.01 0.01" "L2_g10_s01 2 0.10 0.01"
  )
  if [[ $QUICK -eq 1 ]]; then
    ED_ABL=("L1_g05 1 0.05")
    COMB_ABL=("L1_g05_s01 1 0.05 0.01")
  fi
  for cfg in "${ED_ABL[@]}"; do
    read -r label L gamma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "ed-fl-cond_${label}_seed${seed}" \
        "scripts/simulation_error_detected.py $seed $gamma 0.0 --L $L --full-likelihood --no-syndrome-factor"
    done
  done
  for cfg in "${COMB_ABL[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "comb-fl-cond_${label}_seed${seed}" \
        "scripts/simulation_combined.py $seed $gamma $sigma 0.0 --L $L --full-likelihood --no-syndrome-factor"
    done
  done
fi

# -- B2: Sequential protocol (CPU-only) --------------------------------------
if [[ $RUN_SEQUENTIAL -eq 1 ]]; then
  declare -a SEQ_CONFIGS=(
    "L1_s00 1 0.00" "L1_s01 1 0.01" "L1_s05 1 0.05"
    "L2_s00 2 0.00" "L2_s01 2 0.01" "L2_s05 2 0.05"
    "L3_s00 3 0.00" "L3_s01 3 0.01" "L3_s05 3 0.05"
  )
  [[ $QUICK -eq 1 ]] && SEQ_CONFIGS=("L1_s00 1 0.00")
  for cfg in "${SEQ_CONFIGS[@]}"; do
    read -r label L sigma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_cpu "seq_${label}_seed${seed}" \
        "scripts/simulation_sequential.py $seed 0.0 $sigma --L $L"
    done
  done
fi

# -- B3: Quenched-vs-marginalized (combined-FL --marginalize-noise) ----------
if [[ $RUN_MARG -eq 1 ]]; then
  declare -a MARG_CONFIGS=("L1_g05_s01 1 0.05 0.01" "L2_g05_s01 2 0.05 0.01")
  [[ $QUICK -eq 1 ]] && MARG_CONFIGS=("L1_g05_s01 1 0.05 0.01")
  for cfg in "${MARG_CONFIGS[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "comb-fl-marg_${label}_seed${seed}" \
        "scripts/simulation_combined.py $seed $gamma $sigma 0.0 --L $L --full-likelihood --marginalize-noise"
    done
  done
fi

# -- B4: σ_ε breakdown scan (combined-FL, large σ_ε) -------------------------
if [[ $RUN_SIGEPS -eq 1 ]]; then
  declare -a SIGEPS_CONFIGS=(
    "L2_g05_s10 2 0.05 0.10" "L2_g05_s15 2 0.05 0.15"
    "L2_g05_s20 2 0.05 0.20" "L2_g05_s30 2 0.05 0.30"
  )
  [[ $QUICK -eq 1 ]] && SIGEPS_CONFIGS=("L2_g05_s10 2 0.05 0.10")
  for cfg in "${SIGEPS_CONFIGS[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    for seed in $(seq 1 $SEEDS); do
      emit_gpu "comb-fl-sigeps_${label}_seed${seed}" \
        "scripts/simulation_combined.py $seed $gamma $sigma 0.0 --L $L --full-likelihood"
    done
  done
fi

n_gpu=$(wc -l < "$GPU_MANIFEST" | tr -d ' ')
n_cpu=$(wc -l < "$CPU_MANIFEST" | tr -d ' ')
total=$((n_gpu + n_cpu))
echo "Manifest emitted: $n_gpu GPU jobs + $n_cpu CPU jobs = $total total"
echo "  GPU manifest: $GPU_MANIFEST"
echo "  CPU manifest: $CPU_MANIFEST"

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "Dry-run complete; manifests written but no jobs executed."
  exit 0
fi

# -- Execute GPU manifest -----------------------------------------------------
# Round-robin GPU via CUDA_VISIBLE_DEVICES=$(({%}-1) % N_GPUS).
# {%} is GNU parallel's job slot number (1..GPU_JOBS), stable per worker slot,
# so each worker stays on its assigned GPU for the whole run (no JIT thrash).
if [[ $n_gpu -gt 0 ]]; then
  echo ""
  echo "=== Running $n_gpu GPU jobs ($GPU_JOBS concurrent across $N_GPUS GPUs) ==="
  echo "    JAX cache: $JAX_CACHE_DIR (shared across workers to avoid autotune contention)"
  mkdir -p "$JAX_CACHE_DIR"

  # Phase 1: warm-up one cell per GPU sequentially, populating the JAX cache.
  # Subsequent workers reuse cached kernels and avoid concurrent autotuning.
  echo "    Warmup: 1 cell per GPU to populate JAX cache..."
  for gpu_idx in $(seq 0 $((N_GPUS - 1))); do
    warmup_cmd=$(awk -F'\t' -v idx=$((gpu_idx + 1)) 'NR==idx {print $2}' "$GPU_MANIFEST")
    warmup_label=$(awk -F'\t' -v idx=$((gpu_idx + 1)) 'NR==idx {print $1}' "$GPU_MANIFEST")
    echo "      [GPU $gpu_idx] $warmup_label"
    CUDA_VISIBLE_DEVICES=$gpu_idx XLA_PYTHON_CLIENT_MEM_FRACTION=$GPU_MEM_FRAC \
      JAX_COMPILATION_CACHE_DIR="$JAX_CACHE_DIR" \
      bash -c "$warmup_cmd" > "$LOG_DIR/${warmup_label}.warmup.log" 2>&1 || \
      { echo "WARMUP FAILED on GPU $gpu_idx; see $LOG_DIR/${warmup_label}.warmup.log"; }
  done
  echo "    Warmup done. Launching parallel pool..."

  start_gpu=$(date +%s)
  parallel --colsep '\t' --jobs "$GPU_JOBS" --joblog "$GPU_JOBLOG" --bar \
    "CUDA_VISIBLE_DEVICES=\$((({%}-1) % $N_GPUS)) XLA_PYTHON_CLIENT_MEM_FRACTION=$GPU_MEM_FRAC \
       JAX_COMPILATION_CACHE_DIR=$JAX_CACHE_DIR \
       bash -c {2} > $LOG_DIR/{1}.log 2>&1" \
    :::: "$GPU_MANIFEST" || true
  end_gpu=$(date +%s)
  echo "GPU group done in $((end_gpu - start_gpu)) sec."
fi

# -- Execute CPU manifest -----------------------------------------------------
if [[ $n_cpu -gt 0 ]]; then
  echo ""
  echo "=== Running $n_cpu CPU jobs ($CPU_JOBS concurrent) ==="
  start_cpu=$(date +%s)
  parallel --colsep '\t' --jobs "$CPU_JOBS" --joblog "$CPU_JOBLOG" --bar \
    "bash -c {2} > $LOG_DIR/{1}.log 2>&1" \
    :::: "$CPU_MANIFEST" || true
  end_cpu=$(date +%s)
  echo "CPU group done in $((end_cpu - start_cpu)) sec."
fi

# -- Summary ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Suite complete."
if [[ $n_gpu -gt 0 ]]; then
  fail_gpu=$(awk 'NR>1 && $7!=0 {n++} END {print n+0}' "$GPU_JOBLOG")
  echo "GPU: $((n_gpu - fail_gpu))/$n_gpu succeeded ($fail_gpu failed). Joblog: $GPU_JOBLOG"
fi
if [[ $n_cpu -gt 0 ]]; then
  fail_cpu=$(awk 'NR>1 && $7!=0 {n++} END {print n+0}' "$CPU_JOBLOG")
  echo "CPU: $((n_cpu - fail_cpu))/$n_cpu succeeded ($fail_cpu failed). Joblog: $CPU_JOBLOG"
fi
echo ""
echo "Per-cell logs in $LOG_DIR/"
echo "Failed cells: awk -F'\\t' '\$7!=0' $GPU_JOBLOG $CPU_JOBLOG | head"
echo ""
echo "Next steps (analysis + figures):"
echo "  cd scripts"
echo "  uv run python analyze_results.py"
echo "  uv run python analyze_combined.py"
echo "  uv run python reanalysis_within_hamiltonian.py"
echo "  uv run python plot_redesigned.py"
echo "============================================================"
