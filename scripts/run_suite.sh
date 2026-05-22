#!/bin/bash
# run_suite.sh - Reproduce + complement §9 numerical experiments for the EQSP
# metrology paper.
#
# Reproductions: Tables I, III, V, IV apples-to-apples baseline.
# Complements:
#   - A3/A4 ablation: ED-FL / combined-FL with --no-syndrome-factor (cond-only),
#     paired with default ED-FL/combined-FL to quantify joint vs cond Fisher gain.
#   - Sequential protocol (Theorems 22, 24): product-state binary search.
#   - Quenched-vs-marginalized A/B (combined --marginalize-noise) at one config.
#   - σ_ε breakdown scan: combined L=2 at γ=0.05 for σ_ε ∈ {0.10, 0.15, 0.20, 0.30}.
#
# Each script appends to a fixed CSV in scripts/results/, so this script must
# be run sequentially (no parallel writers to the same file).
#
# Usage:
#   bash run_suite.sh                    # full suite (40 seeds, 60 epsilon)
#   bash run_suite.sh --seeds N          # override seed count (default 40)
#   bash run_suite.sh --quick            # 4 seeds, single config per group (smoke test)
#   bash run_suite.sh --skip-bare        # skip bare-GHZ (Table I)
#   bash run_suite.sh --skip-ed          # skip ED protocols (Table III)
#   bash run_suite.sh --skip-combined    # skip combined protocol (Table V)
#   bash run_suite.sh --skip-bare-ham    # skip bare-Hamiltonian (Table IV apples-to-apples + narrative pilot)
#   bash run_suite.sh --skip-ablation    # skip A3/A4 cond-only ablation (B1)
#   bash run_suite.sh --skip-sequential  # skip Sequential protocol sim (B2)
#   bash run_suite.sh --skip-marg        # skip quenched-vs-marginalized A/B (B3)
#   bash run_suite.sh --skip-sigeps      # skip σ_ε breakdown scan (B4)
#
# Outputs (appended; back up before re-running for clean results):
#   results/bare_ghz.csv                       # Table I
#   results/ed_postselect.csv                  # Table III right
#   results/ed_full_likelihood.csv             # Table III left
#   results/combined_postselect.csv            # Table V right
#   results/combined_full_likelihood.csv       # Table V left + Figure 3
#   results/bare_hamiltonian.csv               # Table IV "Bare GHZ" col + §9 pilot
#   results/ed_full_likelihood_cond_only.csv   # B1: ED-FL ablation arm (cond-only)
#   results/combined_full_likelihood_cond_only.csv # B1: combined-FL ablation arm
#   results/sequential.csv                     # B2: sequential protocol
#   results/combined_full_likelihood_marg.csv  # B3: marginalized-noise combined-FL
#   (σ_ε scan appends to combined_full_likelihood.csv with new σ_ε rows)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUMERICS_DIR="$(dirname "$SCRIPT_DIR")"
cd "$NUMERICS_DIR"

# Defaults
SEEDS=40
QUICK=0
RUN_BARE=1
RUN_ED=1
RUN_COMBINED=1
RUN_BARE_HAM=1
RUN_ABLATION=1
RUN_SEQUENTIAL=1
RUN_MARG=1
RUN_SIGEPS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;
    --quick) QUICK=1; SEEDS=4; shift;;
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

echo "============================================================"
echo "EQSP metrology numerical suite"
echo "  Seeds:               $SEEDS"
echo "  Quick:               $QUICK"
echo "  Bare GHZ depol:      $RUN_BARE"
echo "  Error-det Ham:       $RUN_ED"
echo "  Combined Ham:        $RUN_COMBINED"
echo "  Bare GHZ Ham:        $RUN_BARE_HAM"
echo "  ED/comb ablation:    $RUN_ABLATION"
echo "  Sequential proto:    $RUN_SEQUENTIAL"
echo "  Quenched-vs-marg:    $RUN_MARG"
echo "  σ_ε breakdown scan:  $RUN_SIGEPS"
echo "  Working dir:         $NUMERICS_DIR"
echo "============================================================"
echo ""

run_seeds() {
  local label="$1"
  shift
  local cmd=("$@")
  for seed in $(seq 1 "$SEEDS"); do
    echo ">>> [$label] seed=$seed: uv run python ${cmd[*]} $seed"
    uv run python "${cmd[@]/SEED/$seed}" || { echo "FAILED: $label seed=$seed"; exit 1; }
  done
}

# -- Bare GHZ (Table I, depolarizing channel) --------------------------------
if [[ $RUN_BARE -eq 1 ]]; then
  echo ""
  echo "=== Bare GHZ (Table I) ==="
  declare -a BARE_CONFIGS=(
    "noiseless 0.0 0.0"
    "g01 0.01 0.0"
    "g05 0.05 0.0"
    "g05_h30 0.05 0.3"
    "g10 0.10 0.0"
    "g15 0.15 0.0"
    "g20 0.20 0.0"
  )
  if [[ $QUICK -eq 1 ]]; then
    BARE_CONFIGS=("g05 0.05 0.0")
  fi
  for cfg in "${BARE_CONFIGS[@]}"; do
    read -r label gamma h <<< "$cfg"
    run_seeds "bare-ghz/$label" \
      scripts/simulation.py SEED "$gamma" "$h" --mode ghz
  done
fi

# -- Error-detected (Table III, Hamiltonian noise, [[2L+1,1]] code) ----------
if [[ $RUN_ED -eq 1 ]]; then
  echo ""
  echo "=== Error-detected (Table III) ==="
  declare -a ED_CONFIGS=(
    "L1_g01 1 0.01"
    "L1_g05 1 0.05"
    "L1_g10 1 0.10"
    "L2_g01 2 0.01"
    "L2_g05 2 0.05"
    "L2_g10 2 0.10"
    "L3_g01 3 0.01"
    "L3_g05 3 0.05"
    "L3_g10 3 0.10"
  )
  if [[ $QUICK -eq 1 ]]; then
    ED_CONFIGS=("L1_g05 1 0.05")
  fi
  # Both inference modes (post-selection and full-likelihood)
  for cfg in "${ED_CONFIGS[@]}"; do
    read -r label L gamma <<< "$cfg"
    run_seeds "ed-ps/$label" \
      scripts/simulation_error_detected.py SEED "$gamma" 0.0 --L "$L"
    run_seeds "ed-fl/$label" \
      scripts/simulation_error_detected.py SEED "$gamma" 0.0 --L "$L" --full-likelihood
  done
fi

# -- Combined (Table V, transverse + longitudinal noise) ---------------------
if [[ $RUN_COMBINED -eq 1 ]]; then
  echo ""
  echo "=== Combined (Table V) ==="
  declare -a COMB_CONFIGS=(
    "L1_g01_s01 1 0.01 0.01"
    "L1_g05_s01 1 0.05 0.01"
    "L1_g10_s01 1 0.10 0.01"
    "L1_g01_s05 1 0.01 0.05"
    "L1_g05_s05 1 0.05 0.05"
    "L1_g10_s05 1 0.10 0.05"
    "L2_g01_s01 2 0.01 0.01"
    "L2_g05_s01 2 0.05 0.01"
    "L2_g10_s01 2 0.10 0.01"
    "L2_g01_s05 2 0.01 0.05"
    "L2_g05_s05 2 0.05 0.05"
    "L2_g10_s05 2 0.10 0.05"
  )
  if [[ $QUICK -eq 1 ]]; then
    COMB_CONFIGS=("L1_g05_s01 1 0.05 0.01")
  fi
  # Combined: positional args are (seed, gamma, sigma_epsilon, h)
  for cfg in "${COMB_CONFIGS[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    run_seeds "comb-ps/$label" \
      scripts/simulation_combined.py SEED "$gamma" "$sigma" 0.0 --L "$L"
    run_seeds "comb-fl/$label" \
      scripts/simulation_combined.py SEED "$gamma" "$sigma" 0.0 --L "$L" --full-likelihood
  done
fi

# -- Bare GHZ under Hamiltonian noise -----------------------------------------
# Two purposes:
#  (a) Apples-to-apples baseline for Table IV (tab:prefactor-results): same
#      (L, gamma_Ham) grid as the ED simulator so the bare-GHZ vs ED-FL
#      prefactor comparison is at matched code distance and noise rate.
#  (b) §9 narrative pilot at L=2, gamma_Ham in {0.10, 0.30, 0.45} showing
#      bare-GHZ also achieves near-Heisenberg scaling under Hamiltonian noise
#      (line 1872 of sensing.tex).
if [[ $RUN_BARE_HAM -eq 1 ]]; then
  echo ""
  echo "=== Bare GHZ + Hamiltonian noise (Table IV apples-to-apples + §9 pilot) ==="
  # (a) Apples-to-apples matrix matching the ED-FL grid (L x gamma_Ham)
  declare -a BARE_HAM_MATRIX=(
    "L1_g01 1 0.01"
    "L1_g05 1 0.05"
    "L1_g10 1 0.10"
    "L2_g01 2 0.01"
    "L2_g05 2 0.05"
    "L2_g10 2 0.10"
    "L3_g01 3 0.01"
    "L3_g05 3 0.05"
    "L3_g10 3 0.10"
  )
  # (b) §9 narrative pilot at higher gamma_Ham (L=2)
  declare -a BARE_HAM_PILOT=(
    "L2_g030 2 0.30"
    "L2_g045 2 0.45"
  )
  if [[ $QUICK -eq 1 ]]; then
    BARE_HAM_MATRIX=("L1_g05 1 0.05")
    BARE_HAM_PILOT=()
  fi
  for cfg in "${BARE_HAM_MATRIX[@]}" "${BARE_HAM_PILOT[@]}"; do
    read -r label L gamma <<< "$cfg"
    run_seeds "bare-ham/$label" \
      scripts/simulation_bare_hamiltonian.py SEED "$gamma" 0.0 --L "$L"
  done
fi

# -- B1: A3/A4 ablation arm ---------------------------------------------------
# Re-run a representative ED-FL/combined-FL grid with --no-syndrome-factor (cond
# only). Pair against existing ED-FL/combined-FL CSVs to quantify Fisher gain
# of the joint estimator. Run on a focused grid (not full Table III/V) since
# the gain is expected subleading O(γ⁴/(N²ω⁴)).
if [[ $RUN_ABLATION -eq 1 ]]; then
  echo ""
  echo "=== B1: A3/A4 ablation (ED-FL + combined-FL, --no-syndrome-factor) ==="
  declare -a ED_ABL_CONFIGS=(
    "L1_g01 1 0.01"
    "L1_g10 1 0.10"
    "L3_g01 3 0.01"
    "L3_g10 3 0.10"
  )
  declare -a COMB_ABL_CONFIGS=(
    "L1_g01_s01 1 0.01 0.01"
    "L1_g10_s01 1 0.10 0.01"
    "L2_g01_s01 2 0.01 0.01"
    "L2_g10_s01 2 0.10 0.01"
  )
  if [[ $QUICK -eq 1 ]]; then
    ED_ABL_CONFIGS=("L1_g05 1 0.05")
    COMB_ABL_CONFIGS=("L1_g05_s01 1 0.05 0.01")
  fi
  for cfg in "${ED_ABL_CONFIGS[@]}"; do
    read -r label L gamma <<< "$cfg"
    run_seeds "ed-fl-cond/$label" \
      scripts/simulation_error_detected.py SEED "$gamma" 0.0 --L "$L" \
      --full-likelihood --no-syndrome-factor
  done
  for cfg in "${COMB_ABL_CONFIGS[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    run_seeds "comb-fl-cond/$label" \
      scripts/simulation_combined.py SEED "$gamma" "$sigma" 0.0 --L "$L" \
      --full-likelihood --no-syndrome-factor
  done
fi

# -- B2: Sequential protocol (Theorem 22 / 24) -------------------------------
# Validates Heisenberg α=1 for product-state probes with M repeated signal
# applications per qubit. Sweeps L (3 code distances) × σ_ε (4 levels: 0,
# 0.01, 0.05, 0.10) to characterize the device-quality regime M_*² σ_ε² = o(1)
# and its breakdown. No γ dependence in the basic sequential model (Theorem
# 22 noiseless guard band; Theorem 24 longitudinal-only noise).
if [[ $RUN_SEQUENTIAL -eq 1 ]]; then
  echo ""
  echo "=== B2: Sequential protocol (Theorems 22, 24) ==="
  declare -a SEQ_CONFIGS=(
    "L1_s00 1 0.00"
    "L1_s01 1 0.01"
    "L1_s05 1 0.05"
    "L2_s00 2 0.00"
    "L2_s01 2 0.01"
    "L2_s05 2 0.05"
    "L3_s00 3 0.00"
    "L3_s01 3 0.01"
    "L3_s05 3 0.05"
  )
  if [[ $QUICK -eq 1 ]]; then
    SEQ_CONFIGS=("L1_s00 1 0.00")
  fi
  for cfg in "${SEQ_CONFIGS[@]}"; do
    read -r label L sigma <<< "$cfg"
    run_seeds "seq/$label" \
      scripts/simulation_sequential.py SEED 0.0 "$sigma" --L "$L"
  done
fi

# -- B3: Quenched-vs-marginalized A/B (combined --marginalize-noise) ---------
# At one config (L=1, γ=0.05, σ_ε=0.01), run combined-FL with per-shot ε_k
# resampling (inference assumes ε_k=0). Pair against existing combined-FL
# (quenched) at same config. Validates §9.1.3 claim that they agree to leading
# order in N_total σ_ε² = o(1).
if [[ $RUN_MARG -eq 1 ]]; then
  echo ""
  echo "=== B3: Quenched-vs-marginalized A/B (combined-FL --marginalize-noise) ==="
  declare -a MARG_CONFIGS=(
    "L1_g05_s01 1 0.05 0.01"
    "L2_g05_s01 2 0.05 0.01"
  )
  if [[ $QUICK -eq 1 ]]; then
    MARG_CONFIGS=("L1_g05_s01 1 0.05 0.01")
  fi
  for cfg in "${MARG_CONFIGS[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    run_seeds "comb-fl-marg/$label" \
      scripts/simulation_combined.py SEED "$gamma" "$sigma" 0.0 --L "$L" \
      --full-likelihood --marginalize-noise
  done
fi

# -- B4: σ_ε breakdown scan (combined --full-likelihood, large σ_ε) ----------
# Pushes σ_ε beyond Table V's {0.01, 0.05} to map where the device-quality
# condition N_total σ_ε² = o(1) breaks. L=2 (N_total=15) at γ=0.05.
# Appends to combined_full_likelihood.csv (mode='combined_full_likelihood').
if [[ $RUN_SIGEPS -eq 1 ]]; then
  echo ""
  echo "=== B4: σ_ε breakdown scan (combined-FL, L=2, γ=0.05) ==="
  declare -a SIGEPS_CONFIGS=(
    "L2_g05_s10 2 0.05 0.10"
    "L2_g05_s15 2 0.05 0.15"
    "L2_g05_s20 2 0.05 0.20"
    "L2_g05_s30 2 0.05 0.30"
  )
  if [[ $QUICK -eq 1 ]]; then
    SIGEPS_CONFIGS=("L2_g05_s10 2 0.05 0.10")
  fi
  for cfg in "${SIGEPS_CONFIGS[@]}"; do
    read -r label L gamma sigma <<< "$cfg"
    run_seeds "comb-fl-sigeps/$label" \
      scripts/simulation_combined.py SEED "$gamma" "$sigma" 0.0 --L "$L" \
      --full-likelihood
  done
fi

echo ""
echo "============================================================"
echo "Suite complete. CSVs in scripts/results/"
echo ""
echo "Next steps (analysis + figures):"
echo "  cd scripts"
echo "  uv run python analyze_results.py                  # Table I (bare-GHZ depolarizing)"
echo "  uv run python analyze_combined.py                 # Table V (combined protocol)"
echo "  uv run python reanalysis_within_hamiltonian.py    # Table IV (apples-to-apples bare-Ham vs ED)"
echo "  uv run python plot_redesigned.py                  # Figures 1-3"
echo "============================================================"
