#!/usr/bin/env bash
set -euo pipefail

# -------- helpers --------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

ts() { date -u +"%Y%m%dT%H%M%SZ"; }

pick_python() {
  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
  else
    echo "python3"
  fi
}

has_flag() {
  local cmd="$1"; local flag="$2"
  "$cmd" --help 2>&1 | grep -q -- "$flag"
}

ensure_zstd() {
  if ! command -v zstd >/dev/null 2>&1; then
    echo "[INFO] Installing zstd..."
    sudo apt-get update -y
    sudo apt-get install -y zstd
  fi
}

# -------- config --------
PY="$(pick_python)"
SPEC_DEFAULT="experiments/configs/hv_archive_campaign_core.yml"
SPEC="${1:-$SPEC_DEFAULT}"

if [[ ! -f "$SPEC" ]]; then
  echo "[ERROR] Spec not found: $SPEC"
  exit 2
fi

CAMPAIGN_TAG="hv_archive_campaign_core"

# Best-effort output root (default to results/<tag>)
OUTPUT_ROOT="results/${CAMPAIGN_TAG}"
# if the spec contains an output_root field, take the last occurrence
if grep -q "output_root:" "$SPEC"; then
  OUTPUT_ROOT_GUESS="$(grep -R "output_root:" -n "$SPEC" | tail -n 1 | sed -E 's/.*output_root:\s*//')"
  # trim quotes/spaces
  OUTPUT_ROOT_GUESS="$(echo "$OUTPUT_ROOT_GUESS" | sed -E 's/^["'\'']?|["'\'']?$//g' | xargs)"
  if [[ -n "$OUTPUT_ROOT_GUESS" ]]; then
    OUTPUT_ROOT="$OUTPUT_ROOT_GUESS"
  fi
fi

RESULTS_ROOT="$OUTPUT_ROOT"

REF_POINTS="experiments/catalog/hv_ref_points.json"
TIDY_OUT="artifacts/tidy/${CAMPAIGN_TAG}_metrics.csv"
SAMPLE_OUT="experiments/sample_outputs/${CAMPAIGN_TAG}_metrics_sample.csv"

echo "== HV Archive Core: full pipeline =="
echo "Repo root: $REPO_ROOT"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Python: $($PY -V 2>/dev/null || true)"
echo "Spec: $SPEC"
echo "Results root (expected): $RESULTS_ROOT"
echo

# -------- run campaign --------
echo "== RUN: campaign =="
RUNNER="experiments/scripts/run_campaign_variants_v2.py"
if [[ ! -f "$RUNNER" ]]; then
  echo "[ERROR] Runner not found: $RUNNER"
  exit 3
fi

# build runner args robustly
RUN_ARGS=()
if has_flag "$PY $RUNNER" "--spec"; then
  RUN_ARGS+=(--spec "$SPEC")
else
  # fallback if runner uses --config
  RUN_ARGS+=(--config "$SPEC")
fi
if has_flag "$PY $RUNNER" "--resume"; then
  RUN_ARGS+=(--resume)
fi

echo "[CMD] $PY $RUNNER ${RUN_ARGS[*]}"
$PY "$RUNNER" "${RUN_ARGS[@]}"

echo
echo "== CHECK: basic progress files =="
if [[ -f "${RESULTS_ROOT}/runs_index.jsonl" ]]; then
  echo "runs_index lines: $(wc -l < "${RESULTS_ROOT}/runs_index.jsonl")"
else
  echo "[WARN] runs_index.jsonl not found at ${RESULTS_ROOT}/runs_index.jsonl"
  echo "       You may have a different output_root in the spec."
fi

# -------- collect metrics --------
echo
echo "== COLLECT: HV/IGD+ metrics =="
COLLECTOR="experiments/scripts/collect_hv_archive_metrics.py"
if [[ ! -f "$COLLECTOR" ]]; then
  echo "[ERROR] Collector not found: $COLLECTOR"
  exit 4
fi

COL_ARGS=()
COL_ARGS+=(--results-root "$RESULTS_ROOT")
COL_ARGS+=(--out "$TIDY_OUT")
COL_ARGS+=(--sample-out "$SAMPLE_OUT")

# Optional knobs if supported
if has_flag "$PY $COLLECTOR" "--mc-samples"; then
  COL_ARGS+=(--mc-samples 20000)
fi
if has_flag "$PY $COLLECTOR" "--rng-seed"; then
  COL_ARGS+=(--rng-seed 0)
fi
if [[ -f "$REF_POINTS" ]] && has_flag "$PY $COLLECTOR" "--ref-points"; then
  COL_ARGS+=(--ref-points "$REF_POINTS")
fi

echo "[CMD] $PY $COLLECTOR ${COL_ARGS[*]}"
$PY "$COLLECTOR" "${COL_ARGS[@]}"

# -------- report --------
echo
echo "== REPORT: paper-ready assets =="
REPORTER="experiments/scripts/report_hv_archive_campaign.py"
if [[ ! -f "$REPORTER" ]]; then
  echo "[ERROR] Reporter not found: $REPORTER"
  exit 5
fi

REP_ARGS=()
if has_flag "$PY $REPORTER" "--metrics-csv"; then
  REP_ARGS+=(--metrics-csv "$TIDY_OUT")
elif has_flag "$PY $REPORTER" "--tidy-csv"; then
  REP_ARGS+=(--tidy-csv "$TIDY_OUT")
else
  echo "[ERROR] Reporter flags unknown; inspect: $PY $REPORTER --help"
  exit 6
fi

if has_flag "$PY $REPORTER" "--paper-root"; then
  REP_ARGS+=(--paper-root "paper/manuscript")
fi
if has_flag "$PY $REPORTER" "--tag"; then
  REP_ARGS+=(--tag "$CAMPAIGN_TAG")
fi

echo "[CMD] $PY $REPORTER ${REP_ARGS[*]}"
$PY "$REPORTER" "${REP_ARGS[@]}"

# -------- bundle --------
echo
echo "== BUNDLE: minimal transferable artifacts (single file) =="
ensure_zstd

mkdir -p artifacts/bundles
MANIFEST_DIR="artifacts/bundles/${CAMPAIGN_TAG}_manifest_$(ts)"
mkdir -p "$MANIFEST_DIR"

git rev-parse HEAD > "${MANIFEST_DIR}/git_commit.txt" 2>/dev/null || true
$PY -V > "${MANIFEST_DIR}/python_version.txt" 2>/dev/null || true
$PY -m pip freeze > "${MANIFEST_DIR}/pip_freeze.txt" 2>/dev/null || true
cp -f "$SPEC" "${MANIFEST_DIR}/spec.yml"

BUNDLE="artifacts/bundles/${CAMPAIGN_TAG}_bundle_$(ts).tar.zst"

FILES=()
FILES+=("$MANIFEST_DIR")
# index if present
[[ -f "${RESULTS_ROOT}/runs_index.jsonl" ]] && FILES+=("${RESULTS_ROOT}/runs_index.jsonl")
# tidy outputs
[[ -f "$TIDY_OUT" ]] && FILES+=("$TIDY_OUT")
[[ -f "${TIDY_OUT%.csv}.problem_stats.json" ]] && FILES+=("${TIDY_OUT%.csv}.problem_stats.json")
[[ -f "$SAMPLE_OUT" ]] && FILES+=("$SAMPLE_OUT")
# ref points
[[ -f "$REF_POINTS" ]] && FILES+=("$REF_POINTS")
# paper outputs (if they exist)
[[ -f "paper/manuscript/sections/04b_results_hv_archive.tex" ]] && FILES+=("paper/manuscript/sections/04b_results_hv_archive.tex")
[[ -f "paper/manuscript/tables/${CAMPAIGN_TAG}_summary.tex" ]] && FILES+=("paper/manuscript/tables/${CAMPAIGN_TAG}_summary.tex")
[[ -f "paper/manuscript/figures/${CAMPAIGN_TAG}_tradeoff.png" ]] && FILES+=("paper/manuscript/figures/${CAMPAIGN_TAG}_tradeoff.png")
# logs dir if exists
[[ -d "experiments/scripts/logs/${CAMPAIGN_TAG}" ]] && FILES+=("experiments/scripts/logs/${CAMPAIGN_TAG}")

echo "[CMD] tar --zstd -cf $BUNDLE <${#FILES[@]} paths>"
tar --zstd -cf "$BUNDLE" "${FILES[@]}"

echo
echo "[DONE] Bundle created:"
echo "  $BUNDLE"
echo "Copy to your machine via:"
echo "  scp user@server:$REPO_ROOT/$BUNDLE ."
