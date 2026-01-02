#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

ts() { date -u +"%Y%m%dT%H%M%SZ"; }

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

SPEC_DEFAULT="${SCRIPT_DIR}/hv_archive_campaign_core.yml"
SPEC="${1:-$SPEC_DEFAULT}"
if [[ ! -f "$SPEC" ]]; then
  echo "[ERROR] Spec not found: $SPEC"
  exit 2
fi

CAMPAIGN_TAG="hv_archive_campaign_core"

# Best-effort results root (defaults to results/<tag>, overridden if output_root appears in spec)
RESULTS_ROOT="results/${CAMPAIGN_TAG}"
if grep -q "output_root:" "$SPEC"; then
  OR="$(grep -R "output_root:" -n "$SPEC" | tail -n 1 | sed -E 's/.*output_root:\s*//')"
  OR="$(echo "$OR" | sed -E 's/^["'\'']?|["'\'']?$//g' | xargs)"
  [[ -n "$OR" ]] && RESULTS_ROOT="$OR"
fi

REF_POINTS="experiments/catalog/hv_ref_points.json"
TIDY_OUT="artifacts/tidy/${CAMPAIGN_TAG}_metrics.csv"
SAMPLE_OUT="experiments/sample_outputs/${CAMPAIGN_TAG}_metrics_sample.csv"

mkdir -p artifacts/bundles artifacts/tidy experiments/sample_outputs

echo "== HV Archive Core: full pipeline =="
echo "Repo root: $REPO_ROOT"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Python: $($PY -V 2>/dev/null || true)"
echo "Spec: $SPEC"
echo "Results root (expected): $RESULTS_ROOT"
echo

echo "== RUN: campaign =="
RUNNER="experiments/scripts/run_campaign_variants_v2.py"
if [[ ! -f "$RUNNER" ]]; then
  echo "[ERROR] Runner not found: $RUNNER"
  exit 3
fi

echo "[CMD] $PY $RUNNER --spec $SPEC --resume"
$PY "$RUNNER" --spec "$SPEC" --resume

echo
echo "== CHECK: progress =="
if [[ -f "${RESULTS_ROOT}/runs_index.jsonl" ]]; then
  echo "runs_index lines: $(wc -l < "${RESULTS_ROOT}/runs_index.jsonl")"
else
  echo "[WARN] runs_index.jsonl not found at ${RESULTS_ROOT}/runs_index.jsonl"
fi

echo
echo "== COLLECT: HV/IGD+ metrics =="
COLLECTOR="experiments/scripts/collect_hv_archive_metrics.py"
if [[ ! -f "$COLLECTOR" ]]; then
  echo "[ERROR] Collector not found: $COLLECTOR"
  exit 4
fi

COL_ARGS=( --results-root "$RESULTS_ROOT" --out "$TIDY_OUT" --sample-out "$SAMPLE_OUT" )
# add optional flags only if supported
if $PY "$COLLECTOR" --help 2>&1 | grep -q -- "--mc-samples"; then COL_ARGS+=( --mc-samples 20000 ); fi
if $PY "$COLLECTOR" --help 2>&1 | grep -q -- "--rng-seed"; then COL_ARGS+=( --rng-seed 0 ); fi
if [[ -f "$REF_POINTS" ]] && $PY "$COLLECTOR" --help 2>&1 | grep -q -- "--ref-points"; then COL_ARGS+=( --ref-points "$REF_POINTS" ); fi

echo "[CMD] $PY $COLLECTOR ${COL_ARGS[*]}"
$PY "$COLLECTOR" "${COL_ARGS[@]}"

echo
echo "== REPORT: paper-ready assets =="
REPORTER="experiments/scripts/report_hv_archive_campaign.py"
if [[ ! -f "$REPORTER" ]]; then
  echo "[ERROR] Reporter not found: $REPORTER"
  exit 5
fi

REP_ARGS=()
if $PY "$REPORTER" --help 2>&1 | grep -q -- "--metrics-csv"; then
  REP_ARGS+=( --metrics-csv "$TIDY_OUT" )
elif $PY "$REPORTER" --help 2>&1 | grep -q -- "--tidy-csv"; then
  REP_ARGS+=( --tidy-csv "$TIDY_OUT" )
else
  echo "[ERROR] Reporter does not accept --metrics-csv/--tidy-csv. Run: $PY $REPORTER --help"
  exit 6
fi
if $PY "$REPORTER" --help 2>&1 | grep -q -- "--paper-root"; then REP_ARGS+=( --paper-root "paper/manuscript" ); fi
if $PY "$REPORTER" --help 2>&1 | grep -q -- "--tag"; then REP_ARGS+=( --tag "$CAMPAIGN_TAG" ); fi

echo "[CMD] $PY $REPORTER ${REP_ARGS[*]}"
$PY "$REPORTER" "${REP_ARGS[@]}"

echo
echo "== BUNDLE: minimal transferable artifacts =="
if ! command -v zstd >/dev/null 2>&1; then
  echo "[INFO] Installing zstd..."
  sudo apt-get update -y
  sudo apt-get install -y zstd
fi

MANIFEST_DIR="artifacts/bundles/${CAMPAIGN_TAG}_manifest_$(ts)"
mkdir -p "$MANIFEST_DIR"
git rev-parse HEAD > "${MANIFEST_DIR}/git_commit.txt" 2>/dev/null || true
$PY -V > "${MANIFEST_DIR}/python_version.txt" 2>/dev/null || true
$PY -m pip freeze > "${MANIFEST_DIR}/pip_freeze.txt" 2>/dev/null || true
cp -f "$SPEC" "${MANIFEST_DIR}/spec.yml"

BUNDLE="artifacts/bundles/${CAMPAIGN_TAG}_bundle_$(ts).tar.zst"

FILES=( "$MANIFEST_DIR" )
[[ -f "${RESULTS_ROOT}/runs_index.jsonl" ]] && FILES+=( "${RESULTS_ROOT}/runs_index.jsonl" )
[[ -f "$TIDY_OUT" ]] && FILES+=( "$TIDY_OUT" )
[[ -f "${TIDY_OUT%.csv}.problem_stats.json" ]] && FILES+=( "${TIDY_OUT%.csv}.problem_stats.json" )
[[ -f "$SAMPLE_OUT" ]] && FILES+=( "$SAMPLE_OUT" )
[[ -f "$REF_POINTS" ]] && FILES+=( "$REF_POINTS" )
[[ -f "paper/manuscript/sections/04b_results_hv_archive.tex" ]] && FILES+=( "paper/manuscript/sections/04b_results_hv_archive.tex" )
[[ -f "paper/manuscript/tables/${CAMPAIGN_TAG}_summary.tex" ]] && FILES+=( "paper/manuscript/tables/${CAMPAIGN_TAG}_summary.tex" )
[[ -f "paper/manuscript/figures/${CAMPAIGN_TAG}_tradeoff.png" ]] && FILES+=( "paper/manuscript/figures/${CAMPAIGN_TAG}_tradeoff.png" )
[[ -d "experiments/scripts/logs/${CAMPAIGN_TAG}" ]] && FILES+=( "experiments/scripts/logs/${CAMPAIGN_TAG}" )

echo "[CMD] tar --zstd -cf $BUNDLE <${#FILES[@]} paths>"
tar --zstd -cf "$BUNDLE" "${FILES[@]}"

echo
echo "[DONE] Bundle created: $BUNDLE"
echo "Copy to local:"
echo "  scp user@server:$REPO_ROOT/$BUNDLE ."
