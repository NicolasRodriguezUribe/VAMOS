#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

ts() { date -u +"%Y%m%dT%H%M%SZ"; }

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi

SPEC="${1:-}"
if [[ -z "$SPEC" ]]; then
  SPEC="$(find experiments -maxdepth 3 -name "engine_study_full_continuous*.yml" | sort | head -n 1 || true)"
fi
if [[ -z "$SPEC" || ! -f "$SPEC" ]]; then
  echo "[ERROR] Spec not found. Expected: experiments/**/engine_study_full_continuous*.yml"
  exit 2
fi

CAMPAIGN_TAG="engine_study_full_continuous"

RESULTS_ROOT="results/${CAMPAIGN_TAG}"
if grep -q "output_root:" "$SPEC"; then
  OR="$(grep -R "output_root:" -n "$SPEC" | tail -n 1 | sed -E 's/.*output_root:\s*//')"
  OR="$(echo "$OR" | sed -E 's/^["'\'']?|["'\'']?$//g' | xargs)"
  [[ -n "$OR" ]] && RESULTS_ROOT="$OR"
fi

TIDY_OUT="artifacts/tidy/${CAMPAIGN_TAG}.csv"
SAMPLE_OUT="experiments/sample_outputs/${CAMPAIGN_TAG}_sample.csv"

mkdir -p artifacts/bundles artifacts/tidy experiments/sample_outputs

echo "== Engine Study Full Continuous: full pipeline =="
echo "Repo root: $REPO_ROOT"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Python: $($PY -V 2>/dev/null || true)"
echo "Spec: $SPEC"
echo "Results root (expected): $RESULTS_ROOT"
echo

echo "== RUN: campaign =="
RUNNER="experiments/scripts/run_campaign.py"
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
echo "== COLLECT: campaign tidy CSV =="
COLLECTOR="experiments/scripts/collect_campaign_runs.py"
if [[ ! -f "$COLLECTOR" ]]; then
  echo "[ERROR] Collector not found: $COLLECTOR"
  exit 4
fi

COL_ARGS=()
if $PY "$COLLECTOR" --help 2>&1 | grep -q -- "--results-root"; then COL_ARGS+=( --results-root "$RESULTS_ROOT" ); fi
if $PY "$COLLECTOR" --help 2>&1 | grep -q -- "--out"; then COL_ARGS+=( --out "$TIDY_OUT" ); fi
if $PY "$COLLECTOR" --help 2>&1 | grep -q -- "--sample-out"; then COL_ARGS+=( --sample-out "$SAMPLE_OUT" ); fi

echo "[CMD] $PY $COLLECTOR ${COL_ARGS[*]}"
$PY "$COLLECTOR" "${COL_ARGS[@]}"

echo
echo "== REPORT: engine full continuous slice assets =="
REPORTER="experiments/scripts/report_engine_full_continuous_slice.py"
if [[ ! -f "$REPORTER" ]]; then
  echo "[ERROR] Reporter not found: $REPORTER"
  exit 5
fi

REP_ARGS=()
if $PY "$REPORTER" --help 2>&1 | grep -q -- "--tidy-csv"; then
  REP_ARGS+=( --tidy-csv "$TIDY_OUT" )
elif $PY "$REPORTER" --help 2>&1 | grep -q -- "--csv"; then
  REP_ARGS+=( --csv "$TIDY_OUT" )
elif $PY "$REPORTER" --help 2>&1 | grep -q -- "--input"; then
  REP_ARGS+=( --input "$TIDY_OUT" )
else
  echo "[ERROR] Unknown reporter input flag; run: $PY $REPORTER --help"
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
[[ -f "$SAMPLE_OUT" ]] && FILES+=( "$SAMPLE_OUT" )
[[ -f "paper/manuscript/sections/04_results_engine.tex" ]] && FILES+=( "paper/manuscript/sections/04_results_engine.tex" )
[[ -d "paper/manuscript/tables" ]] && FILES+=( "paper/manuscript/tables" )
[[ -d "paper/manuscript/figures" ]] && FILES+=( "paper/manuscript/figures" )
[[ -d "experiments/scripts/logs/${CAMPAIGN_TAG}" ]] && FILES+=( "experiments/scripts/logs/${CAMPAIGN_TAG}" )

echo "[CMD] tar --zstd -cf $BUNDLE <${#FILES[@]} paths>"
tar --zstd -cf "$BUNDLE" "${FILES[@]}"

echo
echo "[DONE] Bundle created: $BUNDLE"
echo "Copy to local:"
echo "  scp user@server:$REPO_ROOT/$BUNDLE ."
