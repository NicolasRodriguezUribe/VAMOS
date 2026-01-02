#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"
cd "$repo_root"

echo "== Environment =="
echo "CWD: $PWD"
if command -v git >/dev/null 2>&1; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git commit: $(git rev-parse HEAD)"
  fi
fi

if command -v python >/dev/null 2>&1; then
  echo "Python: $(python --version 2>&1)"
elif command -v python3 >/dev/null 2>&1; then
  echo "Python: $(python3 --version 2>&1)"
else
  echo "Python: not found"
fi

if [ -f ".venv/bin/python" ]; then
  echo ".venv/bin/python: present"
else
  echo ".venv/bin/python: missing"
fi

echo
echo "== Runner/Collector/Reporter Scripts =="
runner_output="$(ls -lah experiments/scripts | egrep "run_campaign|collect_|report_" || true)"
if [ -n "$runner_output" ]; then
  printf '%s\n' "$runner_output"
fi

echo
echo "== Campaign Specs =="
spec_output="$(find experiments -maxdepth 3 -name "*engine_study_full_continuous*.yml" -o -name "*hv_archive_campaign_core*.yml")"
if [ -n "$spec_output" ]; then
  printf '%s\n' "$spec_output"
fi

runner_paths="$(printf '%s\n' "$runner_output" | awk 'NF && $1 != "total" {if ($(NF-1) == "->") print $(NF-2); else print $NF}' | sed 's#^#experiments/scripts/#')"

echo
echo "== Next =="
echo "Runner scripts:"
if [ -n "$runner_paths" ]; then
  printf '%s\n' "$runner_paths"
fi
echo "Spec files:"
if [ -n "$spec_output" ]; then
  printf '%s\n' "$spec_output"
fi
