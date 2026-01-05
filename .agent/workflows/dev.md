---
description: Common development commands for VAMOS project
---

// turbo-all

This workflow enables auto-running of common development commands.

## Python Environment
1. Activate venv: `.venv\Scripts\activate`
2. Install package: `.venv\Scripts\pip install -e .`
3. Install with extras: `.venv\Scripts\pip install -e ".[backends,benchmarks,notebooks]"`

## Running Code
4. Run Python script: `.venv\Scripts\python <script.py>`
5. Run module: `.venv\Scripts\python -m <module>`
6. Quick Python check: `.venv\Scripts\python -c "<code>"`

## Testing
7. Run unit tests: `.venv\Scripts\pytest tests/engine tests/foundation`
8. Run Integration Gauntlet (E2E): `.venv\Scripts\pytest tests/e2e`
9. Run Reference Verification (Scientific): `.venv\Scripts\pytest tests/reference`
10. Run all tests: `.venv\Scripts\pytest`

## Package Management
10. List installed: `.venv\Scripts\pip list`
11. Show package info: `.venv\Scripts\pip show <package>`
12. Check outdated: `.venv\Scripts\pip list --outdated`

## File Operations
13. List directory: `dir`
14. Find files: `Get-ChildItem -Recurse -Filter <pattern>`
15. View file: `type <file>`
