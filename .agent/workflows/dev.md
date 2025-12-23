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
7. Run all tests: `.venv\Scripts\pytest`
8. Run specific test: `.venv\Scripts\pytest tests/<test_file.py>`
9. Run with verbose: `.venv\Scripts\pytest -v`

## Package Management
10. List installed: `.venv\Scripts\pip list`
11. Show package info: `.venv\Scripts\pip show <package>`
12. Check outdated: `.venv\Scripts\pip list --outdated`

## File Operations
13. List directory: `dir`
14. Find files: `Get-ChildItem -Recurse -Filter <pattern>`
15. View file: `type <file>`
