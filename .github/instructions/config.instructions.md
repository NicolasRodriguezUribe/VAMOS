---
applyTo: "**/*.yaml,**/*.yml,examples/*.yaml"
description: YAML experiment config guidelines for VAMOS
---

# YAML Config Standards for VAMOS

## Structure
Experiment configs follow this hierarchy:
```yaml
defaults:
  algorithm: nsgaii
  engine: numpy
  population_size: 100
  max_evaluations: 20000

problems:
  zdt1:
    n_var: 30
  dtlz2:
    algorithm: moead
    n_obj: 3

nsgaii:
  crossover: {method: sbx, prob: 0.9, eta: 20}
  mutation: {method: pm, prob: "1/n", eta: 20}
```

## Key Rules
- `defaults:` section applies to all problems unless overridden
- Per-problem sections override defaults (e.g., `problems.dtlz2.algorithm: moead`)
- Algorithm-specific sections (`nsgaii:`, `moead:`, `spea2:`) define operator configs
- Use `"1/n"` string for dimension-dependent mutation probability

## Valid Algorithm Names
`nsgaii`, `moead`, `smsemoa`, `nsga3`, `spea2`, `ibea`, `smpso`

## Valid Engine Names
`numpy`, `numba`, `moocore`

## Crossover Methods
`sbx`, `blx_alpha`, `uniform`, `one_point`, `two_point`

## Mutation Methods
`pm` (polynomial), `uniform`, `bitflip`, `swap`, `insert`, `inversion`
