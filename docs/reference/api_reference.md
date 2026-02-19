# API Reference

This reference is automatically generated from the docstrings in the source code.

## Unified API

The primary user-facing entry points.  Most users only need these.

::: vamos.experiment.unified
    options:
      show_root_heading: true
      show_source: true
      members:
        - optimize

## Problem Definition

::: vamos.foundation.problem.base
    options:
      show_root_heading: true
      members:
        - Problem

::: vamos.foundation.problem.builder
    options:
      show_root_heading: true
      members:
        - make_problem
        - FunctionalProblem

::: vamos.foundation.problem.types
    options:
      show_root_heading: true

## Results

::: vamos.experiment.optimization_result
    options:
      show_root_heading: true
      show_source: true

## Algorithm Configuration

::: vamos.engine.algorithm.config.nsgaii
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.moead
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.nsgaiii
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.smsemoa
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.spea2
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.ibea
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.smpso
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.agemoea
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.rvea
    options:
      show_root_heading: true

::: vamos.engine.algorithm.config.generic
    options:
      show_root_heading: true

## Constraint Handling

::: vamos.foundation.constraints
    options:
      show_root_heading: true
      members:
        - ConstraintInfo
        - compute_constraint_info
        - FeasibilityFirstStrategy
        - PenaltyCVStrategy
        - CVAsObjectiveStrategy
        - EpsilonConstraintStrategy
        - get_constraint_strategy

::: vamos.foundation.constraints.utils
    options:
      show_root_heading: true

## Encoding

::: vamos.foundation.encoding
    options:
      show_root_heading: true

## Problem Registry

::: vamos.foundation.problem.registry
    options:
      show_root_heading: true
      members:
        - available_problem_names
        - make_problem_selection

## Tuning

Parameters and Tuners for Hyperparameter Optimization.

::: vamos.engine.tuning.racing.core
    options:
      show_root_heading: true
      show_source: true

::: vamos.engine.tuning.racing.param_space
    options:
      show_root_heading: true

## Diagnostics

::: vamos.experiment.diagnostics.self_check
    options:
      show_root_heading: true
      members:
        - run_self_check
