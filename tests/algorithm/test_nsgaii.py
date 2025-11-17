"""Tests for the NSGA-II implementation."""

import numpy as np
import pytest
from vamos.algorithm import NSGAII, NSGAIIConfig
from vamos.problem import ZDT1Problem


def test_nsgaii_initialization():
    """Test that NSGA-II initializes correctly."""
    problem = ZDT1Problem(n_var=10)
    config = NSGAIIConfig()
    algo = NSGAII(config=config, kernel=None)  # Kernel will be created in __post_init__
    
    assert algo is not None
    assert algo.cfg == config
    assert algo.kernel is not None


def test_nsgaii_run():
    """Test a basic NSGA-II run."""
    problem = ZDT1Problem(n_var=10)
    config = NSGAIIConfig(pop_size=10)
    algo = NSGAII(config=config, kernel=None)
    
    result = algo.run(problem, termination=("n_eval", 100), seed=42)
    
    assert "X" in result
    assert "F" in result
    assert isinstance(result["X"], np.ndarray)
    assert isinstance(result["F"], np.ndarray)
    assert result["X"].shape[0] == config.pop_size
    assert result["F"].shape[0] == config.pop_size
