from vamos.experiment.diagnostics import self_check


def test_self_check_runs_numpy_only():
    results = self_check.run_self_check(verbose=False)
    assert any(r.name == "nsgaii-numpy" and r.status == "ok" for r in results)
