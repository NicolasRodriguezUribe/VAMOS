from vamos.experiment.diagnostics import self_check


def test_self_check_runs_numpy_only():
    results = self_check.run_self_check(verbose=False)
    assert any(r.name == "nsgaii-numpy" and r.status == "ok" for r in results)


def test_self_check_main_respects_quiet_argv(monkeypatch):
    calls: list[bool] = []

    def fake_run_self_check(*, verbose: bool):
        calls.append(verbose)
        return [self_check.CheckResult(name="nsgaii-numpy", status="ok")]

    monkeypatch.setattr(self_check, "run_self_check", fake_run_self_check)

    assert self_check.main(["--quiet"]) == 0
    assert calls == [False]
