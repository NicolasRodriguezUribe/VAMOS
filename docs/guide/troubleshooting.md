# Troubleshooting

## Unknown algorithm or engine

- Check `docs/reference/algorithms.md` for supported algorithms and backends.
- If you are using the CLI, verify `--algorithm` and `--engine` values in `docs/guide/cli.md`.

## Missing optional dependencies

- Backends (numba/moocore): `pip install -e ".[compute]"`
- External baselines (pymoo/jmetalpy/pygmo): `pip install -e ".[research]"`

## Benchmark suites

- Use the suite names shown by the benchmark CLI and match case exactly.
- See `docs/guide/cli.md` for examples of `vamos-benchmark` usage.
