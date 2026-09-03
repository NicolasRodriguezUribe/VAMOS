# Governed experiment reference results

These files are the minimal authoritative numerical inputs used to regenerate
the current paper tables and figures. They are `REFERENCE_BENCHMARK_DATA`, not
VAMOS RunManifest or StudyManifest directories. Raw runs, traces, tuning
databases, backups, derived statistics, plots and LaTeX remain external or
ignored.

The frozen source commit for this retained set is
`dce9f5b04b3aa6826d1cf8e79c13ac0a3bd590f2`. Regeneration is publication-scale
work and is not part of routine CI.

| Files | Generation command | Schema | Expected use | Size budget |
|---|---|---|---|---:|
| `benchmark_paper.csv`, `benchmark_paper_agemoea.csv`, `benchmark_paper_moead.csv`, `benchmark_paper_nsgaii_archive.csv`, `benchmark_paper_nsgaii_ss.csv`, `benchmark_paper_nsgaiii.csv`, `benchmark_paper_smsemoa.csv`, `benchmark_paper_spea2.csv` | `VAMOS_PAPER_ALGORITHM=all python paper/01_run_paper_benchmark.py` (with the algorithm/variant selected as documented by the script) | framework, problem, algorithm, evaluations, seed, runtime, solution count, hypervolume, IGD+ | paper runtime/quality tables and downstream statistical analysis | 2 MiB each; 5 MiB group total |
| `benchmark_zcat_scalability.csv` | `python paper/31_run_zcat_all_tables.py` | framework, problem, objective/variable count, algorithm, evaluations, seed, runtime, solution count, hypervolume, IGD+ | ZCAT scalability rows and plots | 2 MiB |
| `convergence_paper.csv` | `python paper/18_run_convergence_experiment.py` | framework, problem, seed, evaluations, hypervolume | convergence figure | 2 MiB |
| `memory_benchmark.csv` | `python paper/23_run_memory_benchmark.py` | framework, problem, seed, evaluations, peak memory, runtime | memory comparison | 256 KiB |
| `scaling_vectorization.csv` | `python paper/03_run_scaling_experiment.py` | experiment, problem dimensions, population/evaluation budget, seed, engine, timing policy, runtime, runtime/evaluation, hypervolume | vectorization scaling analysis | 512 KiB |
| `mic/instance_selection/representative_instances_mic_runtime_p30p0_k10.csv` and `.json` | selection method and quotas frozen in the JSON companion; no active campaign runner is required to consume them | ranked problem/family feature rows plus selection parameters and selected IDs | small curated MIC instance-selection input | 256 KiB total |

`tuned_nsgaii_resolved.json` is a `SCIENTIFIC_SOURCE_INPUT`: the fully resolved
configuration selected by the archived tuning campaign. Its budget is 16 KiB;
raw studies and histories are not retained in the product tree.

Any new reference result must add its exact path, generation command, source
commit, schema, expected use and size budget here before it is tracked.
