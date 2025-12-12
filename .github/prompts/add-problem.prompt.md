# Add New Problem

Use this prompt when adding a new benchmark or real-world optimization problem to VAMOS.

## Task
Add a new problem named `{PROBLEM_NAME}` with `{N_OBJ}` objectives and `{ENCODING}` encoding.

## Steps

1. **Create problem file** in appropriate location:
   - Benchmark: `src/vamos/problem/{problem_name}.py`
   - Real-world: `src/vamos/problem/real_world/{problem_name}.py`

2. **Implement ProblemProtocol**
   ```python
   class {ProblemClass}:
       def __init__(self, n_var: int, n_obj: int = 2):
           self.n_var = n_var
           self.n_obj = n_obj
           self.xl = np.zeros(n_var)  # lower bounds
           self.xu = np.ones(n_var)   # upper bounds
       
       def evaluate(self, X: np.ndarray) -> np.ndarray:
           # X: (N, n_var) → F: (N, n_obj)
           ...
   ```

3. **Register in specs**
   Add to `src/vamos/problem/registry/specs.py`:
   ```python
   ProblemSpec(
       key="{problem_key}",
       label="{Problem Label}",
       default_n_var=30,
       default_n_obj=2,
       allow_n_obj_override=False,  # True for DTLZ/WFG style
       factory=lambda n_var, n_obj: {ProblemClass}(n_var, n_obj),
       encoding="{encoding}",
   )
   ```
   Add to `PROBLEM_SPECS` dict.

4. **Add reference front** (if known)
   - Place CSV in `data/reference_fronts/{PROBLEM_KEY}.csv`
   - Format: one point per line, objectives space-separated

5. **Add tests**
   - `tests/test_new_benchmarks.py` or new file
   - Test evaluate shapes, bounds, finite outputs

## Checklist
- [ ] Implements `n_var`, `n_obj`, `xl`, `xu`, `evaluate()`
- [ ] Vectorized evaluate (handles batch X)
- [ ] Registered in PROBLEM_SPECS
- [ ] Works with `make_problem_selection("{problem_key}")`
- [ ] Test for shapes and finite outputs
- [ ] Reference front added (if available)
- [ ] Docstring with problem description/reference
