# Run artifact contract examples

These sanitized fixtures illustrate the frozen v1 document shapes. They are
documentation artifacts, not outputs from a real person or machine. IDs,
timestamps, versions, labels, and environment values are fixed examples.

Contents:

- `nsgaii-success/`: successful deterministic built-in NSGA-II run;
- `moead-success/`: successful MOEA/D run with explicit reference directions;
- `failed-run/`: failure before numerical arrays exist;
- `custom-manual/`: stored result from a notebook-local custom problem that is
  safe to load but cannot be replayed automatically;
- `compatibility-report.json`: comparison that downgrades an exact declaration
  to compatible in a different environment.

The three result-bearing directories contain small `result.npz` files using
only numerical arrays and no object dtype or pickle. Manifest artifact byte
lengths and SHA-256 digests match the committed files. Manifest self-hashes use
the canonicalization rule from the contract with
`integrity.manifest_sha256` omitted during hashing.

All artifact references use relative POSIX paths. No example contains a secret,
personal path, hostname, account, or executable source payload.
