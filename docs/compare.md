# Compare Guide

This document describes the standalone compare tooling kept outside the
production mainline.

The production workflow remains compute-only:

- `APEX_CAS`
  - `prepare -> scf -> buildcas -> fcidump -> testcas`
- `APEX_Filter`
  - `load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report`

Compare tools are validation-side utilities. They are intended for retained
artifact checks, rerun validation, and benchmark/regression analysis.

## Recommended Entry Point

Use:

```bash
python scripts/compare_artifacts.py FILE1 FILE2
```

Examples:

```bash
python scripts/compare_artifacts.py ref.FCIDUMP new.FCIDUMP
python scripts/compare_artifacts.py ref_uhf.h5 new_uhf.h5
python scripts/compare_artifacts.py ref_dmrg_basis.npz new_dmrg_basis.npz --format json
```

## Supported Artifact Types

Current type-aware support includes:

- `FCIDUMP`
  - eigenvalue-aware one-electron comparison
  - two-electron RMS / max deltas
  - `ECORE` delta
- `*.h5` / `*.hdf5`
  - step3 UHF HDF5
  - step7 DMRG-basis HDF5
  - step8 DMRG HDF5
  - `APEX_CAS testcas` DMRG HDF5
  - generic structural HDF5 fallback
- `*.npz`
  - key-aware array comparison
- `*.json`
  - flattened scalar-path comparison

## Comparison Philosophy

For matrix-like and tensor-like artifacts, the compare layer prefers
representation-robust signals when possible.

Examples:

- `1-RDM`
  - elementwise delta
  - eigenspectrum delta
  - trace / `trace(gamma^2)`
  - basis-rotation-likely heuristic
- basis states
  - active-subspace match
  - best-overlap matching
  - ordering / pairing differences
- `2PDM`
  - elementwise delta
  - pair-space spectrum
  - trace-like invariants

This helps distinguish:

- true physical differences
- basis / ordering / pairing representation differences

## Python API

For programmatic use:

```python
from shared.comparison import compare_artifacts

result = compare_artifacts("ref_uhf.h5", "new_uhf.h5")
```

Useful lower-level entry points include:

- `compare_fcidumps(...)`
- `compare_density_matrices(...)`
- `compare_basis_states(...)`
- `compare_two_particle_density_tensors(...)`
- `compare_matrix_entries(...)`
- `compare_matrix_spectra(...)`

## Fe2S2-Specific Report

The repository also keeps:

```bash
python scripts/compare_fe2s2_runs.py --current examples/fe2s2 --baseline /path/to/fe2s2
```

That script remains Fe2S2-specific. For general artifact-level comparison, use
`scripts/compare_artifacts.py`.
