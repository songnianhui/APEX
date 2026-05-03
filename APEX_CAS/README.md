# APEX_CAS

`APEX_CAS` is the maintained active-space preparation package inside `APEX`.
Its V1.0.0 workflow is:

```text
prepare -> scf -> buildcas -> fcidump -> testcas
```

It is responsible for:

- generating authoritative `cluster_info.yaml`
- running the high-spin SCF reference
- constructing the active-space orbital state
- writing `FCIDUMP` and `ECORE`
- optionally running a small DMRG smoke test

**[中文文档 / Chinese Documentation](README_CN.md)**

## Installation

```bash
cd APEX_CAS
pip install -e .
```

## Workflow Overview

### 1. `prepare`

`prepare` is the only supported route for generating cluster metadata used by
the rest of the workflow.

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

This writes:

- `*_cluster_info_draft.csv`
- `*_structure_labeled.png`
- finalized `*_cluster_info.yaml` after `--finalize`

The intended use is:

1. generate the draft CSV and labeled structure image
2. review and correct the draft if needed
3. rerun with `--finalize`
4. treat finalized `cluster_info.yaml` as the authority for all later `APEX_CAS` and `APEX_Filter` steps

### 2. `scf`

```bash
apex-cas scf examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

This stage runs the high-spin `UKS/UHF` reference and writes:

- `outputs/scf/*.chk`
- `outputs/scf/*_scf_info.json`
- `outputs/scf/*_cas_info.json`

### 3. `buildcas`

```bash
apex-cas buildcas examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

This stage restores the SCF state, builds the active-space orbital state, and writes:

- `outputs/orbitals/*_cas_data.h5`
- `outputs/orbitals/*_selection.txt`
- `outputs/orbitals/*_orbital_report.md`
- `outputs/orbitals/*_noon_plot.png`

If you need a one-command convenience route, `compute` remains available, but
the recommended documented path is still `scf -> buildcas`.

### 4. `fcidump`

```bash
apex-cas fcidump --case-dir examples/fe2s2
```

This consumes the saved CAS state and selection file and writes:

- `outputs/fcidump/FCIDUMP.*`
- `outputs/fcidump/FCIDUMP.*.ecore`
- `outputs/fcidump/*_fcidump_info.json`

The maintained `fcidump` mainline only generates the Hamiltonian artifacts.
Reference comparisons are validation-side actions and are no longer invoked by
the production `apex-cas fcidump` command.

### 5. `testcas`

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500
```

This is an optional DMRG smoke test on the generated active-space Hamiltonian.

## Quick Example

For the validated oxidized Fe2S2 benchmark:

- [../docs/example.md](/Users/snh/Projects/APEX/docs/example.md)
- [../examples/fe2s2/example.md](/Users/snh/Projects/APEX/examples/fe2s2/example.md)

## CLI Summary

### `apex-cas prepare`

```bash
apex-cas prepare structure.xyz --case-dir <case_dir> --cas-settings <yaml> [--finalize]
```

Use this to create and finalize cluster annotations. Finalized
`cluster_info.yaml` is the authority used downstream.

### `apex-cas scf`

```bash
apex-cas scf structure.xyz --case-dir <case_dir> --cas-settings <yaml>
```

Runs the SCF reference only.

### `apex-cas buildcas`

```bash
apex-cas buildcas structure.xyz --case-dir <case_dir> --cas-settings <yaml>
```

Builds the active-space state from the saved SCF checkpoint.

### `apex-cas compute`

```bash
apex-cas compute structure.xyz --case-dir <case_dir> --cas-settings <yaml>
```

Convenience wrapper for `scf -> buildcas`. Kept for workflow convenience, not
as the primary documented route.

### `apex-cas fcidump`

```bash
apex-cas fcidump --case-dir <case_dir>
```

Writes the active-space Hamiltonian in FCIDUMP format.

### `apex-cas testcas`

```bash
apex-cas testcas FCIDUMP_PATH -M 500 [--symm su2]
```

Runs a small DMRG test on the generated FCIDUMP.

## Configuration

The primary configuration file is:

- `shared/config/cas_settings_template.yaml`

Example case settings:

- `examples/fe2s2/inputs/fe2s2_cas_settings.yaml`

These control SCF method, functional, basis provenance, relativistic settings,
solvation options, convergence thresholds, and orbital-construction behavior.

## Data Authority

For V1.0.0, the intended authority chain is:

1. structure file
2. `apex-cas prepare`
3. finalized `cluster_info.yaml`
4. `scf`
5. `buildcas`
6. `fcidump`

Later stages are expected to consume saved state, not silently reconstruct new
cluster metadata.

## Fe2S2 Rerun Note

For the current repository layout:

- `examples/fe2s2/` is the fresh rerun working case
- `examples/fe2s2_bk2/` is the retained local baseline snapshot

The canonical Fe2S2 walkthrough now assumes that downstream filter bootstrap
files may be regenerated during the rerun rather than committed in advance.
