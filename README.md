# APEX

English | [中文](README_CN.md)

`APEX` is a session-based workflow for transition-metal cluster calculations.
The current V1.0.0 architecture is split into two packages:

- `APEX_CAS`: prepare structure metadata, run SCF, build the active space, and generate `FCIDUMP`
- `APEX_Filter`: start from `APEX_CAS` outputs and run the staged screening chain

The maintained canonical workflow is:

```text
APEX_CAS:
prepare -> scf -> buildcas -> fcidump -> testcas

APEX_Filter:
load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report
```

This production mainline is compute-only. Validation-side comparison and
analysis are intentionally kept separate from the runtime computation path.

Standalone compare tooling is documented in:

- [docs/compare.md](/Users/snh/Projects/APEX/docs/compare.md)

For the maintained Fe2S2 walkthrough, see:

- [docs/example.md](/Users/snh/Projects/APEX/docs/example.md)
- [examples/fe2s2/README_step_by_step.md](/Users/snh/Projects/APEX/examples/fe2s2/README_step_by_step.md)

## Repository Layout

```text
APEX/
├── APEX_CAS/          # Active-space construction and FCIDUMP generation
├── APEX_Filter/       # Session-based active-space screening
├── shared/            # Shared data models, parsers, config templates, templates
├── examples/          # Reproducible example cases
├── docs/              # User-facing workflow documentation
└── plans/             # Recovery and implementation plans
```

## Fe2S2 Case Layout

The maintained release example is:

- `examples/fe2s2/`
  - a complete mainline case directory
  - used to demonstrate the production workflow from `prepare` through `report`

## Core Principles

- `cluster_info.yaml` is the authority for cluster annotations once finalized.
- `APEX_CAS prepare` is the only supported route for generating that authority file.
- `filter_settings.yaml` is only the Step-1 bootstrap file for `APEX_Filter`.
- From `step2 enumerate` onward, numerical controls live in session-local `filter_session/method_controls.yaml`.
- On the active-space route, `APEX_Filter` methods operate on the Hamiltonian defined by the upstream `FCIDUMP`; they do not rebuild a new AO-basis problem from scratch.

## Quick Start

### 1. Prepare cluster metadata

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

If you edit the draft CSV, finalize it with:

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml \
  --finalize
```

### 2. Build the active-space Hamiltonian

```bash
apex-cas scf examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml

apex-cas buildcas examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml

apex-cas fcidump --case-dir examples/fe2s2
```

### 3. Start the filter session

```bash
cp shared/config/filter_settings_template.yaml examples/fe2s2/inputs/fe2s2_filter_settings.yaml

apex-filter load \
  --config examples/fe2s2/inputs/fe2s2_filter_settings.yaml \
  --session examples/fe2s2/filter_session
```

Then continue with:

```bash
apex-filter enumerate --session examples/fe2s2/filter_session
apex-filter uhf --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step2_enumerate/selection_worklist.csv"
apex-filter ccsd --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step3_uhf/selection_worklist.csv"
apex-filter ccsd-t --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step4_ccsd/selection_worklist.csv"
apex-filter ccsdt --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step5_ccsd_t/selection_worklist.csv"
apex-filter dmrg-basis --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step6_ccsdt/selection_worklist.csv"
apex-filter dmrg --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step7_dmrg_basis/selection_worklist.csv"
apex-filter extrapolate --session examples/fe2s2/filter_session
apex-filter report --session examples/fe2s2/filter_session
```

## Configuration Files

- `shared/config/cas_settings_template.yaml`
  - template for `APEX_CAS` SCF / active-space settings
- `shared/config/filter_settings_template.yaml`
  - step-1 bootstrap template for `APEX_Filter`
- `shared/config/method_controls_template.yaml`
  - template for step-local numerical controls copied into each filter session

## Current Mainline Scope

The maintained example case is oxidized `Fe2S2(SCH3)4^{2-}`. The repository is
organized around a clean V1.0.0 workflow with:

- shared authority in `shared/`
- no hidden fallback reconstruction of `cluster_info`
- session-local numerical control through `method_controls.yaml`
- `step8` DMRG routed through `pyscf_dmrgci_sz` when requested by the session controls

The higher-order `step11+` branch remains in the tree, but it is not part of
the closed V1.0.0 rerun / cleanup / authority-validation scope.

## Package READMEs

- [APEX_CAS/README.md](/Users/snh/Projects/APEX/APEX_CAS/README.md)
- [APEX_Filter/README.md](/Users/snh/Projects/APEX/APEX_Filter/README.md)
