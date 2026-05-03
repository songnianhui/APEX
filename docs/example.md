# Fe2S2 Ox Rerun Guide

This guide describes the current Fe2S2 oxidized rerun workflow for `APEX V1.0.0`.

Scope:
- `APEX_CAS`: `prepare -> scf -> buildcas -> fcidump -> testcas`
- `APEX_Filter`: `load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report`

Local directory roles:
- `examples/fe2s2/`
  - fresh rerun working case
- `examples/fe2s2_bk2/`
  - retained local baseline snapshot
- `APEX_bk/examples/fe2s2/`
  - historical APEX benchmark baseline
- `examples/fe2s2/chan_ref/`
  - Chan-facing comparison bundle

Use this guide together with:
- [examples/fe2s2/example.md](/Users/snh/Projects/APEX/examples/fe2s2/example.md)
- [plans/fe2s2_rerun_compare_guide_20260502.md](/Users/snh/Projects/APEX/plans/fe2s2_rerun_compare_guide_20260502.md)

Comparison policy during the rerun:
1. if `APEX_bk` has the corresponding artifact, compare to `APEX_bk`
2. otherwise compare structure/schema against `examples/fe2s2_bk2`
3. use `chan_ref` for the final benchmark-facing comparison report

## 0. Required Inputs

Before starting, make sure these files exist:

- `examples/fe2s2/inputs/fe2s2.xyz`
- `examples/fe2s2/inputs/fe2s2_cas_settings.yaml`

The following files are typically generated during the rerun:

- `examples/fe2s2/inputs/fe2s2_cluster_info.yaml`
- `examples/fe2s2/inputs/fe2s2_filter_settings.yaml`
- `examples/fe2s2/filter_session/method_controls.yaml`

Do not assume the fresh rerun directory begins with a committed full session.

## 1. Prepare Cluster Metadata

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

Produces:
- `examples/fe2s2/inputs/fe2s2_cluster_info_draft.csv`
- `examples/fe2s2/inputs/fe2s2_structure_labeled.png`

After reviewing the draft, finalize:

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml \
  --finalize
```

Confirm:
- metal labels, bridge atoms, and terminal ligands are correct
- finalized `cluster_info.yaml` is treated as the downstream authority

## 2. Run SCF

```bash
apex-cas scf examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

Produces:
- `outputs/scf/*.chk`
- `outputs/scf/*_scf_info.json`
- `outputs/scf/*_cas_info.json`

Confirm:
- SCF converges
- charge and spin are correct
- the high-spin reference is chemically sensible

## 3. Build the Active Space

```bash
apex-cas buildcas examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

Produces:
- `outputs/orbitals/*_orbital_report.md`
- `outputs/orbitals/*_selection.txt`
- `outputs/orbitals/*_noon_plot.png`
- `outputs/orbitals/*_cas_data.h5`
- `outputs/orbitals/*_orbital_gallery.html`
- `outputs/orbitals/*_orbital_gallery_server.py`

Confirm:
- orbital report and NOON plot are chemically sensible
- if needed, replace the auto-generated `*_selection.txt` with a validated retained selection before `fcidump`

## 4. Generate FCIDUMP

```bash
apex-cas fcidump --case-dir examples/fe2s2
```

Produces:
- `outputs/fcidump/FCIDUMP.*`
- `outputs/fcidump/FCIDUMP.*.ecore`
- `outputs/fcidump/*_fcidump_info.json`

Confirm:
- the active space matches the intended benchmark selection
- the `.ecore` sidecar is present

## 5. Optional APEX_CAS DMRG Smoke Test

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500
```

Optional SU2 version:

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500 --symm su2
```

Produces:
- `outputs/fcidump/dmrg/*_dmrg_info.json`
- `outputs/fcidump/dmrg/*_dmrg_results.h5`
- `outputs/fcidump/dmrg/*_noon_plot.png`

Confirm:
- smoke-test energy is sensible
- NOON pattern is reasonable

For matrix-like retained results, prefer spectral and invariant comparisons over
raw elementwise deltas.

## 6. Start the Filter Session

Create the bootstrap config if it is not already present:

```bash
cp shared/config/filter_settings_template.yaml examples/fe2s2/inputs/fe2s2_filter_settings.yaml
```

Then run:

```bash
apex-filter load \
  --config examples/fe2s2/inputs/fe2s2_filter_settings.yaml \
  --session examples/fe2s2/filter_session
```

Produces:
- `filter_session/session.json`
- `filter_session/step1_load/cas_arrays.npz`
- `filter_session/step1_load/cas_meta.json`
- `filter_session/step1_load/cluster_info.json`
- `filter_session/step1_load/fcidump_ref.json`
- `filter_session/step1_load/settings.json`
- `filter_session/method_controls.yaml`

Confirm:
- the resolved FCIDUMP path is correct
- `cluster_info.yaml` was picked up
- `method_controls.yaml` exists and will be the only numerical control surface from step 2 onward

`filter_settings.yaml` is only the step-1 bootstrap file. Downstream numerical
controls live in `filter_session/method_controls.yaml`.

## 7. Enumerate Configurations

```bash
apex-filter enumerate --session examples/fe2s2/filter_session
```

Produces:
- `step2_enumerate/enumeration.json`
- `step2_enumerate/enumeration_layers.json`
- `step2_enumerate/selection_candidates.csv`
- `step2_enumerate/selection_worklist.csv`
- `step2_enumerate/selection_guide.md`

For the benchmark ladder, keep the validated representative state after
mirror-state verification.

## 8. Run the Maintained Filter Ladder

```bash
apex-filter uhf --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step2_enumerate/selection_worklist.csv"
apex-filter ccsd --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step3_uhf/selection_worklist.csv"
apex-filter ccsd-t --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step4_ccsd/selection_worklist.csv"
apex-filter ccsdt --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step5_ccsd_t/selection_worklist.csv"
apex-filter dmrg-basis --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step6_ccsdt/selection_worklist.csv"
apex-filter dmrg --session examples/fe2s2/filter_session --pick "file examples/fe2s2/filter_session/step7_dmrg_basis/selection_worklist.csv"
apex-filter extrapolate --session examples/fe2s2/filter_session
apex-filter report --session examples/fe2s2/filter_session
```

Key outputs by step:
- `step3_uhf/uhf_summary.json`
- `step4_ccsd/ccsd_summary.json`
- `step5_ccsd_t/ccsd_t_summary.json`
- `step6_ccsdt/ccsdt_summary.json`
- `step7_dmrg_basis/dmrg_basis_summary.json`
- `step7_dmrg_basis/dmrg_basis_qc.{json,csv}`
- `step8_dmrg/dmrg_summary.json`
- `step9_extrapolate/dmrg_extrapolation_summary.json`
- `step10_report/final_summary.json`
- `step10_report/final_report_energies.csv`
- `step10_report/final_report_observables.csv`

Confirm:
- step summaries are produced
- selected labels are the intended benchmark route
- the DMRG ladder is smooth and the extrapolated value is sensible
- the final report includes the expected consensus/ranking lines in
  `final_summary.json` and `final_report_energies.csv`

The mainline `report` step is compute-only. Detailed benchmark-facing compare
reports are generated separately and are not part of the production workflow.

For this V1.0.0 guide, the maintained mainline stops after `report`.

## 9. Out of Scope for the Closed V1.0.0 Mainline

The following remain available in the tree, but are not part of the closed
V1.0.0 rerun / cleanup / authority-validation scope:

- `apex-filter fno-uccsdtq`
- `apex-filter cc-composite`
- broader `step11+` higher-order follow-on work

## 10. Final Benchmark Comparison

Use the fresh rerun artifacts together with the compare guide and `chan_ref`
bundle to produce the final benchmark-facing comparison report.

For the current repository state, the retained validation-side compare report is:

- [examples/fe2s2/fe2s2_rerun_compare_report_20260503.md](/Users/snh/Projects/APEX/examples/fe2s2/fe2s2_rerun_compare_report_20260503.md)

For the detailed per-step comparison checklist, use:

- [plans/fe2s2_rerun_compare_guide_20260502.md](/Users/snh/Projects/APEX/plans/fe2s2_rerun_compare_guide_20260502.md)
