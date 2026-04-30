# Fe2S2 Ox Full Reproduction Guide

This guide reproduces the oxidized `Fe2S2(SCH3)4^{2-}` benchmark that is now
validated in this repository.

Scope:
- `APEX_CAS`: `prepare -> scf -> buildcas -> fcidump -> testcas`
- `APEX_Filter`: `load -> enumerate -> uhf -> ccsd -> ccsd-t -> ccsdt -> dmrg-basis -> dmrg -> extrapolate -> report`

If you want to avoid overwriting the committed benchmark artifacts, run the
commands in a fresh clone or in a scratch copy of `examples/fe2s2/`.

## 0. Files You Need

Before starting, make sure these files exist:

- `examples/fe2s2/inputs/fe2s2.xyz`
- `examples/fe2s2/inputs/fe2s2_cas_settings.yaml`
- `examples/fe2s2/inputs/fe2s2_filter_settings.yaml`
- `examples/fe2s2/inputs/fe2s2_cluster_info.yaml`
- `examples/fe2s2/filter_session/method_controls.yaml`

The committed repository already contains these files. If you are starting from
a fresh structure-only source, use `apex-cas prepare` to regenerate the cluster
annotation files first.

## 1. Prepare The Cluster Metadata

`APEX_CAS prepare` is the authoritative way to generate or validate
`cluster_info.yaml`.

Command:

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

What it produces:

- `examples/fe2s2/inputs/fe2s2_cluster_info_draft.csv`
- `examples/fe2s2/inputs/fe2s2_structure_labeled.png`
- `examples/fe2s2/inputs/fe2s2_cluster_info.yaml` when finalized

What to confirm or edit:

- review the draft CSV annotations
- verify the metal labels, bridge atoms, and terminal ligands
- keep the finalized `cluster_info.yaml` as the authority for the rest of the workflow

If you need to regenerate the authoritative file after editing the draft CSV,
run:

```bash
apex-cas prepare examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml \
  --finalize
```

## 2. Run APEX_CAS SCF

Command:

```bash
apex-cas scf examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

What it produces:

- `examples/fe2s2/outputs/scf/C4H12Fe2S6_uks_BP86_tzp-dkh.chk`
- `examples/fe2s2/outputs/scf/C4H12Fe2S6_uks_BP86_tzp-dkh_scf_info.json`
- `examples/fe2s2/outputs/scf/C4H12Fe2S6_uks_BP86_tzp-dkh_cas_info.json`

What to confirm:

- SCF convergence
- target charge and spin
- the high-spin UHF/UKS reference is chemically sensible

## 3. Build The Active Space

Command:

```bash
apex-cas buildcas examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_cas_settings.yaml
```

What it produces:

- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_orbital_report.md`
- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_selection.txt`
- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_noon_plot.png`
- `examples/fe2s2/outputs/orbitals/C4H12Fe2S6_uks_BP86_tzp-dkh_cas_data.h5`

What to confirm or edit:

- inspect the orbital report and NOON plot
- make sure the active orbitals are chemically reasonable
- if needed, edit `*_selection.txt` before generating FCIDUMP

For this benchmark, the committed selection already matches the validated run.

## 4. Generate FCIDUMP

Command:

```bash
apex-cas fcidump --case-dir examples/fe2s2
```

What it produces:

- `examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh`
- `examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh.ecore`
- `examples/fe2s2/outputs/fcidump/C4H12Fe2S6_uks_BP86_tzp-dkh_fcidump_info.json`

What to confirm:

- the active space is `(20o,30e)`
- `ECORE` sidecar is present
- the FCIDUMP stem is the one used by the downstream filter session

## 5. Optional DMRG Smoke Test In APEX_CAS

Command:

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500
```

Optional SU2 version:

```bash
apex-cas testcas examples/fe2s2/outputs/fcidump/FCIDUMP.C4H12Fe2S6_uks_BP86_tzp-dkh -M 500 --symm su2
```

What it produces:

- `examples/fe2s2/outputs/fcidump/dmrg/C4H12Fe2S6_uks_BP86_tzp-dkh_dmrg_info.json`
- `examples/fe2s2/outputs/fcidump/dmrg/C4H12Fe2S6_uks_BP86_tzp-dkh_dmrg_results.h5`
- `examples/fe2s2/outputs/fcidump/dmrg/C4H12Fe2S6_uks_BP86_tzp-dkh_noon_plot.png`

What to confirm:

- the DMRG smoke test energy is sensible
- the NOON pattern is reasonable

## 6. Load The Filter Session

Command:

```bash
apex-filter load \
  --config examples/fe2s2/inputs/fe2s2_filter_settings.yaml \
  --session examples/fe2s2/filter_session
```

What it produces:

- `examples/fe2s2/filter_session/session.json`
- `examples/fe2s2/filter_session/step1_load/cas_arrays.npz`
- `examples/fe2s2/filter_session/step1_load/cas_meta.json`
- `examples/fe2s2/filter_session/step1_load/cluster_info.json`
- `examples/fe2s2/filter_session/step1_load/fcidump_ref.json`
- `examples/fe2s2/filter_session/step1_load/settings.json`
- `examples/fe2s2/filter_session/method_controls.yaml`

What to confirm or edit:

- verify the resolved FCIDUMP path
- verify `cluster_info.yaml` was picked up
- inspect `filter_session/method_controls.yaml` before downstream steps

For this benchmark, `filter_settings.yaml` is only the step-1 bootstrap file.
The numerical controls from step 2 onward live in `method_controls.yaml`.

## 7. Enumerate Electronic Configurations

Command:

```bash
apex-filter enumerate --session examples/fe2s2/filter_session
```

What it produces:

- `examples/fe2s2/filter_session/step2_enumerate/enumeration.json`
- `examples/fe2s2/filter_session/step2_enumerate/enumeration_layers.json`
- `examples/fe2s2/filter_session/step2_enumerate/selection_candidates.csv`
- `examples/fe2s2/filter_session/step2_enumerate/selection_worklist.csv`
- `examples/fe2s2/filter_session/step2_enumerate/selection_guide.md`

What to confirm or edit:

- confirm the oxidized Fe2S2 family stack is correct
- after mirror-state verification, keep one representative state for the main benchmark ladder
- for this guide, the representative label is `Fe1↑Fe2↓|2xFe(III)|d:none`

The enumeration controls come from `examples/fe2s2/filter_session/method_controls.yaml`.
If you want to change the size of the enumeration, adjust the `enumerate` section
there and rerun `apex-filter enumerate`.

## 8. Run UHF

Command:

```bash
apex-filter uhf \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step2_enumerate/selection_worklist.csv"
```

What it produces:

- `examples/fe2s2/filter_session/step3_uhf/uhf_summary.json`
- `examples/fe2s2/filter_session/step3_uhf/results/*_uhf.npz`
- `examples/fe2s2/filter_session/step3_uhf/results/*_uhf.h5`
- `examples/fe2s2/filter_session/step3_uhf/results/*_post_scf_observables.json`

What to confirm:

- `converged = true`
- `E_total` matches the `Fe2S2` oxidized benchmark table
- `s2`, `two_s`, and the Fe-site spin observables are sensible

## 9. Run UCCSD

Command:

```bash
apex-filter ccsd \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step3_uhf/selection_worklist.csv"
```

What it produces:

- `examples/fe2s2/filter_session/step4_ccsd/ccsd_summary.json`
- `examples/fe2s2/filter_session/step4_ccsd/scripts/*_ccsd_results.npz`
- `examples/fe2s2/filter_session/step4_ccsd/scripts/*_post_scf_observables.json`

What to confirm:

- `E_total` is within benchmark tolerance of Chan Table 5
- `s2`, `two_s`, `two_sz_fe1`, `two_sz_fe2` are written to the summary and sidecar JSON

## 10. Run UCCSD(T)

Command:

```bash
apex-filter ccsd-t \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step4_ccsd/selection_worklist.csv"
```

What it produces:

- `examples/fe2s2/filter_session/step5_ccsd_t/ccsd_t_summary.json`
- `examples/fe2s2/filter_session/step5_ccsd_t/scripts/*_ccsd_t_results.npz`
- `examples/fe2s2/filter_session/step5_ccsd_t/scripts/*_post_scf_observables.json`

What to confirm:

- energy and spin observables remain in the same benchmark band as Chan

## 11. Run UCCSDT

Before this step, set the HAST-UCC environment if needed:

```bash
export PYTHONPATH=/Users/snh/hast-ucc:$PYTHONPATH
```

Command:

```bash
apex-filter ccsdt \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step5_ccsd_t/selection_worklist.csv"
```

What it produces:

- `examples/fe2s2/filter_session/step6_ccsdt/ccsdt_summary.json`
- `examples/fe2s2/filter_session/step6_ccsdt/scripts/*_ccsdt_results.npz`
- `examples/fe2s2/filter_session/step6_ccsdt/scripts/*_ccsdt_results.h5`
- `examples/fe2s2/filter_session/step6_ccsdt/scripts/*_post_scf_observables.json`

What to confirm:

- `energy` is close to Chan Table 5 `UCCSDT`
- `observables_complete = true`
- `lambda_converged = true`
- the `.h5` checkpoint is present so the observable stage can be restarted if needed

## 12. Prepare The DMRG Basis

Command:

```bash
apex-filter dmrg-basis \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step6_ccsdt/selection_worklist.csv"
```

What it produces:

- `examples/fe2s2/filter_session/step7_dmrg_basis/dmrg_basis_summary.json`
- `examples/fe2s2/filter_session/step7_dmrg_basis/dmrg_basis_qc.json`
- `examples/fe2s2/filter_session/step7_dmrg_basis/dmrg_basis_qc.csv`
- `examples/fe2s2/filter_session/step7_dmrg_basis/results/*_dmrg_basis.npz`
- `examples/fe2s2/filter_session/step7_dmrg_basis/results/*_dmrg_basis.h5`

What to confirm:

- the basis QA metrics are healthy
- the basis labels and ordering are stable
- the benchmark run keeps the representative state only

The committed `method_controls.yaml` already contains the validated Fe2S2 oxidized benchmark settings for this stage.

## 13. Run DMRG

Command:

```bash
apex-filter dmrg \
  --session examples/fe2s2/filter_session \
  --pick "file examples/fe2s2/filter_session/step7_dmrg_basis/selection_worklist.csv"
```

What it produces:

- `examples/fe2s2/filter_session/step8_dmrg/dmrg_summary.json`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_dmrg.npz`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_dmrg.h5`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_dmrg.log`
- `examples/fe2s2/filter_session/step8_dmrg/results/*_scratch`

What to confirm:

- the bond-dimension ladder is the validated Fe2S2 set
- the `M=100..2400` energies decrease smoothly
- `M=2000` and `M=2400` converge
- the `.h5` files include the schedule, diagnostics, and `2pdm`

For this benchmark, the DMRG step is used as an energy benchmark ladder.
Spin-resolved observables are intentionally not required here.

## 14. Extrapolate To Infinite DMRG Bond Dimension

Command:

```bash
apex-filter extrapolate --session examples/fe2s2/filter_session
```

What it produces:

- `examples/fe2s2/filter_session/step9_extrapolate/dmrg_extrapolation_summary.json`

What to confirm:

- the fit uses the converged `M=100..2400` ladder
- the extrapolated value is close to the Chan reference

## 15. Generate The Final Report

Command:

```bash
apex-filter report --session examples/fe2s2/filter_session
```

What it produces:

- `examples/fe2s2/filter_session/step10_report/final_summary.json`
- `examples/fe2s2/filter_session/step10_report/final_report.md`

What to confirm:

- the final ranking is present
- the report includes the `CCSDT + DMRG consensus` line
- the benchmark tables in `examples/fe2s2/chan_ref/` are consistent with the report

## 16. Optional Higher-Order Branch

These steps are not required to reproduce the main Fe2S2 oxidized benchmark,
but they remain available in the session:

- `apex-filter fno-uccsdtq`
- `apex-filter cc-composite`

For the current benchmark reproduction guide, you can stop after `report`.

## 17. Reference Files

The main benchmark comparison artifacts are:

- [examples/fe2s2/chan_ref/fe2s2_oxidized_apex_vs_chan2026_tables.md](chan_ref/fe2s2_oxidized_apex_vs_chan2026_tables.md)
- [examples/fe2s2/chan_ref/fe2s2_oxidized_apex_vs_chan2026_energy_table.csv](chan_ref/fe2s2_oxidized_apex_vs_chan2026_energy_table.csv)
- [examples/fe2s2/chan_ref/fe2s2_oxidized_apex_vs_chan2026_observables_table.csv](chan_ref/fe2s2_oxidized_apex_vs_chan2026_observables_table.csv)

If you only need the final benchmark numbers, use those tables directly.
