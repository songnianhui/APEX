# Fe4S4H4 Validation Checklist

This checklist assumes the target case directory is:

`examples/fe4s4h4`

## 1. APEX_CAS compute

Run:

```bash
apex-cas compute examples/fe4s4h4/inputs/fe4s4h4.xyz \
  --cas-settings examples/fe4s4h4/inputs/fe4s4h4_cas_settings.yaml \
  --case-dir examples/fe4s4h4
```

Expected immediate console metadata:

- `Structure: H4Fe4S4 (charge=-2, S=0.0)`
- full-cluster symmetry close to `D2`
- metal-framework symmetry close to `Td`
- reduction symmetry `C3`

Expected output files after success:

- `examples/fe4s4h4/outputs/scf/*.chk`
- `examples/fe4s4h4/outputs/orbitals/cas_data.h5`
- `examples/fe4s4h4/outputs/orbitals/orbital_report.yaml`
- `examples/fe4s4h4/outputs/orbitals/noon_plot.png`

## 2. FCIDUMP generation

Run:

```bash
apex-cas fcidump --case-dir examples/fe4s4h4
```

Expected output:

- `examples/fe4s4h4/outputs/fcidump/FCIDUMP*`

## 3. APEX_Filter session bootstrap

Run:

```bash
apex-filter load \
  --config examples/fe4s4h4/inputs/fe4s4h4_filter_settings.yaml \
  --session examples/fe4s4h4/filter_session
```

Expected outcome:

- session directory `examples/fe4s4h4/filter_session`
- `step1_load` completed
- loaded cluster metadata should carry:
  - `family_scheme = literature_fe4s4_cubane`
  - `config_reduction_mode = none`

## 4. Enumeration baseline

Run:

```bash
apex-filter enumerate --session examples/fe4s4h4/filter_session
```

Expected layer counts:

- `Raw spin patterns = 6`
- `Spin families = 3`
- `Spin x oxidation = 24`
- `Spin x oxidation x d = 600`
- `Total configs (saved) = 600`

Expected metadata in step output:

- `family_scheme = literature_fe4s4_cubane`
- `config_reduction_mode = none`

Expected artifacts:

- `step2_enumerate/enumeration.json`
- `step2_enumerate/enumeration_layers.json`
- `step2_enumerate/selection_guide.md`
- `step2_enumerate/selection_candidates.csv`
- `step2_enumerate/selection_worklist.csv`

## 5. Minimal downstream smoke test

Recommended first pass:

```bash
apex-filter uhf --session examples/fe4s4h4/filter_session --pick "top-per-family 2"
```

After that, inspect:

- `step3_uhf/uhf_summary.json`
- convergence rate
- per-family energy spread

If UHF looks healthy, continue:

```bash
apex-filter ccsd   --session examples/fe4s4h4/filter_session --pick "top-per-family 2"
apex-filter ccsd-t --session examples/fe4s4h4/filter_session --pick "top-per-family 1"
```

## 6. What to compare against

For this example, the key consistency checks are:

- `Fe4S4H4` and `Fe4S4(SCH3)4` should now share the same family scheme:
  - `literature_fe4s4_cubane`
- They should therefore both report:
  - `Spin families = 3`
- But they should differ at the fully expanded guess layer:
  - `Fe4S4(SCH3)4`: `24`
  - `Fe4S4H4`: `600`
