# FeMoco Chan Benchmark References

This directory stores Chan-reference benchmark artifacts for the `FeMoco` LLDUC model.

## Files

- [femoco_chan2026_llduc_benchmark.json](/Users/snh/Projects/APEX/examples/femoco/chan_ref/femoco_chan2026_llduc_benchmark.json)
  - direct benchmark values extracted from the 2026 FeMoco paper
  - includes:
    - summary consensus energies
    - selected-state `UHF / UCCSD / UCCSDT` values for `BS7-C`, `BS8-E`, `BS8-F`
    - `UDMRG` values for `BS8-E` from Supplementary Table 10
    - corresponding `⟨S²⟩`

- [FeMoco_orbital_selection_setup.yaml](/Users/snh/Projects/APEX/examples/femoco/chan_ref/FeMoco_orbital_selection_setup.yaml)
- [chan_FeMoco_orbital_setup.yaml](/Users/snh/Projects/APEX/examples/femoco/chan_ref/chan_FeMoco_orbital_setup.yaml)
  - orbital-selection / setup reference material already present in the repo

## Primary source

- [2026-Classical solution of the FeMo-cofactor model to chemical accuracy and its implications.pdf](/Users/snh/Projects/APEX/ref/2026-Classical%20solution%20of%20the%20FeMo-cofactor%20model%20to%20chemical%20accuracy%20and%20its%20implications.pdf)

Relevant literature sections for the current JSON:

- main text: energy-summary discussion around Section 4.6
- supplementary:
  - `Supplementary Table 9`
  - `Supplementary Table 10`

## Notes

- `Supplementary Table 9` and `Supplementary Table 10` report energies shifted by `-22140.0 Hartrees`.
- In the JSON file, both the shifted values and recovered absolute total energies are stored.
- This is not yet a complete dump of every FeMoco benchmark number in the 2026 paper; it is the first structured subset that is directly reusable for APEX benchmark comparisons.
