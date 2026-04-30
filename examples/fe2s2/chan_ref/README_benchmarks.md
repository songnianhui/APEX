# Fe2S2 Chan Benchmark References

This directory stores Chan-reference artifacts for the `Fe2S2(SCH3)4` benchmark family.

Current scope:
- oxidized `Fe2S2` only

The reduced benchmark now lives in its own dedicated example directory:
- [examples/fe2s2_red/chan_ref](/Users/snh/Projects/APEX/examples/fe2s2_red/chan_ref)

## Files

- [fe2s2_chan2026_oxidized_benchmark.json](/Users/snh/Projects/APEX/examples/fe2s2/chan_ref/fe2s2_chan2026_oxidized_benchmark.json)
  - Chan 2026 Supplementary Tables 5 and 7
  - system: `[Fe2S2(SCH3)4]^{2-}`
  - model: `(20o,30e)`
  - observables:
    - `E_active`
    - recovered `E_total`
    - `⟨S²⟩`
    - `2⟨S⟩`
    - `2Sz(Fe1)`
    - `2Sz(Fe2)`

- [fe2s2_oxidized_apex_vs_chan2026_tables.md](/Users/snh/Projects/APEX/examples/fe2s2/chan_ref/fe2s2_oxidized_apex_vs_chan2026_tables.md)
  - APEX vs Chan tabulated oxidized benchmark comparison
  - includes:
    - `step3-6` energy ladder vs Chan Table 5
    - `step8 M=1000` DMRG comparison vs Chan Table 7 / SA-DMRG / README
    - currently available `⟨S²⟩` benchmark

- [fe2s2_oxidized_apex_vs_chan2026_energy_table.csv](/Users/snh/Projects/APEX/examples/fe2s2/chan_ref/fe2s2_oxidized_apex_vs_chan2026_energy_table.csv)
  - machine-readable energy comparison table for oxidized `Fe2S2`

## Source

Primary source:

- [2026-Classical solution of the FeMo-cofactor model to chemical accuracy and its implications.pdf](/Users/snh/Projects/APEX/ref/2026-Classical%20solution%20of%20the%20FeMo-cofactor%20model%20to%20chemical%20accuracy%20and%20its%20implications.pdf)

Relevant pages:

- `47-50`

Relevant tables:

- oxidized:
  - `Supplementary Table 5`
  - `Supplementary Table 7`

## Notes

- The paper states that the core energy for this active-space model is:
  - `E_core = -4976.26532397 Eh`
- Table energies are reported as active-space energies.
- `E_total` in the JSON files is recovered as:
  - `E_total = E_core + E_active`
