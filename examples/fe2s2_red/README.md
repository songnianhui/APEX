# Fe2S2 Reduced Benchmark Workspace

This directory is reserved for the formal `[Fe2S2(SCH3)4]^{3-}` benchmark workflow.

It is separated from `examples/fe2s2/` so that the oxidized and reduced benchmarks can
evolve independently while sharing the same Chan 2026 benchmark conventions.

## Planned structure

- `inputs/`
  - reduced benchmark inputs
- `outputs/`
  - `APEX_CAS` outputs
- `filter_session/`
  - `APEX_Filter` session data
- `chan_ref/`
  - Chan 2026 reduced benchmark reference values

## Reference

Primary benchmark literature:

- [2026-Classical solution of the FeMo-cofactor model to chemical accuracy and its implications.pdf](/Users/snh/Projects/APEX/ref/2026-Classical%20solution%20of%20the%20FeMo-cofactor%20model%20to%20chemical%20accuracy%20and%20its%20implications.pdf)

Relevant tables for the reduced `Fe2S2` model:

- Supplementary Table 6
- Supplementary Table 8
