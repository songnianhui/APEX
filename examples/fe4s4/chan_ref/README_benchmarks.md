# Fe4S4 Chan Benchmark References

This directory stores Chan-reference artifacts for the `Fe4S4(SCH3)4` benchmark family.

## Current local reference status

At present, the locally confirmed Chan-reference assets for `Fe4S4` are:

- [fe4s4](/Users/snh/Projects/APEX/examples/fe4s4/chan_ref/fe4s4)
  - FCIDUMP-like integral file
  - header confirms:
    - `NORB = 36`
    - `NELEC = 54`
    - `MS2 = 0`

- [fe4s4_chan2017_benchmark_inventory.json](/Users/snh/Projects/APEX/examples/fe4s4/chan_ref/fe4s4_chan2017_benchmark_inventory.json)
  - structured inventory of what is directly recoverable from the 2017 SP-MPS paper

## Literature source

Primary local paper:

- [2017-Spin-Projected Matrix Product States- Versatile Tool for Strongly Correlated Systems.pdf](/Users/snh/Projects/APEX/ref/2017-Spin-Projected%20Matrix%20Product%20States-%20Versatile%20Tool%20for%20Strongly%20Correlated%20Systems.pdf)

Relevant section:

- pages `30-32`
- Figure `10`

## What is directly extractable now

The current local literature supports the following benchmark facts:

- system:
  - `[Fe4S4(SCH3)4]^{2-}`
- active space:
  - `CAS(54e,36o)`
- benchmark workflow in the paper:
  - `24` broken-symmetry initial guesses
  - `SP-MPS(D=200)` singlet calculations from all 24 guesses
  - comparison of the `24` converged states by relative energy
  - validation against `state-averaged SA-MPS(D=2500)` for the lowest states
- qualitative benchmark conclusion:
  - the 24 states collapse into `3` distinct spin-spin correlation patterns
  - the lowest-energy pattern found by `SP-MPS(D=200)` matches the larger-`D` SA-MPS ground-state pattern

## What is not yet available as a clean numeric table

The current local sources do **not** yet provide a clean machine-readable table of:

- the `24` relative-energy values from Figure 10(b)
- the per-state spin-spin correlation matrix values
- absolute total energies
- `⟨S²⟩` tables analogous to the `Fe2S2` 2026 tables

So `Fe4S4` is currently in a different benchmark state from `Fe2S2`:

- `Fe2S2` already has table-grade JSON benchmark values
- `Fe4S4` currently has:
  - FCIDUMP metadata
  - figure-based benchmark inventory
  - qualitative literature conclusions

## Next step

To upgrade `Fe4S4` to the same level as `Fe2S2`, the next action should be:

- figure-digitize or otherwise recover the numeric relative-energy bars in Figure 10(b)
- and, if possible, recover the corresponding spin-spin correlation values per state
