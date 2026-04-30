# Fe2S2 Oxidized: APEX vs Chan 2026 Tables

This note collects the directly comparable oxidized `Fe2S2(SCH3)4^{2-}` benchmark
numbers in one place.

Primary Chan reference:
- [2026-Classical solution of the FeMo-cofactor model to chemical accuracy and its implications.pdf](/Users/snh/Projects/APEX/ref/2026-Classical%20solution%20of%20the%20FeMo-cofactor%20model%20to%20chemical%20accuracy%20and%20its%20implications.pdf)

Supporting Chan public reference:
- `Fe2S2` README values mirrored in:
  - [fe2s2_chan2026_oxidized_benchmark.json](/Users/snh/Projects/APEX/examples/fe2s2/chan_ref/fe2s2_chan2026_oxidized_benchmark.json)

Core energy used by Chan for the `(20o,30e)` model:
- `E_core = -4976.26532397 Eh`

## Energy Ladder: Step3-6 vs Chan Table 5

| APEX step | APEX theory | Our `E_total` (Eh) | Chan reference | Chan `E_total` (Eh) | `Our - Chan` (mEh) |
|---|---|---:|---|---:|---:|
| `step3` | `UHF` | `-5092.778046833884` | Table 5 `UHF` | `-5092.778015970000` | `-0.030864` |
| `step4` | `CCSD` | `-5092.845874256951` | Table 5 `UCCSD` | `-5092.845793970000` | `-0.080287` |
| `step5` | `CCSD(T)` | `-5092.861451756011` | Table 5 `UCCSD(T)` | `-5092.861361970000` | `-0.089786` |
| `step6` | `CCSDT` | `-5092.866305089044` | Table 5 `UCCSDT` | `-5092.866206970000` | `-0.098119` |

## DMRG Comparison

Current `APEX_Filter step8` benchmark route:
- `backend = pyscf_dmrgci_sz`
- `basis_mode = original_identity`
- `schedule_mode = benchmark`
- representative single BS state:
  - `Fe1↑Fe2↓|2xFe(III)|d:none`

`step8` DMRG ladder against Chan Table 7:

| Bond dimension | Our `E_total` (Eh) | Chan Table 7 `E_total` (Eh) | `Our - Chan` (mEh) |
|---:|---:|---:|---:|
| `100` | `-5092.862318636900` | `-5092.864229970000` | `+1.911333` |
| `200` | `-5092.868278923903` | `-5092.867121970000` | `-1.156954` |
| `400` | `-5092.870307300223` | `-5092.868767970000` | `-1.539330` |
| `600` | `-5092.870748485378` | `-5092.869405970000` | `-1.342515` |
| `800` | `-5092.870902878984` | `-5092.869723970000` | `-1.178909` |
| `1000` | `-5092.870965604963` | `-5092.869894970000` | `-1.070635` |
| `1200` | `-5092.870996551455` | `-5092.870126970000` | `-0.869581` |
| `1600` | `-5092.871019920727` | `-5092.870494970000` | `-0.524951` |
| `2000` | `-5092.871027879793` | `-5092.870663970000` | `-0.363910` |
| `2400` | `-5092.871031079445` | `-5092.870799970000` | `-0.231109` |

`step9` DMRG `D→∞` extrapolation:

| Our reference | Our `E_total` (Eh) | Chan reference | Chan `E_total` (Eh) | `Our - Chan` (mEh) | Note |
|---|---:|---|---:|---:|---|
| `step9 DMRG extrapolated` | `-5092.871060136653` | Table 5 `SA-DMRG[D=12k]` | `-5092.870932970000` | `-0.127167` | 10-point `D=100..2400` fit |
| `step9 DMRG extrapolated` | `-5092.871060136653` | Chan public README exact | `-5092.870933070000` | `-0.127067` | public same-system exact reference |

## Non-Energy Benchmark

The current APEX benchmark route now preserves the following non-energy
observables for `step3-5`:
- `⟨S²⟩`
- `2S`
- `2Sz(Fe1)`
- `2Sz(Fe2)`

For `2Sz(Fe)` we compare against the mirror Chan broken-symmetry state after a
global sign flip.

| APEX step | APEX theory | Observable | Our value | Chan reference | Chan value | `Our - Chan` |
|---|---|---|---:|---|---:|---:|
| `step3` | `UHF` | `⟨S²⟩` | `4.893026377727` | Table 5 `UHF ⟨S²⟩` | `4.893150` | `-1.23622273e-04` |
| `step3` | `UHF` | `2S` | `3.535648305700` | Table 5 `UHF 2⟨S⟩` | `3.535703` | `-5.46943000e-05` |
| `step3` | `UHF` | `2Sz(Fe1)` | `4.152152647689` | Table 5 `UHF 2Sz(Fe1)` | `-4.197729` | `+4.55763523e-02` |
| `step3` | `UHF` | `2Sz(Fe2)` | `-4.152950038001` | Table 5 `UHF 2Sz(Fe2)` | `4.200378` | `-4.74279620e-02` |
| `step4` | `UCCSD` | `⟨S²⟩` | `3.601070246689` | Table 5 `UCCSD ⟨S²⟩` | `3.601338` | `-2.67753311e-04` |
| `step4` | `UCCSD` | `2S` | `2.924828784387` | Table 5 `UCCSD 2⟨S⟩` | `2.924965` | `-1.36215613e-04` |
| `step4` | `UCCSD` | `2Sz(Fe1)` | `3.879400190395` | Table 5 `UCCSD 2Sz(Fe1)` | `-3.916799` | `+3.73988096e-02` |
| `step4` | `UCCSD` | `2Sz(Fe2)` | `-3.880450615359` | Table 5 `UCCSD 2Sz(Fe2)` | `3.919647` | `-3.91963846e-02` |
| `step5` | `UCCSD(T)` | `⟨S²⟩` | `3.104748611002` | Table 5 `UCCSD(T) ⟨S²⟩` | `3.104838` | `-8.93889983e-05` |
| `step5` | `UCCSD(T)` | `2S` | `2.663194568134` | Table 5 `UCCSD(T) 2⟨S⟩` | `2.663243` | `-4.84318659e-05` |
| `step5` | `UCCSD(T)` | `2Sz(Fe1)` | `3.774314074843` | Table 5 `UCCSD(T) 2Sz(Fe1)` | `-3.818287` | `+4.39729252e-02` |
| `step5` | `UCCSD(T)` | `2Sz(Fe2)` | `-3.775284870055` | Table 5 `UCCSD(T) 2Sz(Fe2)` | `3.821016` | `-4.57311299e-02` |
| `step6` | `UCCSDT` | `⟨S²⟩` | `2.720002081625` | Table 5 `UCCSDT ⟨S²⟩` | `2.719797` | `+2.05081625e-04` |
| `step6` | `UCCSDT` | `2S` | `2.446738795804` | Table 5 `UCCSDT 2⟨S⟩` | `2.446620` | `+1.18795804e-04` |
| `step6` | `UCCSDT` | `2Sz(Fe1)` | `3.716049340039` | Table 5 `UCCSDT 2Sz(Fe1)` | `-3.763482` | `+4.74326600e-02` |
| `step6` | `UCCSDT` | `2Sz(Fe2)` | `-3.717048978685` | Table 5 `UCCSDT 2Sz(Fe2)` | `3.766248` | `-4.91910213e-02` |

## Notes

- `step3-6` energy comparisons are same-system, same active-space benchmark comparisons against Chan Table 5.
- `step3-5` non-energy comparisons use the same representative broken-symmetry state on the APEX side and compare to Chan after mirror-state sign alignment for `2Sz(Fe)`.
- `step6` now has complete observables (`s2`, `two_s`, `two_sz_fe1`, `two_sz_fe2`) and a restart-capable `.h5` checkpoint.
- `step8/step9` use `pyscf_dmrgci_sz` rather than Chan’s unrestricted `BLOCK2` route, so the DMRG comparison is numerically useful but not a strict backend identity comparison.
- Current `step9` extrapolation from `D=100..2400` slightly overshoots Chan’s public exact value by about `0.127 mEh`; this is still a small same-system discrepancy.
