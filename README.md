# APEX

English | [中文](README_CN.md)

**APEX** — Automated Progressive Electronic structure eXploration for transition metal clusters.

## 1. Overview

High-accuracy quantum chemistry of transition metal clusters (DMRG, coupled cluster, etc.) requires careful manual setup: defining the active space, enumerating spin/electronic configurations, constructing orbital bases, and running hierarchical calculations. This project automates the entire workflow, from structure input to FCIDUMP integral file generation.

The canonical reference case is the **FeMo-cofactor** (Fe₇MoS₉C) of nitrogenase, following the workflow established by Li et al. (JCP, 2019) and Zhai et al. (2026):

```
Fe₇MoS₉C → (113e, 76o) active space → 35 spin isomers → 78,750 electronic configs
          → UHF/CCSD/CCSDT/CCSDTQ/DMRG filtering funnel → FCIDUMP → chemical accuracy
```

### What the Agent Does

Given a molecular structure file (XYZ/PDB) plus charge and spin:

1. **Structure Analysis** — Identifies metal centers, bridging atoms, terminal ligands, and approximate point group symmetry
2. **Active Space Construction** — Builds (n_electrons, n_orbitals) at minimal/standard/extended levels using knowledge-base rules
3. **Spin Isomer Enumeration** — Enumerates all collinear broken-symmetry spin isomers satisfying Σ(±Sᵢ) = target Sz, with optional symmetry reduction
4. **Electronic Configuration Enumeration** — Oxidation-first enumeration: assigns oxidation states first, then spin isomers, then d-orbital choices
5. **Filtering Funnel Design** — Designs a hierarchical protocol: UHF → UCCSD → UCCSD(T) → DMRG → CCSDTQ → DMRG
6. **Input File Generation** — Generates ready-to-run input scripts for PySCF, BLOCK2, HAST-UCC, ORCA, and Gaussian
7. **Energy Extrapolation** — DMRG D-extrapolation, CC composite energy, FNO extrapolation, and MP2 space correction
8. **Result Parsing & Verification** — Parses QC outputs (`.chk` and `.npz`), verifies convergence, validates oxidation states via population analysis
9. **Automatic Pipeline Execution** — End-to-end execution of the progressive filtering funnel with high-spin UHF deduplication and FCIDUMP generation

## 2. Complete Workflow

```
XYZ/PDB Input
     │
     ▼
┌─────────────────────────┐
│ 2.1 Structure Analysis   │  parse_structure() → ClusterInfo
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 2.2 Active Space Build   │  build_active_space() → ActiveSpace
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 2.3 Spin Isomers         │  enumerate_spin_isomers() + symmetry reduction
└──────────┬──────────────┘
           ▼
┌─────────────────────────────────────────────────────┐
│ 2.4 Electronic Configs (oxidation-first, v2 API)     │  generate_all_configs_v2()
└──────────┬──────────────────────────────────────────┘
           ▼
┌─────────────────────────┐
│ 2.5 Symmetry Reduction   │  reduce_configs_by_symmetry()
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ 2.6 Filtering Funnel     │  design_filtering_funnel() → FilteringPlan
└──────────┬──────────────┘
           ▼
┌──────────────────────────────────────────────────┐
│ 2.7 Progressive Pipeline (UHF→UCCSD→DMRG→CCSDTQ) │  run_pipeline()
└──────────┬───────────────────────────────────────┘
           ▼
┌─────────────────────────┐
│ 2.8 FCIDUMP Generation   │  generate_fcidump() / generate_fcidump_for_results()
└─────────────────────────┘
```

### 2.1 Structure Input

`parse_structure()` reads an XYZ or PDB file via ASE and returns a `ClusterInfo` dataclass. It identifies transition metal centers (30 metals supported), bridging atoms, terminal ligands, and detects approximate point group symmetry (C3, C4).

```python
from agent.structure_analyzer import parse_structure

cluster_info = parse_structure("structure.xyz", charge=-2, target_spin=0.0)
# ClusterInfo: metals, bridging_atoms, terminal_ligands, formula, symmetry_group
```

Reference: `agent/structure_analyzer.py`

### 2.2 Active Space Construction

`build_active_space()` constructs an `ActiveSpace` at three levels using knowledge-base rules and cluster template matching:

| Level | Orbitals Included | Example (Fe₂S₂) | Example (FeMo-co) |
|-------|-------------------|------------------|--------------------|
| **minimal** | Metal d only | (10e, 10o) | (56e, 40o) |
| **standard** | + bridging p + interstitial | (22e, 16o) | (113e, 76o) |
| **extended** | + terminal ligand donors | system-dependent | (277e, 404o) |

Reference: `agent/active_space_builder.py`

### 2.3 Spin Isomer Enumeration

`enumerate_spin_isomers()` enumerates all collinear broken-symmetry spin isomers satisfying the constraint:

```
Valid isomers: {σ ∈ {+1, -1}ᴺ | Σᵢ σᵢ·Sᵢ = target_Sz}
```

Each isomer is labeled `BSn-ijk` where n = number of minority-spin metals. Symmetry reduction via `apply_symmetry_reduction()` groups equivalent isomers into `SpinIsomerFamily` objects. FeMo-co: 35 isomers → 10 families under C₃ symmetry.

Reference: `agent/spin_config.py`

### 2.4 Electronic Configuration Enumeration

The `generate_all_configs_v2()` function uses an **oxidation-first** approach, which is the preferred API (the older `generate_all_configs()` is spin-isomer-first):

```
for oxidation_assignment:
    per_metal_S = {i: get_local_spin(element_i, ox_i)}
    spin_isomers = enumerate_spin_isomers(cluster_info, oxidation_states=ox)
    for spin_isomer:
        for d_orbital_combo (Cartesian product):
            yield ElectronicConfig
```

**Oxidation state constraint:** `sum(metal_ox) + ligand_charge = total_charge`

Each metal's allowed oxidation states come from the knowledge base. Use `forced_oxidation` (CLI: `--metals-oxidation`) to override.

**d-orbital enumeration:** For each minority-spin metal site with a partially filled shell, `enumerate_d_orbital_configs()` determines which d-orbital hosts the extra electron. For example, Fe(II) d⁶ has 5 choices (one per singly-occupied orbital).

**Config count formula:**

```
Total = Σ_ox  Π_isomers  Π_sites_with_choice  n_d_choices(site)
```

FeMo-co: 35 isomers × 18 Fe(II)/Fe(III) assignments × 5 d-orbital choices per Fe(II) = 78,750 configurations.

Reference: `agent/electronic_config.py`

### 2.5 Symmetry Reduction

`reduce_configs_by_symmetry()` detects equivalent metal sites (same element, similar geometry, coordination environment) and keeps only one representative per equivalence class. This can reduce the config count significantly for symmetric clusters.

Reference: `agent/electronic_config.py`

### 2.6 Filtering Funnel Design

`design_filtering_funnel()` creates a hierarchical filtering plan with three styles:

| Style | Levels | Description |
|-------|--------|-------------|
| `femoco` | 6 | Aggressive filtering (default, designed for FeMo-co scale) |
| `conservative` | 4 | Keeps more configurations at each stage |
| `minimal` | 2 | UHF → final method |

FeMo-co example (femoco style):

```
78,750 UHF       →   840  (top 24/isomer)
  840 UCCSD      →   420  (top 12/isomer)
  420 UCCSD(T)   →    35  (top 1/isomer)
   35 DMRG       →    11  (by energy)
   11 CCSDTQ     →     3  (by energy)
    3 DMRG       →     2  (final candidates)
```

Reference: `agent/filtering.py`

### 2.7 Progressive Pipeline Execution

`run_pipeline()` executes the filtering funnel level-by-level. For each level:

1. Generate input scripts for surviving configs
2. Execute via subprocess
3. Parse `.npz` results
4. Select best configs via `select_from_*()` functions
5. Pass survivors to the next level

**High-spin UHF deduplication:** At the UHF level, configs are grouped by `(charge, spin, basis, d_count_targets)`. One high-spin UHF is run per group using `pyscf_uhf_highspin.py.j2`, then individual BS-UHF scripts load the shared NPZ — dramatically reducing redundant SCF calculations.

**Resume support:** `continue_from` parameter (CLI: `--continue-from`) allows resuming from a specific level index.

**Pipeline output directory:**

```
{structure_dir}/pipeline_output/
├── level_0_UHF/
│   ├── group_XXXXX_highspin.py       # shared high-spin computation
│   ├── {label}_uhf.py                # per-config BS-UHF scripts
│   ├── *.npz                          # results
│   └── pipeline.log
├── level_1_UCCSD/
├── level_2_UCCSD(T)/
├── level_3_DMRG/
├── ...
├── pipeline_summary.json
└── report.md
```

Reference: `agent/pipeline.py`

### 2.8 FCIDUMP Generation

Two modes are supported via `generate_fcidump()`:

- **Full-space** (`"full"`): Spin-free 1e/2e integrals in the full MO basis via PySCF's `ao2mo`
- **Active-space** (`"active"`): CASCI-extracted integrals with proper core energy, using ncore/cas parameters from DMRG NPZ when available

Output files are saved as `{label}_full.FCIDUMP` and `{label}_cas{nelecas}_{ncas}.FCIDUMP` in the `{structure_dir}/fcidump/` directory.

`generate_fcidump_for_results()` batch-generates FCIDUMP for all final pipeline results, auto-locating UHF and DMRG NPZ files from the pipeline directory.

Reference: `agent/fcidump.py`

## 3. Project Structure

```
APEX/
├── agent/                          # Core Python package
│   ├── __init__.py                 # Data model: 14 dataclasses + 1 enum
│   ├── __main__.py                 # Entry point: python -m agent
│   ├── main.py                     # CLI with 5 subcommands
│   ├── structure_analyzer.py       # Module 1: XYZ/PDB parsing, metal/ligand identification
│   ├── active_space_builder.py     # Module 2a: Rule-based active space construction
│   ├── orbital_constructor.py      # Module 2b: DFT→UNO/LUO pipeline (PySCF)
│   ├── orbital_ordering.py         # Module 2c: Fiedler/GA orbital reordering for DMRG
│   ├── spin_config.py              # Module 3: Collinear BS spin isomer enumeration
│   ├── electronic_config.py        # Module 4: Oxidation states + d-orbital occupancy
│   ├── bs_guess_builder.py         # Module 4b: ElectronicConfig → density matrix
│   ├── filtering.py                # Module 5: Hierarchical filtering funnel + selection
│   ├── input_generator.py          # Module 6: Template-based input file generation
│   ├── energy_extrapolation.py     # Module 7: DMRG/CC/FNO/MP2 extrapolation
│   ├── population_analysis.py      # Module 8: Mulliken/Meta-Löwdin population analysis
│   ├── result_parser.py            # Module 9: QC output parsing (.chk, .npz, stdout)
│   ├── pipeline.py                 # Module 10: End-to-end progressive filtering execution
│   ├── fcidump.py                  # Module 11: FCIDUMP integral file generation
│   ├── report.py                   # Markdown/JSON report generation
│   └── llm_agent.py                # LLM Agent layer (Anthropic API)
├── knowledge_base/
│   ├── transition_metals.yaml      # 30 transition metals (3d, 4d, 5d)
│   ├── ligand_database.yaml        # Bridging/donor atoms, terminal ligands
│   ├── cluster_templates.yaml      # FeMo-co, Fe₄S₄, Fe₂S₂, Mn₄CaO₅ reference data
│   └── basis_sets.yaml             # Recommended basis sets and DFT functionals
├── templates/
│   ├── pyscf_uhf.py.j2             # BS-UHF with spin-flip initial guess
│   ├── pyscf_uhf_highspin.py.j2    # High-spin UHF + d-count encoding (shared pre-computation)
│   ├── pyscf_ccsd.py.j2            # UCCSD/UCCSD(T) with T1 diagnostic
│   ├── pyscf_casscf.py.j2          # CASSCF/RASSCF with state averaging
│   ├── block2_dmrg.py.j2           # DMRG with adaptive sweep schedule
│   ├── hast_ucc.py.j2              # High-order tailored CC pipeline
│   └── slurm_job.sh.j2             # SLURM batch submission template
├── tests/
│   ├── test_core.py
│   ├── test_structure_analyzer.py
│   ├── test_active_space_builder.py
│   ├── test_spin_config.py
│   ├── test_electronic_config.py
│   ├── test_energy_extrapolation.py
│   ├── test_input_generator.py
│   ├── test_pipeline.py
│   └── run_pipeline_demo.py        # Interactive pipeline demo script
├── examples/
│   ├── fe2s2/fe2s2.xyz             # Fe₂S₂ dimer
│   ├── fe4s4/fe4s4.xyz             # Fe₄S₄ cubane
│   ├── vh4/vh4.xyz                 # VH₄ (single V site)
│   ├── vcl4/vcl4.xyz               # VCl₄ (single V with Cl ligands)
│   ├── v2o3/v2o3.xyz               # V₂O₃ dimer
│   ├── femoco/                     # FeMo-cofactor (expected output only)
│   ├── run_analysis.py             # Example analysis script
│   └── run_commands.txt            # Ready-to-run CLI examples
└── requirements.txt
```

## 4. Installation

### Prerequisites

- Python >= 3.10
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` contains:

| Package | Purpose |
|---------|---------|
| `numpy>=1.24` | Numerical arrays |
| `scipy>=1.10` | Optimization (curve fitting, linear assignment) |
| `ase>=3.22` | Structure file parsing (XYZ, PDB) |
| `pyyaml>=6.0` | Knowledge base loading |
| `jinja2>=3.1` | Template rendering |
| `anthropic>=0.40` | LLM Agent layer (optional) |

### Mandatory: PySCF

PySCF is required for the core pipeline functionality:

```bash
pip install pyscf>=2.5
```

PySCF is used for:
- UHF/UCCSD/UCCSD(T) calculations in the pipeline
- High-spin UHF pre-computation and d-count encoding
- FCIDUMP integral file generation
- Population analysis and oxidation state validation
- Orbital construction (UNO/LUO)

### Optional dependencies

| Package | Purpose |
|---------|---------|
| `block2` | DMRG calculations beyond PySCF's built-in |
| `hast-ucc` | High-order CCSDT/CCSDTQ via tailored CC |

## 5. Usage

### 5.1 `analyze` — Full Analysis Pipeline

```
python -m agent analyze STRUCTURE [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `STRUCTURE` | (required) | Path to XYZ or PDB structure file |
| `--charge` | `0` | Total molecular charge |
| `--spin` | `0.0` | Target spin S (e.g., 1.5 for S=3/2) |
| `--level` | `standard` | Active space level: `minimal`, `standard`, `extended` |
| `--output, -o` | (auto) | Output directory (default: `{structure_dir}/pipeline_output`) |
| `--report-format` | `markdown` | Report format: `markdown` or `json` |
| `--generate-inputs` | off | Generate QC input files |
| `--execute` | off | Execute the progressive filtering pipeline |
| `--n-final` | `5` | Number of final configs to keep (with `--execute`) |
| `--code` | `pyscf` | Target QC code: `pyscf`, `block2`, `orca`, `gaussian` |
| `--method` | `uhf` | QC method: `uhf`, `ccsd`, `casscf`, `dmrg` |
| `--basis` | `cc-pVDZ` | Basis set name |
| `--max-configs` | (none) | Limit on electronic configurations to enumerate |
| `--style` | `femoco` | Filtering funnel style: `femoco`, `conservative`, `minimal` |
| `--metals` | (auto) | Override auto-detected metal elements |
| `--metals-oxidation` | (none) | Force oxidation states, e.g., `0:3,1:3` |
| `--continue-from` | `0` | Resume pipeline from this level index (0-based) |
| `--no-fcidump` | off | Disable FCIDUMP generation for final results |

Examples:

```bash
# VH4 — single V(IV) site
python -m agent analyze examples/vh4/vh4.xyz --charge 0 --spin 0.5 --basis sto-3g --execute --n-final 1 --metals-oxidation "0:4"

# VCl4 — V(IV) with Cl ligands
python -m agent analyze examples/vcl4/vcl4.xyz --charge 0 --spin 0.5 --basis sto-3g --execute --n-final 1 --metals-oxidation "0:4"

# Fe₂S₂ dimer
python -m agent analyze examples/fe2s2/fe2s2.xyz --charge -2 --spin 0 --execute --n-final 3

# Fe₄S₄ cubane with config limit
python -m agent analyze examples/fe4s4/fe4s4.xyz --charge -2 --spin 0 --basis sto-3g --execute --n-final 3 --max-configs 50

# Resume pipeline from level 2
python -m agent analyze examples/fe2s2/fe2s2.xyz --charge -2 --spin 0 --execute --continue-from 2

# Disable FCIDUMP generation
python -m agent analyze examples/fe2s2/fe2s2.xyz --charge -2 --spin 0 --execute --no-fcidump
```

### 5.2 `spin` — Spin Isomer Enumeration

```
python -m agent spin --metals ELEMENTS... --spin TARGET_SZ [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--metals` | (required) | Space-separated metal element symbols |
| `--spin` | (required) | Target total Sz |
| `--oxidation` | (auto) | Oxidation state per metal |
| `--symmetry` | `C1` | Approximate point group |

Example:

```bash
python -m agent spin --metals Fe Fe Fe Fe Fe Fe Fe Mo --spin 2.0 --oxidation 3 3 3 3 3 3 3 3
```

### 5.3 `filter` — Filtering Funnel Design

```
python -m agent filter --n-configs N --n-electrons N --n-orbitals N [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-configs` | (required) | Total electronic configurations |
| `--n-electrons` | (required) | Active space electrons |
| `--n-orbitals` | (required) | Active space orbitals |
| `--n-isomers` | (none) | Number of spin isomers |
| `--style` | `femoco` | Funnel style: `femoco`, `conservative`, `minimal` |

Example:

```bash
python -m agent filter --n-configs 78750 --n-electrons 113 --n-orbitals 76 --n-isomers 35
```

### 5.4 `orbitals` — Orbital Construction

```
python -m agent orbitals STRUCTURE [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `STRUCTURE` | (required) | Path to structure file |
| `--charge` | `0` | Total charge |
| `--spin` | `0.0` | Target spin S |
| `--basis` | `cc-pVDZ` | Basis set |
| `--functional` | `B3LYP` | DFT functional for UKS |
| `--orbital-type` | `restricted_uno` | `restricted_uno` (UNO) or `unrestricted_luo` (LUO) |
| `--output, -o` | (none) | Output .npz file for orbital data |

### 5.5 `fcidump` — FCIDUMP Generation

```
python -m agent fcidump STRUCTURE [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `STRUCTURE` | (required) | Path to structure file (XYZ/PDB) |
| `--charge` | `0` | Total charge |
| `--spin` | `0.0` | Target spin S |
| `--basis` | `cc-pVDZ` | Basis set |
| `--uhf-npz` | (none) | Path to UHF `*_uhf.npz` (skips SCF if provided) |
| `--dmrg-npz` | (none) | Path to DMRG `*_dmrg_results.npz` (for ncore/cas info) |
| `--pipeline-dir` | (none) | Pipeline output dir to auto-locate NPZ files |
| `--active-electrons` | (none) | Active space electrons (if no DMRG NPZ) |
| `--active-orbitals` | (none) | Active space orbitals (if no DMRG NPZ) |
| `--mode` | `both` | FCIDUMP mode: `both`, `full`, `active` |
| `--output, -o` | (auto) | Output directory (default: `{structure_dir}/fcidump/`) |

Examples:

```bash
# Generate FCIDUMP from pipeline results
python -m agent fcidump examples/vh4/vh4.xyz --charge 0 --spin 0.5 \
    --pipeline-dir examples/vh4/pipeline_output --mode both

# From specific NPZ files
python -m agent fcidump examples/vh4/vh4.xyz \
    --uhf-npz examples/vh4/pipeline_output/level_0_UHF/label_uhf.npz \
    --dmrg-npz examples/vh4/pipeline_output/level_3_DMRG/label_dmrg_results.npz \
    --active-electrons 3 --active-orbitals 5

# Quick standalone (runs UHF from scratch)
python -m agent fcidump examples/fe2s2/fe2s2.xyz --charge -2 --spin 0
```

## 6. Module Architecture

The system has a four-layer architecture:

### Layer A: Pure Python (no QC dependency)

Modules 1, 2a, 3, 4, 5 — rule-based analysis and enumeration. Fully unit-testable, no PySCF required.

```
structure_analyzer.py → active_space_builder.py → spin_config.py → electronic_config.py → filtering.py
```

### Layer B: PySCF-driven

Modules 2b, 2c, 4b, 6, 9, 11 — computational orbital construction, input generation, result parsing, FCIDUMP. Requires PySCF.

```
orbital_constructor.py → orbital_ordering.py → bs_guess_builder.py
input_generator.py → result_parser.py → fcidump.py
```

### Layer C: Pipeline Orchestration

`pipeline.py` — Executes the progressive filtering funnel end-to-end with high-spin UHF deduplication, `continue_from` resume support, and optional FCIDUMP generation. Reports always saved to `{structure_dir}/pipeline_output/report.md`.

### Layer D: LLM Agent

`llm_agent.py` — Claude API wrapper for interactive reasoning, ambiguity resolution, and result interpretation.

## 7. Python API

### Structure Analysis

```python
from agent.structure_analyzer import parse_structure

cluster_info = parse_structure("structure.xyz", charge=-2, target_spin=0.0)
print(cluster_info.formula)           # "Fe2S2"
print(len(cluster_info.metals))       # 2
print(cluster_info.symmetry_group)    # "C1"
```

### Active Space Construction

```python
from agent.active_space_builder import build_active_space
from agent import ActiveSpaceLevel

aspace = build_active_space(cluster_info, ActiveSpaceLevel.STANDARD)
print(f"({aspace.n_electrons}e, {aspace.n_orbitals}o)")
```

### Spin Isomer Enumeration

```python
from agent.spin_config import enumerate_spin_isomers, apply_symmetry_reduction, label_isomers
import numpy as np

isomers = enumerate_spin_isomers(cluster_info, oxidation_states={0: 3, 1: 3})
families = apply_symmetry_reduction(isomers, "C3",
    np.array([m.position for m in cluster_info.metals]))
families = label_isomers(families)
```

### Electronic Configuration Enumeration (v2 API)

```python
from agent.electronic_config import generate_all_configs_v2, reduce_configs_by_symmetry

# Oxidation-first enumeration (preferred)
configs = generate_all_configs_v2(cluster_info, forced_oxidation={0: 3, 1: 3})
configs = reduce_configs_by_symmetry(configs, cluster_info)
print(f"Total: {len(configs)}")
```

### Filtering Funnel

```python
from agent.filtering import design_filtering_funnel

plan = design_filtering_funnel(
    n_configs=len(configs), active_space=aspace,
    n_spin_isomers=35, style="femoco",
)
```

### Pipeline Execution

```python
from agent.pipeline import run_pipeline

final_results = run_pipeline(
    configs=configs,
    active_space=aspace,
    cluster_info=cluster_info,
    plan=plan,
    workdir="pipeline_output",
    code="pyscf",
    basis_set="cc-pVDZ",
    n_final=5,
    continue_from=0,          # resume from level 0
    generate_fcidump=True,    # generate FCIDUMP for final results
)

for rank, r in enumerate(final_results, 1):
    print(f"  #{rank} {r.method} E = {r.energy:.12f} [{r.config.label}]")
```

### FCIDUMP Generation

```python
from agent.fcidump import generate_fcidump, generate_fcidump_for_results

# Single configuration
info = generate_fcidump(
    cluster_info, aspace, "path/to/uhf.npz", "output_dir/",
    dmrg_npz="path/to/dmrg_results.npz", mode="both",
)
print(info["full_space"])   # path to full-space FCIDUMP
print(info["active_space"]) # path to active-space FCIDUMP

# Batch for pipeline results
info_list = generate_fcidump_for_results(
    final_results, cluster_info, aspace, "pipeline_output", plan,
    basis_set="cc-pVDZ",
)
```

### Energy Extrapolation

```python
from agent.energy_extrapolation import dmrg_d_extrapolation, cc_composite_energy

# DMRG D-extrapolation
result = dmrg_d_extrapolation(
    bond_dims=[500, 1000, 2000, 5000, 10000],
    energies=[-100.05, -100.02, -100.01, -100.003, -100.001],
)
print(f"E_inf = {result.energy:.6f} +/- {result.uncertainty:.6f}")
```

### Result Parsing

```python
from agent.result_parser import parse_npz_result, to_calculation_result

result = parse_npz_result("config_label_uhf.npz")
calc = to_calculation_result(result, electronic_config)
print(f"Energy: {calc.energy}, Converged: {calc.converged}")
```

## 8. LLM Agent

The agent exposes all analysis tools through an interactive LLM interface:

```python
from agent.llm_agent import run_agent, run_interactive

result = run_agent(
    "Analyze the Fe7MoS9C cluster at charge=-1 for DMRG calculation. "
    "What active space and how many spin isomers should I expect?"
)

# Or interactively:
run_interactive()
```

Requires `anthropic` package and `ANTHROPIC_API_KEY` environment variable. The agent has 9 tools:

| Tool | Description |
|------|-------------|
| `analyze_structure` | Parse structure file, identify metals/ligands |
| `build_active_space` | Construct active space at specified level |
| `enumerate_spin_configs` | List all spin isomers with symmetry reduction |
| `enumerate_electronic_configs` | Full electronic configuration enumeration |
| `design_filtering_protocol` | Hierarchical filtering funnel design |
| `generate_input_file` | Generate QC input scripts |
| `parse_results` | Parse QC output files |
| `extrapolate_energy` | DMRG/CC energy extrapolation |
| `query_knowledge_base` | Look up metal/ligand/cluster reference data |

## 9. Knowledge Base

### `transition_metals.yaml`

Properties for **30 transition metals** (3d: Sc–Zn, 4d: Y–Cd, 5d: La, Hf–Hg):
- Electron configuration, common oxidation states
- High-spin d-electron counts and S values
- Covalent radii for bond detection
- Per-oxidation-state high-spin data

### `ligand_database.yaml`

- Bridging elements (S, O, N, C, Cl, Se) with typical charges and active orbital counts
- Donor atom properties
- Common terminal ligands (cysteine thiolate, histidine imidazole, etc.)

### `cluster_templates.yaml`

Validated reference data for known clusters:
- **FeMo-cofactor**: (113e, 76o) active space, 35 spin isomers, 10 BS families under C₃ symmetry, 78,750 electronic configurations
- **Fe₄S₄ cubane**: (22e, 20o) minimal, (46e, 32o) with sulfur
- **Fe₂S₂ dimer**: (10e, 10o) minimal, (22e, 16o) with sulfur
- **Mn₄CaO₅ OEC**: (24e, 20o) minimal for S₁ state

### `basis_sets.yaml`

Recommended basis sets and DFT functionals for common elements in transition metal clusters.

## 10. Input File Templates

| Template | Code | Method | Features |
|----------|------|--------|----------|
| `pyscf_uhf.py.j2` | PySCF | BS-UHF | Spin-flip initial guess, Mulliken analysis, DIIS control, NPZ output |
| `pyscf_uhf_highspin.py.j2` | PySCF | High-spin UHF | Shared high-spin computation, d-count encoding, saves NPZ for BS-UHF reuse |
| `pyscf_ccsd.py.j2` | PySCF | UCCSD/UCCSD(T) | T₁ diagnostic, density fitting, frozen natural orbitals, loads UHF NPZ |
| `pyscf_casscf.py.j2` | PySCF | CASSCF/RASSCF | State averaging, orbital rotation, DMRG interface |
| `block2_dmrg.py.j2` | BLOCK2 | DMRG | Adaptive sweep schedule, Fiedler ordering, noise terms |
| `hast_ucc.py.j2` | HAST-UCC | CCSDT/CCSDTQ | High-order tailored CC pipeline |
| `slurm_job.sh.j2` | SLURM | — | Batch submission with resource specification |

When Jinja2 templates are not found, `input_generator.py` falls back to built-in generators that produce complete, runnable Python scripts (PySCF) or shell scripts (other codes).

---

## Appendix

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_pipeline.py -v
pytest tests/test_electronic_config.py -v

# Pipeline demo
python tests/run_pipeline_demo.py --example fe2s2 --dry-run
python tests/run_pipeline_demo.py --example vh4 --charge 0 --spin 0.5
```

All tests are pure-Python and require no QC codes. Test coverage:

| Test file | Modules Covered |
|-----------|-----------------|
| `test_core.py` | All core modules (smoke tests) |
| `test_structure_analyzer.py` | Structure parsing, metal identification, symmetry |
| `test_active_space_builder.py` | Active space rules, oxidation states |
| `test_spin_config.py` | Spin enumeration, symmetry reduction, Heisenberg ranking |
| `test_electronic_config.py` | Oxidation enumeration, d-orbital choices, full config generation |
| `test_energy_extrapolation.py` | DMRG fit, CC composite, FNO, MP2 correction |
| `test_input_generator.py` | Template rendering, batch generation, batch submission scripts |
| `test_pipeline.py` | NPZ parsing, selection functions, end-to-end pipeline execution |
| `run_pipeline_demo.py` | Interactive demo with preset examples |

### References

- Zhai et al., "Classical solution of the FeMo-cofactor model to chemical accuracy and its implications," arXiv:2601.04621, 2026
- Li, Zhai, Chan, "The electronic complexity of the ground-state of the FeMo cofactor of nitrogenase as relevant to quantum simulations," JCP 150, 024302, 2019
- Reiher et al., "Elucidating reaction mechanisms on quantum computers," PNAS 114, 7555, 2017

### License

This project is for research use.

**Author:** Song@Elab
