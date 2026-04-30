# APEX_CAS — Automated Active Space Construction for Transition Metal Clusters

A standalone Python toolkit for automated active space construction for transition metal clusters, extracted from the [APEX](https://github.com/user/APEX) project. It supports structure parsing, SCF computation (UKS/UHF with sf-X2C), UNO/LUO/AVAS orbital construction, NOON quality validation, orbital visualization, and FCIDUMP generation. Based on Chan group's methodology (Li & Chan, JCTC 2017; Li et al., JCP 2019).

**[中文文档 / Chinese Documentation](README_CN.md)**

---

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy >= 1.24
- pyyaml >= 6.0
- pyscf >= 2.4 (required for computing-based CAS)
- h5py (for state persistence)

---

## Quick Example — Fe₂S₂ Walkthrough

The following uses `[Fe2S2(SCH3)4]^{2-}` to demonstrate the full pipeline from scratch to FCIDUMP.

### Step 1: Prepare inputs

Create a case directory with a structure file (`.xyz`) and a settings file (`.yaml`):

```
examples/fe2s2/
├── inputs/
│   ├── fe2s2.xyz                # Molecular structure (standard XYZ format)
│   └── fe2s2_settings.yaml      # CAS computation settings
└── outputs/                     ← Auto-generated after running
```

Create the settings YAML from the template:

```bash
cp shared/config/cas_settings_template.yaml examples/fe2s2/inputs/fe2s2_settings.yaml
```

Key settings (2017 paper parameters, BP86/sto-3g):

```yaml
charge: -2                       # Total charge
spin: 0.0                        # High-spin S (singlet S=0 here)

scf_method: "uks"                # UKS or UHF
xc_functional: "BP86"            # DFT functional

basis_set_default: "sto-3g"      # Default basis set
# basis_per_element:             # Per-element basis overrides
#   Fe: "def2-TZVP"
#   S:  "def2-TZVP"
#   C:  "def2-SVP"
#   H:  "def2-SVP"

conv_tol: 1.0e-8                 # SCF convergence threshold
max_cycle: 500                   # Max SCF iterations
localization_method: "pm"        # Localization: "pm" or "boys"
```

All fields in the template are commented out — uncomment only what you need to change.

### Step 2: Run compute

```bash
apex-cas compute examples/fe2s2/inputs/fe2s2.xyz \
  --case-dir examples/fe2s2 \
  --cas-settings examples/fe2s2/inputs/fe2s2_settings.yaml
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `structure` | Yes | Path to XYZ/PDB structure file |
| `--case-dir` | No | Case directory (inputs/outputs resolved under it) |
| `--cas-settings` | No | YAML settings file (defaults used if omitted) |
| `--charge` / `--spin` | No | Override charge/spin from YAML |

After completion, the `outputs/` directory contains:

```
examples/fe2s2/outputs/
├── scf/
│   ├── scf_info.json            # SCF info: energy, active space size, method
│   └── *.chk                    # PySCF checkpoint file
├── orbitals/
│   ├── orbital_report.yaml      # Orbital classification (with selected field, see Step 3)
│   ├── noon_plot.png            # NOON occupation number plot
│   └── cas_data.h5              # HDF5 orbital data
└── fcidump/                     ← Generated in Step 4
```

**`scf_info.json` example:**

```json
{
  "energy": -5021.07,
  "E_core": -5000.12,
  "E_act": -20.95,
  "E_vir": 0.00,
  "E_tot": -5021.07,
  "converged": true,
  "n_electrons": 10,
  "n_orbitals": 10,
  "cpt_cas_type": "uno",
  "source_method": "UKS-BP86/UNO"
}
```

### Step 3: Review/Adjust active orbitals

1. **Check the NOON plot**: Open `outputs/orbitals/noon_plot.png` — verify that orbitals with occupation between 0.02–1.98 (active region) are chemically reasonable

2. **Review the orbital report**: Open `outputs/orbitals/orbital_report.yaml`. Each orbital entry contains:

```yaml
- index: 78
  occupation: 1.8234
  auto_label: "active_0"
  chemical_label: "Fe0_3dxy"
  block: "active"
  selected: true              # ← Controls inclusion in active space
```

3. **Adjust (optional)**:
   - Set `selected: false` to exclude unwanted orbitals
   - Set `selected: true` to include additional orbitals (e.g., certain virtuals)
   - If no changes needed, use the automatic selection as-is

### Step 4: Generate FCIDUMP

```bash
apex-cas fcidump --case-dir examples/fe2s2
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--case-dir` | Yes | Case directory (auto-locates outputs/) |
| `--selection` | No | Custom orbital_report.yaml path (auto-detected by default) |
| `--reference-fcidump` | No | Reference FCIDUMP for comparison (resolved relative to case-dir) |
| `--target-spin` | No | Target MS2 value (default 0.0) |

**Output:**

```
examples/fe2s2/outputs/fcidump/FCIDUMP
examples/fe2s2/outputs/fcidump/FCIDUMP.ecore <---E_core
```

Standard FCIDUMP format, directly usable by Block2, CheMPS2, Dice, etc.

**Compare with reference (optional):**

```bash
apex-cas fcidump --case-dir examples/fe2s2 \
  --reference-fcidump chan_ref/fe2s2
```

---

## CLI Usage

### `apex-cas compute` — Parse structure → SCF → CAS → Visualize

```bash
apex-cas compute structure.xyz [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `structure` | positional | — | Path to XYZ/PDB structure file |
| `--case-dir` | str | parent dir of structure | Case directory for outputs |
| `--cas-settings` | str | — | YAML file with all CAS settings (see below) |
| `--charge` | int | 0 (or from YAML) | Total charge (overrides YAML) |
| `--spin` | float | 0.0 (or from YAML) | High-spin S value (overrides YAML) |
| `--no-cubes` | flag | — | Skip cube file generation |
| `--cube-grid` | str | 80x80x80 | Cube grid resolution |

**Example:**

```bash
# With YAML settings (recommended):
apex-cas compute Fe2S2.xyz --cas-settings fe2s2_settings.yaml

# With defaults only:
apex-cas compute Fe2S2.xyz --charge -2 --spin 0
```

### CAS Settings YAML

All computation parameters are managed through a YAML configuration file. A template is available at `shared/config/cas_settings_template.yaml`. 

**Configurable fields:**

| Field | Description |
|-------|-------------|
| `preset` | "default" (Chan 2019) or "fast" |
| `scf_method` | "uks" or "uhf" |
| `xc_functional` | DFT functional (e.g. B3LYP, BP86) |
| `basis_set_default` | Default basis set |
| `basis_per_element` | Per-element basis set overrides |
| `relativistic` | "none", "sf-x2c", or "dkh" |
| `solvation_model` | "none" or "ddcosmo" |
| `solvation_epsilon` | Solvent dielectric constant |
| `conv_tol` | Energy convergence threshold |
| `max_cycle` | Max SCF iterations |
| `init_guess` | "atom", "minao", "huckel", or "vsap" |
| `scf_damp` | Density damping factor |
| `scf_level_shift` | Virtual orbital level shift |
| `diis_space` | Number of DIIS vectors |
| `localization_method` | "pm" or "boys" |

### `apex-cas fcidump` — Load state → user YAML → FCIDUMP

```bash
apex-cas fcidump --case-dir <dir> [options]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--case-dir` | str | required | Case directory containing outputs/ |
| `--target-spin` | float | 0.0 | Target spin S for MS2 |
| `--output` | str | FCIDUMP | Output filename |
| `--reference-fcidump` | str | — | Reference FCIDUMP for comparison |
| `--selection` | str | auto | Path to user-edited orbital_report.yaml |

**Example:**

```bash
# After editing orbital_report.yaml to select orbitals:
apex-cas fcidump --case-dir examples/fe2s2/outputs_2019 \
    --target-spin 0 --reference-fcidump chan_ref/fe2s2
```

---

## Computational Pipeline

The full pipeline consists of 6 steps. The CLI `compute` command executes Steps 1–5; the `fcidump` command executes Step 6.

### Step 1: Structure Parsing

**What**: Parse XYZ/PDB files to identify metal centers, bridging atoms, and terminal ligands.

**Function**: `parse_structure(filepath, charge=0, target_spin=0.0)` from `structure_analyzer.py`

**Key logic**:
- Identifies metals from the `TRANSITION_METALS` set (3d, 4d, 5d series)
- Identifies bridging atoms (S, O, N, Se, Cl, P, etc.) that connect ≥2 metals
- Identifies terminal ligands bonded to a single metal

**Output**: `ClusterInfo` dataclass:
- `metals`: list of `MetalCenter` (element, index, position, neighbors, coordination)
- `bridging_atoms`: list of `BridgingAtom` (element, index, bridged_metals)
- `terminal_ligands`: list of `TerminalLigand`
- `all_elements`, `all_positions`, `formula`, `total_charge`, `target_spin`

---

### Step 2: SCF Computation

**What**: Build PySCF molecule and run high-spin SCF (UKS or UHF) with scalar relativistic corrections and optional solvation.

**Functions** (from `CAS_builder_computing.py`):
- `build_mol_with_basis(cluster_info, settings)` → PySCF `Mole` object with per-element basis support
- `_run_high_spin_scf(mol, settings)` → converged SCF object

**ComputationSettings parameters** (from `shared/models.py`, defaults = Chan 2019):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scf_method` | "uks" | "uks" or "uhf" |
| `xc_functional` | "B3LYP" | DFT functional |
| `basis_set_default` | "def2-TZVP" | Default basis for unspecified elements |
| `basis_set_per_element` | {Fe: def2-TZVP, S: def2-TZVP, C: def2-SVP, H: def2-SVP, O: def2-SVP, N: def2-SVP} | Per-element basis (priority over default) |
| `relativistic` | "sf-x2c" | "none", "sf-x2c", or "dkh" |
| `solvation_model` | "ddcosmo" | "none" or "ddcosmo" |
| `solvation_epsilon` | 4.0 | Solvent dielectric constant |
| `conv_tol` | 1e-8 | SCF convergence tolerance |
| `max_cycle` | 200 | Max SCF iterations |
| `scf_verbose` | 4 | PySCF SCF output verbosity level |

**Presets** (from `computation_defaults.py`):
- `"default"` — Chan 2019: B3LYP/def2-TZVP + sf-X2C + ddCOSMO(ε=4.0), conv_tol=1e-8
- `"fast"` — B3LYP/def2-SVP, no relativistic, no solvation, conv_tol=1e-6

**Output**: `(mol, mf, chkfile_path)` tuple

---

### Step 3: Orbital Construction

**What**: Transform SCF orbitals into chemically meaningful active space orbitals.

**Entry point**: `build_computed_CAS(cluster_info, settings, cpt_cas_type="uno", localization_method="pm", save_dir=".")` from `CAS_builder_computing.py`

**Returns**: `CAS` dataclass with `mo_coeff_full`, `occupations_full`, `orbital_labels_full`, etc.

Three paths, dispatched by `cpt_cas_type`:

#### UNO path (default, `"uno"`)

Based on Chan 2017/2019 methodology:

1. **`compute_unos(mol, mf)`** — Diagonalize the total (α+β) 1-RDM via generalized eigenvalue problem `eig(S·D·S, S)` → natural orbitals sorted by decreasing occupation number
2. **`split_localize(mol, mo_coeff_uno, occ_uno, ...)`** — Partition orbitals into:
   - Core: occ > 1.98
   - Active: 0.02 ≤ occ ≤ 1.98
   - Virtual: occ < 0.02
   - Localize each block separately (Pipek-Mezey or Boys). Assign chemical character labels (e.g., "Fe1_dxy", "S3_px").
3. **Active selection**: All orbitals with NOON in [0.02, 1.98] are automatically selected as active
4. **Electron count**: `n_electrons = round(sum(occupations[active_indices]))`

#### LUO path (`"luo"`)

For DMRG calculations needing unrestricted orbital representation:

1. Localize alpha and beta MOs separately (occupied + virtual blocks)
2. Select active by `_select_by_projection_threshold(mol, mo_coeff, cluster_info, threshold=0.05)` — keeps orbitals with projection weight onto metal-d + bridging-p subspace > threshold
3. Returns separate `mo_coeff_alpha` and `mo_coeff_beta`

#### AVAS path (`"avas"`)

Automated Valence Active Space (Sayfutyarova et al., JCTC 2017):

1. `_construct_avas(mol, mf, cluster_info, avas_config)`
2. `avas_select(mol, mo_coeff, valence_orbitals, threshold=0.4)` — projects MOs onto target AO subspace and keeps those above threshold
3. Valence orbitals auto-inferred from knowledge base or specified via `AVASConfig`

---

### Step 4: Quality Validation

**What**: Assess active space quality using Natural Orbital Occupation Number (NOON) analysis.

**Function**: `validate_noon(active_orbitals, expected_types=None, noon_lo=0.02, noon_hi=1.98)` from `CAS_quality.py`

**Returns**: `ActiveSpaceQuality`:
- `n_doubly_occupied`: orbitals with NOON > noon_hi (penalty: -0.1 each)
- `n_empty`: orbitals with NOON < noon_lo (penalty: -0.05 each)
- `missing_orbital_types`: expected types not found (penalty: -0.1 each)
- `quality_score`: 0.0–1.0, computed from penalties

**Display**: `print_quality_report(quality)` → human-readable multi-line report

---

### Step 5: State Persistence & Visualization

**Functions** from `orbital_visualizer.py`:

**`save_cas_state(cas, mol, mf, output_dir)`** saves:
- `outputs/scf/chkfile` — PySCF checkpoint
- `outputs/scf/scf_info.json` — energy, convergence info
- `outputs/orbitals/cas_data.h5` — HDF5 with `mo_coeff_full`, `occupations_full`, `orbital_labels_full`

**`plot_orbitals(cas, mol, output_dir, cluster_info=None, ...)`** generates:
- `orbital_report.yaml` — each orbital: `index`, `occupation`, `auto_label`, `chemical_label`, `block`, `selected`
- `noon_plot.png` — NOON bar chart
- `cubes/` — cube files for orbital visualization in VESTA/Jmol (optional)

**`load_cas_state(case_dir)`** restores `(CAS, mol, mf)` from disk

---

### Step 6: FCIDUMP Generation

**What**: Transform AO integrals to active-space MO basis and write standard FCIDUMP format.

**Workflow** (CLI `apex-cas fcidump`):

1. `load_cas_state(case_dir)` → restore CAS, mol, mf
2. `load_user_selection(yaml_path)` → read user-edited YAML (orbitals marked `selected: true`)
3. `generate_fcidump_from_selection(mol, mf, mo_coeff_loc, occupations, selected_indices, output_path, target_spin)`:
   - Extracts active MO coefficients
   - Computes `n_electrons = round(sum(occupations[selected_indices]))`
   - `transform_active_integrals(mol, mf, mo_active, n_electrons, target_spin)` → h1e, eri, ecore, ms2, ncore
   - `write_fcidump(integrals, output_path)` → standard FCIDUMP file
4. Optional: `compare_fcidumps(ref_path, new_path)` → comparison of h1e, h2e, ecore

---

## Python API

```python
from apex_cas import (
    # Data models
    ClusterInfo, CAS, ComputationSettings, AVASConfig,
    # Core functions
    parse_structure,
    build_computed_CAS,
    validate_noon, print_quality_report,
)
from apex_cas.computation_defaults import apply_overrides, PRESETS
from apex_cas.orbital_visualizer import (
    save_cas_state, load_cas_state,
    plot_orbitals, load_user_selection,
)
from apex_cas.FCIDUMP_generator import (
    generate_fcidump_from_selection,
    compare_fcidumps,
)

# Step 1: Parse structure
cluster_info = parse_structure("structure.xyz", charge=-2, target_spin=0.0)

# Step 2+3: Build CAS with custom settings
settings = apply_overrides(
    ComputationSettings(),
    xc_functional="B3LYP",
    basis_set_default="def2-TZVP",
    relativistic="sf-x2c",
)
cas = build_computed_CAS(cluster_info, settings, cpt_cas_type="uno")
print(f"Active space: CAS({cas.n_electrons}e, {cas.n_orbitals}o)")

# Step 4: Validate
quality = validate_noon(cas)
print(print_quality_report(quality))

# Step 6: Generate FCIDUMP (after user selection)
cas, mol, mf = load_cas_state("case_dir")
selected_indices, labels, meta = load_user_selection("orbital_report.yaml")
generate_fcidump_from_selection(
    mol, mf, cas.mo_coeff_full, cas.occupations_full,
    selected_indices, "FCIDUMP", target_spin=0.0,
)
```

---

## Reproduction Scripts

Located in `scripts/`:

| Script | Paper | DFT | Basis | Solvation |
|--------|-------|-----|-------|-----------|
| `run_fe2s2_2017.py` | Li & Chan, JCTC 2017 | BP86 | tzp-dkh (all) | none |
| `run_fe2s2_2019.py` | Li et al., JCP 2019 | B3LYP | TZP-DKH(Fe,S) + def2-SVP(C,H) | COSMO ε=4.0 |

Both target **CAS(30e, 20o)** for **[Fe₂S₂(SCH₃)₄]²⁻**.

```bash
# From APEX project root:
python APEX_CAS/scripts/run_fe2s2_2017.py --root /path/to/APEX
python APEX_CAS/scripts/run_fe2s2_2019.py --root /path/to/APEX
```

Output goes to `examples/fe2s2/outputs_2017/` and `examples/fe2s2/outputs_2019/` respectively.

---

## Project Structure

```
APEX_CAS/
├── README.md              # This file (English)
├── README_CN.md           # Chinese documentation
├── pyproject.toml
├── requirements.txt
├── apex_cas/
│   ├── __init__.py            # Public API exports
│   ├── __main__.py            # Entry point
│   ├── _paths.py              # Data file path resolution
│   ├── main.py                # CLI (compute + fcidump)
│   ├── structure_analyzer.py  # Step 1: Structure parsing
│   ├── CAS_builder_computing.py    # Step 2–3: SCF + orbital construction
│   ├── CAS_builder_noncomputing.py  # Rule-based CAS (knowledge base)
│   ├── computation_defaults.py      # Presets & settings utilities
│   ├── CAS_quality.py              # Step 4: NOON validation
│   ├── orbital_visualizer.py        # Step 5: Visualization & state persistence
│   ├── FCIDUMP_generator.py         # Step 6: FCIDUMP generation
│   ├── orbital_optimizer.py         # UCCSD orbital optimization
│   └── models.py                    # Re-exports from shared.models
├── scripts/
│   ├── run_fe2s2_2017.py    # Fe₂S₂ 2017 reproduction
│   └── run_fe2s2_2019.py    # Fe₂S₂ 2019 reproduction
├── tests/
│   ├── test_structure_analyzer.py
│   ├── test_active_space_builder.py
│   ├── test_CAS_quality.py
│   ├── test_computation_settings.py
│   ├── test_orbital_optimizer.py
│   └── test_stage1_methods.py
└── ref/                     # Reference literature guides
```

---

## Origin

This project is extracted from [APEX](https://github.com/user/APEX) — Automated Progressive Electronic structure eXploration. The upstream project includes additional features like spin configuration enumeration, filtering protocols, and quantum chemistry input generation.
