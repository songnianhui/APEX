"""QC input rendering utilities.

Template-based input file generation for quantum chemistry codes.
Uses Jinja2 templates to render code-specific input files from
ElectronicConfig and CAS data.
"""

import os
from datetime import datetime

import jinja2

from .models import (
    CAS,
    ClusterInfo,
    ElectronicConfig,
)



def _sanitize_label(label: str) -> str:
    """Return a filesystem-safe label (no pipes or spaces)."""
    return label.replace("|", "_").replace(" ", "_")


# ──────────────────────────────────────────────────────────────────
# Template loader
# ──────────────────────────────────────────────────────────────────

_APEX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TEMPLATE_DIRS = [
    os.path.join(_APEX_ROOT, "shared", "templates"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
]


def _get_jinja_env():
    """Get Jinja2 environment with the templates directory."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader([d for d in _TEMPLATE_DIRS if os.path.isdir(d)]),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def list_available_templates() -> list:
    """List all available input file templates."""
    templates = set()
    for template_dir in _TEMPLATE_DIRS:
        if os.path.isdir(template_dir):
            for f in os.listdir(template_dir):
                if f.endswith(".j2"):
                    templates.add(f)
    return sorted(templates)


# ──────────────────────────────────────────────────────────────────
# Unified input generation interface
# ──────────────────────────────────────────────────────────────────

def generate_input(config: ElectronicConfig,
                    active_space: CAS,
                    cluster_info: ClusterInfo,
                    code: str = "pyscf",
                    method: str = "uhf",
                    **kwargs) -> str:
    """Generate input file for a QC code.

    Args:
        config: Electronic configuration.
        active_space: Active space specification.
        cluster_info: Cluster description.
        code: Target QC code ("pyscf", "block2", "hast_ucc", "orca", "gaussian", "molpro", "bagel").
        method: Calculation method ("uhf", "ccsd", "ccsd-t", "ccsdt", "casscf", "dmrg", "bsdft").
        **kwargs: Additional parameters (basis_set, bond_dim, etc.).

    Returns:
        Rendered input file content as string.
    """
    context = _build_context(config, active_space, cluster_info, method, **kwargs)

    template_name = _get_template_name(code, method)

    try:
        env = _get_jinja_env()
        template = env.get_template(template_name)
        return template.render(**context)
    except jinja2.TemplateNotFound:
        # Fall back to built-in generation
        return _generate_builtin(config, active_space, cluster_info, code, method, **kwargs)


def generate_batch(configs: list,
                    active_space: CAS,
                    cluster_info: ClusterInfo,
                    code: str = "pyscf",
                    method: str = "uhf",
                    output_dir: str = None,
                    **kwargs) -> list:
    """Generate input files for a batch of configurations.

    Args:
        configs: List of ElectronicConfig objects.
        active_space: Active space specification.
        cluster_info: Cluster description.
        code: Target QC code.
        method: Calculation method.
        output_dir: If provided, write files to this directory.
        **kwargs: Additional parameters.

    Returns:
        List of (filename, content) tuples.
    """
    results = []
    for i, config in enumerate(configs):
        content = generate_input(config, active_space, cluster_info,
                                  code, method, **kwargs)
        ext = _get_file_extension(code, method)
        filename = f"{config.label.replace('|', '_')}_{method}{ext}"
        filename = filename.replace(" ", "_")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)

        results.append((filename, content))

    return results


# ──────────────────────────────────────────────────────────────────
# Template name resolution
# ──────────────────────────────────────────────────────────────────

def _get_template_name(code, method):
    """Map (code, method) to template filename."""
    mapping = {
        ("pyscf", "uhf"): "pyscf_uhf.py.j2",
        ("pyscf", "uhf_highspin"): "pyscf_uhf_highspin.py.j2",
        ("pyscf", "ccsd"): "pyscf_ccsd.py.j2",
        ("pyscf", "ccsd-t"): "pyscf_ccsd.py.j2",
        ("pyscf", "casscf"): "pyscf_casscf.py.j2",
        ("pyscf", "casci"): "pyscf_casscf.py.j2",
        ("block2", "dmrg"): "block2_dmrg.py.j2",
        ("hast_ucc", "ccsdt"): "hast_ucc.py.j2",
        ("hast_ucc", "ccsdtq"): "hast_ucc.py.j2",
        ("orca", "bsdft"): "orca_bsdft.inp.j2",
        ("orca", "casscf"): "orca_casscf.inp.j2",
        ("gaussian", "bsdft"): "gaussian_bsdft.gjf.j2",
        ("molpro", "caspt2"): "molpro_caspt2.inp.j2",
        ("bagel", "caspt2"): "bagel_caspt2.json.j2",
    }
    return mapping.get((code.lower(), method.lower()),
                        f"{code.lower()}_{method.lower()}.j2")


def _get_file_extension(code, method):
    """Get file extension for a given code and method."""
    ext_map = {
        "pyscf": ".py",
        "block2": ".py",
        "hast_ucc": ".py",
        "orca": ".inp",
        "gaussian": ".gjf",
        "molpro": ".inp",
        "bagel": ".json",
    }
    return ext_map.get(code.lower(), ".inp")


# ──────────────────────────────────────────────────────────────────
# Context building
# ──────────────────────────────────────────────────────────────────

class _ConfigProxy:
    """Wrap an ElectronicConfig so Jinja2 templates can use both attribute
    access (``config.spin_isomer``) and dict-style ``.get()`` calls
    (``config.get('ccsd_conv_tol', 1e-8)``).

    Any extra keys supplied at construction time (e.g. ``uhf_npz``) are
    also available via ``.get()`` or attribute access.
    """

    def __init__(self, config, extra=None):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_extra", extra or {})

    # dict-style access for template .get() calls
    def get(self, key, default=None):
        if key in self._extra:
            return self._extra[key]
        return getattr(self._config, key, default)

    def __contains__(self, key):
        return hasattr(self._config, key) or key in self._extra

    # attribute pass-through to the real config
    def __getattr__(self, name):
        if name in self._extra:
            return self._extra[name]
        return getattr(self._config, name)

    def __bool__(self):
        return self._config is not None


def _build_context(config, active_space, cluster_info, method, **kwargs):
    """Build template context from input data."""
    # Geometry string
    geometry_lines = []
    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        geometry_lines.append(f"  {elem} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
    geometry = "\n".join(geometry_lines)

    # Spin info
    minority_sites = config.minority_spin_sites if config else []
    spin_assignment = config.spin_assignment if config else {}

    # D-orbital assignments
    d_orbital_assignments = config.d_orbital_assignments if config else {}

    # Oxidation state info
    oxidation = {}
    if config and config.oxidation:
        oxidation = config.oxidation.assignments

    # d-electron count targets per metal site (for UHF density encoding)
    d_count_targets = {}
    if oxidation and cluster_info and cluster_info.metals:
        from shared.chem_knowledge import get_d_electron_count
        for site_idx, ox_state in oxidation.items():
            if site_idx < len(cluster_info.metals):
                element = cluster_info.metals[site_idx].element
                d_count_targets[site_idx] = get_d_electron_count(element, ox_state)

    context = {
        "config": _ConfigProxy(config, extra={
            k: v for k, v in kwargs.items()
            if k not in ("basis_set", "bond_dim", "max_scf_cycles",
                         "conv_tol", "xc_functional")
        }),
        "active_space": active_space,
        "cluster_info": cluster_info,
        "geometry": geometry,
        "charge": cluster_info.total_charge,
        # PySCF uses 2S = Nalpha - Nbeta, while many external QC inputs use multiplicity = 2S + 1.
        "spin": int(round(2 * cluster_info.target_spin)),
        "spin_multiplicity": int(round(2 * cluster_info.target_spin)) + 1,
        "target_S": cluster_info.target_spin,
        "n_electrons": active_space.n_electrons,
        "n_orbitals": active_space.n_orbitals,
        "minority_sites": minority_sites,
        "spin_assignment": spin_assignment,
        "d_orbital_assignments": d_orbital_assignments,
        "oxidation": oxidation,
        "d_count_targets": d_count_targets,
        "method": method,
        "basis_set": kwargs.get("basis_set", "cc-pVDZ"),
        "bond_dim": kwargs.get("bond_dim", 5000),
        "max_scf_cycles": kwargs.get("max_scf_cycles", 2000),
        "conv_tol": kwargs.get("conv_tol", 1e-8),
        "xc_functional": kwargs.get("xc_functional", "B3LYP"),
        "label": _sanitize_label(config.label) if config else "unknown",
        "timestamp": datetime.now().isoformat(),
        "metal_atom_indices": [m.index for m in cluster_info.metals],
    }

    # Add all kwargs to context
    context.update(kwargs)

    return context


# ──────────────────────────────────────────────────────────────────
# Built-in generators (when templates are not available)
# ──────────────────────────────────────────────────────────────────

def _generate_builtin(config, active_space, cluster_info, code, method, **kwargs):
    """Generate input using built-in Python string templates.

    These generators are fallback paths when a Jinja template is not
    available. They are convenience emitters, not the canonical workflow
    definition for Chan-2026 reproduction.
    """
    generators = {
        "pyscf": {
            "uhf": _gen_pyscf_uhf,
            "ccsd": _gen_pyscf_ccsd,
            "ccsd-t": _gen_pyscf_ccsd_t,
            "casscf": _gen_pyscf_casscf,
        },
        "block2": {
            "dmrg": _gen_block2_dmrg,
        },
        "orca": {
            "bsdft": _gen_orca_bsdft,
        },
        "gaussian": {
            "bsdft": _gen_gaussian_bsdft,
        },
    }

    gen = generators.get(code.lower(), {}).get(method.lower())
    if gen:
        return gen(config, active_space, cluster_info, **kwargs)

    # Generic fallback
    return f"# Auto-generated input for {code}/{method}\n# Config: {config.label if config else 'N/A'}\n"


def _gen_pyscf_uhf(config, active_space, cluster_info, **kwargs):
    """Generate PySCF UHF input with broken-symmetry guess."""
    ctx = _build_context(config, active_space, cluster_info, "uhf", **kwargs)

    minority_list = ctx["minority_sites"]
    metal_labels = [m.label for m in cluster_info.metals]

    lines = [
        "#!/usr/bin/env python3",
        f'# Auto-generated PySCF UHF input — {ctx["label"]}',
        f'# Generated: {ctx["timestamp"]}',
        "",
        "from pyscf import gto, scf",
        "import numpy as np",
        "",
        f'mol = gto.M(',
        f'    atom="""',
        f'{ctx["geometry"]}',
        f'    """,',
        f'    charge={ctx["charge"]},',
        f'    spin={ctx["spin"]},',
        f'    basis="{ctx["basis_set"]}",',
        f'    symmetry=False,',
        f'    verbose=4,',
        f')',
        "",
        "mf = scf.UHF(mol)",
        f"mf.conv_tol = {ctx['conv_tol']}",
        f"mf.max_cycle = {ctx['max_scf_cycles']}",
        "",
        "# Build broken-symmetry initial guess",
        "dm_init = mf.get_init_guess(key='atom')",
        "dm_a, dm_b = dm_init[0].copy(), dm_init[1].copy()",
        "",
        "# Flip spins on minority-spin sites",
    ]

    if minority_list:
        for site_idx in minority_list:
            metal = cluster_info.metals[site_idx]
            lines.append(f"# Flip {metal.label} (atom index {metal.index})")
            lines.append(f"atom_idx = {metal.index}")
            lines.append("aoslice = mol.aoslice_by_atom()[atom_idx]")
            lines.append("ao_s, ao_e = aoslice[2], aoslice[3]")
            lines.append("tmp = dm_a[ao_s:ao_e, :].copy()")
            lines.append("dm_a[ao_s:ao_e, :] = dm_b[ao_s:ao_e, :].copy()")
            lines.append("dm_b[ao_s:ao_e, :] = tmp")
            lines.append("tmp = dm_a[:, ao_s:ao_e].copy()")
            lines.append("dm_a[:, ao_s:ao_e] = dm_b[:, ao_s:ao_e].copy()")
            lines.append("dm_b[:, ao_s:ao_e] = tmp")
            lines.append("")

    # D-orbital assignment: adjust beta population
    d_assignments = ctx.get("d_orbital_assignments", {})
    metal_atom_indices = ctx.get("metal_atom_indices", [])
    if d_assignments:
        lines.append("# --- D-orbital assignment: adjust beta d-orbital population ---")
        lines.append(f"d_orbital_assignments = {dict(d_assignments)}")
        lines.append(f"metal_atom_indices = {metal_atom_indices}")
        lines.append("")
        lines.append("for site_idx, d_idx in d_orbital_assignments.items():")
        lines.append("    atom_idx = metal_atom_indices[site_idx]")
        lines.append("    aoslice = mol.aoslice_by_atom()[atom_idx]")
        lines.append("    ao_s, ao_e = aoslice[2], aoslice[3]")
        lines.append("    # Find d-type AO functions (l=2) on this atom")
        lines.append("    d_ao_start, d_ao_end = None, None")
        lines.append("    bas_offset = 0")
        lines.append("    for ibas in range(mol.nbas):")
        lines.append("        l_ang = mol.bas_angular(ibas)")
        lines.append("        nctr = mol.bas_nctr(ibas)")
        lines.append("        nshells = nctr * (2 * l_ang + 1)")
        lines.append("        if bas_offset + nshells > ao_s and bas_offset < ao_e:")
        lines.append("            if l_ang == 2:")
        lines.append("                if d_ao_start is None:")
        lines.append("                    d_ao_start = max(bas_offset, ao_s)")
        lines.append("                d_ao_end = min(bas_offset + nshells, ao_e)")
        lines.append("        bas_offset += nshells")
        lines.append("    if d_ao_start is not None and d_ao_end is not None and (d_ao_end - d_ao_start) > d_idx:")
        lines.append("        d_block = dm_b[np.ix_(range(d_ao_start, d_ao_end), range(d_ao_start, d_ao_end))]")
        lines.append("        eigvals, eigvecs = np.linalg.eigh(d_block)")
        lines.append("        sort_idx = np.argsort(eigvals)[::-1]")
        lines.append("        eigvals = eigvals[sort_idx]")
        lines.append("        eigvecs = eigvecs[:, sort_idx]")
        lines.append("        if d_idx > 0 and d_idx < len(eigvals):")
        lines.append("            swap = np.eye(len(eigvals))")
        lines.append("            swap[:, [0, d_idx]] = swap[:, [d_idx, 0]]")
        lines.append("            rotated_eigvecs = eigvecs @ swap")
        lines.append("            new_d_block = rotated_eigvecs @ np.diag(eigvals) @ rotated_eigvecs.T")
        lines.append("            dm_b[np.ix_(range(d_ao_start, d_ao_end), range(d_ao_start, d_ao_end))] = new_d_block")
        lines.append("            print(f'  Metal site {site_idx} (atom {atom_idx}): beta d-orbital {d_idx} assigned')")
        lines.append("    else:")
        lines.append("        print(f'  WARNING: Could not locate d-orbitals on site {site_idx}')")
        lines.append("")

    lines.extend([
        "dm_init = np.array([dm_a, dm_b])",
        "",
        "mf.kernel(dm0=dm_init)",
        "",
        "print(f'UHF Energy: {mf.e_tot:.10f}')",
        "print(f'<S^2>: {mf.spin_square():.4f}')",
        "",
        "# Save checkpoint",
        f'mf.chkfile = "{ctx["label"].replace("|", "_")}_uhf.chk"',
        "mf.dump_chk()",
    ])

    return "\n".join(lines)


def _gen_pyscf_ccsd(config, active_space, cluster_info, **kwargs):
    """Generate fallback PySCF UCCSD input."""
    ctx = _build_context(config, active_space, cluster_info, "ccsd", **kwargs)

    lines = [
        "#!/usr/bin/env python3",
        f'# Auto-generated PySCF CCSD input — {ctx["label"]}',
        "",
        "from pyscf import gto, scf, cc",
        "import numpy as np",
        "",
        f'mol = gto.M(',
        f'    atom="""',
        f'{ctx["geometry"]}',
        f'    """,',
        f'    charge={ctx["charge"]},',
        f'    spin={ctx["spin"]},',
        f'    basis="{ctx["basis_set"]}",',
        f'    symmetry=False,',
        f'    verbose=4,',
        f')',
        "",
        "# Run UHF first",
        "mf = scf.UHF(mol)",
        f"mf.conv_tol = {ctx['conv_tol']}",
        "mf.kernel()",
        "",
        "# UCCSD",
        "mycc = cc.UCCSD(mf)",
        "mycc.kernel()",
        "",
        "print(f'UCCSD Energy: {mycc.e_tot:.10f}')",
        "print(f'Correlation energy: {mycc.e_corr:.10f}')",
    ]

    return "\n".join(lines)


def _gen_pyscf_ccsd_t(config, active_space, cluster_info, **kwargs):
    """Generate fallback PySCF UCCSD(T) input."""
    base = _gen_pyscf_ccsd(config, active_space, cluster_info, **kwargs)
    lines = base.splitlines()
    lines.extend([
        "",
        "# UCCSD(T) correction",
        "et = mycc.ccsd_t()",
        "print(f'UCCSD(T) Energy: {mycc.e_tot + et:.10f}')",
    ])
    return "\n".join(lines)


def _gen_pyscf_casscf(config, active_space, cluster_info, **kwargs):
    """Generate PySCF CASSCF input."""
    ctx = _build_context(config, active_space, cluster_info, "casscf", **kwargs)

    lines = [
        "#!/usr/bin/env python3",
        f'# Auto-generated PySCF CASSCF input — {ctx["label"]}',
        "",
        "from pyscf import gto, scf, mcscf",
        "",
        f'mol = gto.M(',
        f'    atom="""',
        f'{ctx["geometry"]}',
        f'    """,',
        f'    charge={ctx["charge"]},',
        f'    spin={ctx["spin"]},',
        f'    basis="{ctx["basis_set"]}",',
        f'    symmetry=False,',
        f')',
        "",
        "mf = scf.UHF(mol)",
        "mf.kernel()",
        "",
        f"# CASSCF with ({ctx['n_electrons']}e, {ctx['n_orbitals']}o) active space",
        f"mc = mcscf.UCASSCF(mf, {ctx['n_orbitals']}, {ctx['n_electrons']})",
        "mc.kernel()",
        "",
        "print(f'CASSCF Energy: {mc.e_tot:.10f}')",
    ]

    return "\n".join(lines)


def _gen_block2_dmrg(config, active_space, cluster_info, **kwargs):
    """Generate BLOCK2 DMRG input via PySCF interface."""
    ctx = _build_context(config, active_space, cluster_info, "dmrg", **kwargs)
    bond_dim = ctx["bond_dim"]

    lines = [
        "#!/usr/bin/env python3",
        f'# Auto-generated BLOCK2 DMRG input — {ctx["label"]}',
        "",
        "from pyscf import gto, scf, mcscf",
        "from pyscf.mcscf import DMRGCI",
        "",
        f'mol = gto.M(',
        f'    atom="""',
        f'{ctx["geometry"]}',
        f'    """,',
        f'    charge={ctx["charge"]},',
        f'    spin={ctx["spin"]},',
        f'    basis="{ctx["basis_set"]}",',
        f'    symmetry=False,',
        f')',
        "",
        "mf = scf.UHF(mol)",
        "mf.kernel()",
        "",
        f"# DMRG with ({ctx['n_electrons']}e, {ctx['n_orbitals']}o) active space",
        f"mc = DMRGCI(mf, {ctx['n_orbitals']}, {ctx['n_electrons']})",
        f"mc.maxM = {bond_dim}",
        "mc.kernel()",
        "",
        "print(f'DMRG Energy: {mc.e_tot:.10f}')",
    ]

    return "\n".join(lines)


def _gen_orca_bsdft(config, active_space, cluster_info, **kwargs):
    """Generate ORCA broken-symmetry DFT input."""
    ctx = _build_context(config, active_space, cluster_info, "bsdft", **kwargs)

    lines = [
        f'# Auto-generated ORCA BS-DFT input — {ctx["label"]}',
        "",
        f"! {ctx['xc_functional']} {ctx['basis_set']} TightSCF",
        "",
        f"%scf",
        f"   MaxIter {ctx['max_scf_cycles']}",
        f"   FlipSpin",
    ]

    # Add FlipSpin targets for minority-spin metals
    for site_idx in ctx["minority_sites"]:
        metal = cluster_info.metals[site_idx]
        lines.append(f"   {{ {metal.label} 1 }}")

    lines.extend([
        "end",
        "",
        "* xyz",
        f"  {ctx['charge']} {ctx['spin_multiplicity']}",
    ])

    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        lines.append(f"  {elem} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")

    lines.extend(["*", ""])

    return "\n".join(lines)


def _gen_gaussian_bsdft(config, active_space, cluster_info, **kwargs):
    """Generate Gaussian BS-DFT input."""
    ctx = _build_context(config, active_space, cluster_info, "bsdft", **kwargs)

    lines = [
        f"%mem=64GB",
        f"%nprocshared=32",
        f"# {ctx['xc_functional']}/{ctx['basis_set']} Guess=Mix SCF=QC",
        "",
        f"{ctx['label']} — Broken-symmetry DFT",
        "",
        f"{ctx['charge']} {ctx['spin_multiplicity']}",
    ]

    for elem, pos in zip(cluster_info.all_elements, cluster_info.all_positions):
        lines.append(f" {elem} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")

    lines.extend(["", ""])

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Batch submission scripts
# ──────────────────────────────────────────────────────────────────

def generate_batch_submission(input_files: list,
                               scheduler: str = "slurm",
                               job_name: str = "active_space",
                               n_tasks: int = 1,
                               time_limit: str = "72:00:00",
                               **kwargs) -> str:
    """Generate a batch job submission script.

    Args:
        input_files: List of input file paths.
        scheduler: "slurm" or "pbs".
        job_name: Job name.
        n_tasks: Number of parallel tasks.
        time_limit: Wall time limit.
        **kwargs: Additional scheduler parameters.

    Returns:
        Job script content.
    """
    if scheduler.lower() == "slurm":
        return _gen_slurm_script(input_files, job_name, n_tasks, time_limit, **kwargs)
    elif scheduler.lower() == "pbs":
        return _gen_pbs_script(input_files, job_name, n_tasks, time_limit, **kwargs)
    else:
        return _gen_simple_script(input_files)


def _gen_slurm_script(input_files, job_name, n_tasks, time_limit, **kwargs):
    """Generate SLURM batch script."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --ntasks={n_tasks}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output=%j.out",
        "",
    ]

    if "mem" in kwargs:
        lines.append(f"#SBATCH --mem={kwargs['mem']}")

    lines.append("")
    lines.append("# Run calculations sequentially")
    for f in input_files:
        if isinstance(f, tuple):
            f = f[0]
        if f.endswith(".py"):
            lines.append(f"python {f}")
        else:
            lines.append(f"# Run: {f}")

    return "\n".join(lines)


def _gen_pbs_script(input_files, job_name, n_tasks, time_limit, **kwargs):
    """Generate PBS batch script."""
    lines = [
        "#!/bin/bash",
        f"#PBS -N {job_name}",
        f"#PBS -l nodes=1:ppn={n_tasks}",
        f"#PBS -l walltime={time_limit}",
        "",
    ]

    for f in input_files:
        if isinstance(f, tuple):
            f = f[0]
        if f.endswith(".py"):
            lines.append(f"python {f}")

    return "\n".join(lines)


def _gen_simple_script(input_files):
    """Generate a simple shell script."""
    lines = ["#!/bin/bash", ""]
    for f in input_files:
        if isinstance(f, tuple):
            f = f[0]
        lines.append(f"python {f}" if f.endswith(".py") else f"# {f}")
    return "\n".join(lines)
