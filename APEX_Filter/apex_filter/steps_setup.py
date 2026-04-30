"""Session setup steps for the interactive APEX_Filter pipeline."""

import os

from .session import SessionManager


def _validate_active_space_inputs(cas, fcid):
    """Fail early if CAS metadata and FCIDUMP dimensions disagree."""
    if cas.n_orbitals != fcid.norb:
        raise ValueError(
            "CAS/FCIDUMP orbital mismatch: "
            f"CAS has {cas.n_orbitals} orbitals but FCIDUMP has NORB={fcid.norb}"
        )
    if cas.n_electrons != fcid.nelec:
        raise ValueError(
            "CAS/FCIDUMP electron mismatch: "
            f"CAS has {cas.n_electrons} electrons but FCIDUMP has NELEC={fcid.nelec}"
        )


def step_load(config_path: str, session_dir: str):
    """Load CAS + FCIDUMP + ClusterInfo from a YAML config file."""
    from .CAS_loader import load_filter_inputs

    print("=" * 60)
    print("Step 1: Loading CAS + FCIDUMP from config")
    print("=" * 60)

    inputs = load_filter_inputs(config_path)
    cas = inputs.cas
    fcid = inputs.fcidump_data
    ci = inputs.cluster_info
    settings = inputs.settings

    _validate_active_space_inputs(cas, fcid)

    fcidump_path = os.path.abspath(inputs.fcidump_path)

    print(f"  CAS      : ({cas.n_electrons}e, {cas.n_orbitals}o)")
    print(f"  FCIDUMP  : NORB={fcid.norb}, NELEC={fcid.nelec}, MS2={fcid.ms2}")
    print(f"  Metals   : {', '.join(m.element for m in ci.metals)}")
    print(f"  Charge={ci.total_charge}, Target S={ci.target_spin}")

    sm = SessionManager(session_dir)
    sm.create()
    sm.save_load_state(
        ci,
        cas,
        fcidump_path,
        settings,
        os.path.abspath(config_path),
        apex_cas_provenance=inputs.config_raw.get("_apex_cas_provenance"),
    )

    print(f"  Session saved to: {session_dir}")
    print("Step 1 complete.")
