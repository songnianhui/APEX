"""Module: LLM Agent Layer

Provides an interactive, reasoning-augmented interface on top of the core
analysis modules using the Anthropic API. The agent exposes tools for:

- Pre-calculation: structure analysis, active space building, spin/electronic
  config enumeration, filtering funnel design
- Peri-calculation: orbital construction, BS guess building, input generation
- Post-calculation: result parsing, energy extrapolation, population analysis

Usage:
    from apex_filter.llm_agent import run_agent

    result = run_agent("Analyze Fe7MoS9C for DMRG calculation at charge=-1, Sz=1.5")
"""

import json
import os
import sys
import traceback

import numpy as np

from .models import (
    CAS,
    ActiveSpaceLevel,
    ClusterInfo,
    ElectronicConfig,
    ExtrapolatedEnergy,
)



# ──────────────────────────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "analyze_structure",
        "description": (
            "Parse a molecular structure file (XYZ/PDB) and identify metal centers, "
            "bridging atoms, terminal ligands, and approximate symmetry. "
            "Returns a ClusterInfo summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to structure file (XYZ or PDB).",
                },
                "charge": {
                    "type": "integer",
                    "description": "Total charge of the cluster.",
                    "default": 0,
                },
                "target_spin": {
                    "type": "number",
                    "description": "Target spin S (e.g., 1.5 for S=3/2).",
                    "default": 0.0,
                },
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "build_active_space",
        "description": (
            "Construct an active space (n_electrons, n_orbitals) for a cluster "
            "based on its metal composition and ligand environment. "
            "Supports minimal (metal d only), standard (d + bridging p), "
            "and extended (d + p + peripheral) levels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to structure file.",
                },
                "charge": {
                    "type": "integer",
                    "default": 0,
                },
                "target_spin": {
                    "type": "number",
                    "default": 0.0,
                },
                "level": {
                    "type": "string",
                    "enum": ["minimal", "standard", "extended"],
                    "default": "standard",
                    "description": "Active space level.",
                },
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "enumerate_spin_configs",
        "description": (
            "Enumerate all collinear broken-symmetry spin isomers for a cluster. "
            "Each metal site's spin points up (+1) or down (-1), with the constraint "
            "that sum(sign_i * S_i) = target Sz. Optionally group into symmetry families."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metal element symbols, e.g., ['Fe','Fe','Fe','Mo'].",
                },
                "target_sz": {
                    "type": "number",
                    "description": "Target total Sz.",
                },
                "oxidation_states": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Oxidation state for each metal, e.g., [3,3,3,3].",
                },
                "symmetry": {
                    "type": "string",
                    "default": "C1",
                    "description": "Approximate point group, e.g., 'C3'.",
                },
            },
            "required": ["metals", "target_sz"],
        },
    },
    {
        "name": "enumerate_electronic_configs",
        "description": (
            "Enumerate all electronic configurations as the Cartesian product of "
            "spin isomers × oxidation state assignments × d-orbital choices. "
            "Returns the total count and a summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to structure file.",
                },
                "charge": {
                    "type": "integer",
                    "default": 0,
                },
                "target_spin": {
                    "type": "number",
                    "default": 0.0,
                },
                "max_configs": {
                    "type": "integer",
                    "description": "Maximum configs to enumerate (for large systems).",
                },
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "design_filtering_protocol",
        "description": (
            "Design a hierarchical filtering funnel for reducing the number of "
            "electronic configurations through progressively more accurate "
            "(and expensive) calculations: UHF → CCSD → CCSD(T) → DMRG → CCSDTQ."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_configs": {
                    "type": "integer",
                    "description": "Total number of electronic configurations.",
                },
                "n_electrons": {
                    "type": "integer",
                    "description": "Active space electrons.",
                },
                "n_orbitals": {
                    "type": "integer",
                    "description": "Active space orbitals.",
                },
                "n_spin_isomers": {
                    "type": "integer",
                    "description": "Number of spin isomers.",
                },
                "style": {
                    "type": "string",
                    "enum": ["femoco", "conservative", "minimal"],
                    "default": "femoco",
                },
            },
            "required": ["n_configs", "n_electrons", "n_orbitals"],
        },
    },
    {
        "name": "generate_input_file",
        "description": (
            "Generate a QC input file for a specific electronic configuration. "
            "Supports PySCF (UHF/CCSD/CASSCF), BLOCK2 (DMRG), ORCA, and Gaussian."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to structure file.",
                },
                "charge": {"type": "integer", "default": 0},
                "target_spin": {"type": "number", "default": 0.0},
                "code": {
                    "type": "string",
                    "default": "pyscf",
                    "description": "QC code: pyscf, block2, orca, gaussian.",
                },
                "method": {
                    "type": "string",
                    "default": "uhf",
                    "description": "Method: uhf, ccsd, casscf, dmrg.",
                },
                "basis": {
                    "type": "string",
                    "default": "cc-pVDZ",
                },
                "n_configs": {
                    "type": "integer",
                    "default": 1,
                    "description": "Number of input files to generate.",
                },
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "parse_results",
        "description": (
            "Parse QC output files to extract energies, convergence status, "
            "and other results. Supports PySCF checkpoint, BLOCK2 output, "
            "and HAST-UCC output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to output/checkpoint file.",
                },
                "format": {
                    "type": "string",
                    "enum": ["pyscf_uhf", "pyscf_ccsd", "block2_dmrg", "hast_ucc"],
                    "description": "Output file format.",
                },
            },
            "required": ["filepath", "format"],
        },
    },
    {
        "name": "extrapolate_energy",
        "description": (
            "Extrapolate energies to the complete basis/bond-dimension limit. "
            "Supports DMRG D-extrapolation, CC composite energy, "
            "FNO threshold extrapolation, and MP2 space correction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": [
                        "dmrg_d_extrapolation",
                        "cc_composite",
                        "fno_extrapolation",
                        "mp2_space_correction",
                    ],
                    "description": "Extrapolation method.",
                },
                "params": {
                    "type": "object",
                    "description": "Method-specific parameters (e.g., bond_dims, energies).",
                },
            },
            "required": ["method", "params"],
        },
    },
    {
        "name": "query_knowledge_base",
        "description": (
            "Look up properties of transition metals, ligands, cluster templates, "
            "or basis sets from the built-in knowledge base. "
            "Useful for understanding expected behavior of specific elements."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["transition_metals", "ligands", "cluster_templates", "basis_sets"],
                    "description": "Knowledge base category.",
                },
                "element": {
                    "type": "string",
                    "description": "Element to look up (e.g., 'Fe', 'Mo').",
                },
            },
            "required": ["category"],
        },
    },
]


# ──────────────────────────────────────────────────────────────────
# Tool dispatch
# ──────────────────────────────────────────────────────────────────

def _tool_analyze_structure(params):
    from apex_cas.structure_analyzer import parse_structure
    info = parse_structure(
        params["filepath"],
        charge=params.get("charge", 0),
        target_spin=params.get("target_spin", 0.0),
    )
    return _serialize_cluster_info(info)


def _tool_build_active_space(params):
    from apex_cas.structure_analyzer import parse_structure
    from apex_cas.CAS_builder_noncomputing import build_NC_CAS
    info = parse_structure(
        params["filepath"],
        charge=params.get("charge", 0),
        target_spin=params.get("target_spin", 0.0),
    )
    level_map = {
        "minimal": ActiveSpaceLevel.MINIMAL,
        "standard": ActiveSpaceLevel.STANDARD,
        "extended": ActiveSpaceLevel.EXTENDED,
    }
    cases, _ = build_NC_CAS(info, level_map[params.get("level", "standard")])
    aspace = cases["rule"]
    return {
        "n_electrons": aspace.n_electrons,
        "n_orbitals": aspace.n_orbitals,
        "n_qubits": 2 * aspace.n_orbitals,
        "description": aspace.description,
        "level": params.get("level", "standard"),
    }


def _tool_enumerate_spin_configs(params):
    from .models import MetalCenter
    from .spin_config import enumerate_spin_isomers, apply_symmetry_reduction, label_isomers
    from apex_cas.CAS_builder_noncomputing import get_local_spin, get_common_oxidation_states

    metals = []
    for i, elem in enumerate(params["metals"]):
        metals.append(MetalCenter(
            element=elem, index=i,
            position=np.zeros(3), label=f"{elem}{i+1}",
        ))

    cluster = ClusterInfo(
        metals=metals,
        target_spin=params["target_sz"],
        symmetry_group=params.get("symmetry", "C1"),
    )

    ox_states = None
    if "oxidation_states" in params:
        ox_states = {i: ox for i, ox in enumerate(params["oxidation_states"])}

    isomers = enumerate_spin_isomers(cluster, target_Sz=params["target_sz"],
                                      oxidation_states=ox_states)

    metal_positions = np.array([m.position for m in metals])
    families = apply_symmetry_reduction(
        isomers, params.get("symmetry", "C1"), metal_positions
    )
    families = label_isomers(families)

    return {
        "total_isomers": len(isomers),
        "n_families": len(families),
        "families": [
            {
                "label": f.label,
                "n_minority": f.n_minority,
                "n_isomers": len(f.isomers),
                "representative": {
                    "spin_assignment": {str(k): v for k, v in f.representative.spin_assignment.items()},
                    "Sz": f.representative.Sz,
                },
            }
            for f in families
        ],
    }


def _tool_enumerate_electronic_configs(params):
    from apex_cas.structure_analyzer import parse_structure
    from apex_cas.CAS_builder_noncomputing import build_NC_CAS
    from .spin_config import enumerate_spin_isomers, apply_symmetry_reduction, label_isomers
    from .electronic_config import generate_all_configs, estimate_computational_cost

    info = parse_structure(
        params["filepath"],
        charge=params.get("charge", 0),
        target_spin=params.get("target_spin", 0.0),
    )
    cases, _ = build_NC_CAS(info)
    aspace = cases["rule"]
    isomers = enumerate_spin_isomers(info)

    max_cfg = params.get("max_configs")
    configs = generate_all_configs(isomers, info, max_configs=max_cfg)

    cost = estimate_computational_cost(
        len(configs), aspace.n_electrons, aspace.n_orbitals
    )

    return {
        "total_configs": len(configs),
        "n_spin_isomers": len(isomers),
        "active_space": {
            "n_electrons": aspace.n_electrons,
            "n_orbitals": aspace.n_orbitals,
        },
        "cost_estimate": cost.get("recommendation", "N/A"),
    }


def _tool_design_filtering_protocol(params):
    from .filtering import design_filtering_funnel

    aspace = CAS(
        n_electrons=params["n_electrons"],
        n_orbitals=params["n_orbitals"],
    )

    plan = design_filtering_funnel(
        params["n_configs"], aspace, params.get("n_spin_isomers"),
        style=params.get("style", "femoco"),
    )

    return {
        "total_configs": plan.total_configs,
        "levels": [
            {
                "method": level.method,
                "n_input": level.n_input,
                "n_output": level.n_output,
                "criterion": level.selection_criterion,
            }
            for level in plan.levels
        ],
        "final_n_configs": plan.levels[-1].n_output if plan.levels else 0,
    }


def _tool_generate_input_file(params):
    from apex_cas.structure_analyzer import parse_structure
    from apex_cas.CAS_builder_noncomputing import build_NC_CAS
    from .spin_config import enumerate_spin_isomers
    from .electronic_config import generate_all_configs
    from .input_generator import generate_input

    info = parse_structure(
        params["filepath"],
        charge=params.get("charge", 0),
        target_spin=params.get("target_spin", 0.0),
    )
    cases, _ = build_NC_CAS(info)
    aspace = cases["rule"]
    isomers = enumerate_spin_isomers(info)
    configs = generate_all_configs(isomers, info, max_configs=params.get("n_configs", 1))

    if not configs:
        return {"error": "No electronic configurations generated"}

    content = generate_input(
        configs[0], aspace, info,
        code=params.get("code", "pyscf"),
        method=params.get("method", "uhf"),
        basis_set=params.get("basis", "cc-pVDZ"),
    )
    return {"content_preview": content[:500], "total_length": len(content)}


def _tool_parse_results(params):
    from .result_parser import (
        parse_pyscf_uhf, parse_pyscf_ccsd,
        parse_block2_dmrg, parse_hast_output,
        check_convergence,
    )
    parsers = {
        "pyscf_uhf": parse_pyscf_uhf,
        "pyscf_ccsd": parse_pyscf_ccsd,
        "block2_dmrg": parse_block2_dmrg,
        "hast_ucc": parse_hast_output,
    }
    parser = parsers.get(params["format"])
    if not parser:
        return {"error": f"Unknown format: {params['format']}"}

    result = parser(params["filepath"])
    result["converged_ok"] = check_convergence(result)
    return _make_serializable(result)


def _tool_extrapolate_energy(params):
    from .energy_extrapolation import (
        dmrg_d_extrapolation, cc_composite_energy,
        fno_extrapolation, mp2_space_correction,
    )
    method = params["method"]
    p = params["params"]

    if method == "dmrg_d_extrapolation":
        result = dmrg_d_extrapolation(p["bond_dims"], p["energies"])
    elif method == "cc_composite":
        result = cc_composite_energy(p["e_ccsdt_full"], p["e_ccsdtq_fno"], p["e_ccsdt_fno"])
    elif method == "fno_extrapolation":
        result = fno_extrapolation(p["thresholds"], p["energies"])
    elif method == "mp2_space_correction":
        result = mp2_space_correction(p["e_small_cas"], p["e_mp2_small"], p["e_mp2_large"])
    else:
        return {"error": f"Unknown method: {method}"}

    return {
        "method": result.method,
        "energy": result.energy,
        "uncertainty": result.uncertainty,
        "description": result.description,
    }


def _tool_query_knowledge_base(params):
    import yaml
    from ._paths import data_file as _kb_file
    category = params["category"]
    element = params.get("element")

    file_map = {
        "transition_metals": "transition_metals.yaml",
        "ligands": "ligand_database.yaml",
        "cluster_templates": "cluster_templates.yaml",
        "basis_sets": "basis_sets.yaml",
    }

    filename = file_map.get(category)
    if not filename:
        return {"error": f"Unknown category: {category}"}

    filepath = _kb_file(filename)
    if not os.path.exists(filepath):
        return {"error": f"Knowledge base file not found: {filepath}"}

    with open(filepath) as f:
        data = yaml.safe_load(f)

    if element and isinstance(data, dict):
        if element in data:
            return _make_serializable({element: data[element]})
        return {"info": f"Element '{element}' not found in {category}", "available": list(data.keys())}

    return _make_serializable(data)


TOOL_DISPATCH = {
    "analyze_structure": _tool_analyze_structure,
    "build_active_space": _tool_build_active_space,
    "enumerate_spin_configs": _tool_enumerate_spin_configs,
    "enumerate_electronic_configs": _tool_enumerate_electronic_configs,
    "design_filtering_protocol": _tool_design_filtering_protocol,
    "generate_input_file": _tool_generate_input_file,
    "parse_results": _tool_parse_results,
    "extrapolate_energy": _tool_extrapolate_energy,
    "query_knowledge_base": _tool_query_knowledge_base,
}


# ──────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────

def _serialize_cluster_info(info):
    """Convert ClusterInfo to JSON-serializable dict."""
    result = {
        "formula": info.formula,
        "total_charge": info.total_charge,
        "target_spin": info.target_spin,
        "symmetry": info.symmetry_group,
        "n_metals": len(info.metals),
        "metals": [
            {
                "element": m.element,
                "index": m.index,
                "label": m.label,
            }
            for m in info.metals
        ],
        "n_bridging": len(info.bridging_atoms),
        "bridging_atoms": [
            {
                "element": b.element,
                "index": b.index,
                "bridged_metals": b.bridged_metals,
            }
            for b in info.bridging_atoms
        ],
    }
    return result


def _make_serializable(obj):
    """Convert numpy types and dataclasses to JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, "__dataclass_fields__"):
        return _make_serializable({
            k: getattr(obj, k) for k in obj.__dataclass_fields__
        })
    return obj


# ──────────────────────────────────────────────────────────────────
# Agent runner
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert quantum chemistry assistant specialized in
active space analysis for transition metal clusters. You help users:

1. Analyze molecular structures to identify metal centers and ligands
2. Build appropriate active spaces for multi-reference calculations
3. Enumerate spin isomers and electronic configurations
4. Design filtering protocols for high-accuracy calculations
5. Generate input files for QC codes (PySCF, BLOCK2, HAST-UCC)
6. Parse and extrapolate energies from calculation results

Use the available tools to perform analysis. Explain your reasoning clearly.
When results are ambiguous, discuss the alternatives and trade-offs.

Key reference: FeMo-cofactor (Fe₇MoS₉C) has (113e, 76o) active space,
35 spin isomers at Sz=2.0 (all Fe(III)+Mo(III)), and ~78,750 electronic
configurations when mixed oxidation states are considered.
"""


def run_agent(user_message: str, max_turns: int = 20, model: str = "claude-sonnet-4-20250514") -> str:
    """Run an interactive agent session with tool use.

    Args:
        user_message: The user's request.
        max_turns: Maximum number of tool-use rounds.
        model: Anthropic model to use.

    Returns:
        Final assistant response text.
    """
    try:
        import anthropic
    except ImportError:
        return ("Error: 'anthropic' package not installed. "
                "Install with: pip install anthropic")

    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[{"type": "custom", **t} for t in TOOL_DEFINITIONS],
            messages=messages,
        )

        # Process response
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Check if we need to handle tool calls
        tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No tool calls — return text response
            text_blocks = [b for b in assistant_content if b.type == "text"]
            return "\n".join(b.text for b in text_blocks)

        # Execute tool calls
        tool_results = []
        for block in tool_use_blocks:
            try:
                handler = TOOL_DISPATCH.get(block.name)
                if handler:
                    result = handler(block.input)
                else:
                    result = {"error": f"Unknown tool: {block.name}"}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {str(e)}"}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, indent=2, default=str),
            })

        messages.append({"role": "user", "content": tool_results})

    return "Agent reached maximum turns without completing."


def run_interactive():
    """Run an interactive agent loop from the command line."""
    print("APEX (LLM-powered)")
    print("Type your question or 'quit' to exit.")
    print()

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        response = run_agent(user_input)
        print(response)
        print()


if __name__ == "__main__":
    run_interactive()
