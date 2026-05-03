"""Orbital visualization utilities for V1.0.0 active-space review artifacts."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional as _Optional

import numpy as np
from pyscf.tools import cubegen

from .ao_shell_analysis import _generate_ao_shell_markdown
from shared.cluster_info_labels import (
    require_authoritative_cluster_info as _require_authoritative_cluster_info,
    resolve_explicit_label as _resolve_explicit_label,
    resolve_metal_site_label as _resolve_metal_site_label,
)

logger = logging.getLogger(__name__)


def _generate_orbital_report(
    mol,
    mo_coeff_loc: np.ndarray,
    occupations: np.ndarray,
    labels: list[str],
    cluster_info=None,
    output_path: str = "orbital_report.md",
    occ_active_lo: float = 0.02,
    occ_active_hi: float = 1.98,
    active_indices: _Optional[list[int]] = None,
    cas_type: str = "",
    selection_method: str = "",
    projection_weights: _Optional[np.ndarray] = None,
    projection_weights_metal: _Optional[np.ndarray] = None,
    projection_weights_bridging: _Optional[np.ndarray] = None,
) -> str:
    """Generate a markdown orbital report for human review and selection."""
    nmo = len(occupations)
    aoslices = mol.aoslice_by_atom()
    ao_labels = mol.ao_labels()
    atom_labels = _build_atom_labels(mol, cluster_info)
    all_ao_contribs = _precompute_ao_contributions(
        mol,
        mo_coeff_loc,
        aoslices,
        ao_labels,
        atom_labels=atom_labels,
        top_n=3,
    )
    active_index_set = set(active_indices) if active_indices is not None else None

    orbital_rows = []
    n_core = 0
    n_active_noon = 0
    n_virtual = 0

    for i in range(nmo):
        occ = float(occupations[i])
        if occ > occ_active_hi:
            block = "core"
            n_core += 1
        elif occ < occ_active_lo:
            block = "virtual"
            n_virtual += 1
        else:
            block = "active"
            n_active_noon += 1

        ao_contrib = all_ao_contribs[i]
        is_selected = i in active_index_set if active_index_set is not None else block == "active"
        proj_w = float(projection_weights[i]) if projection_weights is not None else 0.0
        proj_w_m = float(projection_weights_metal[i]) if projection_weights_metal is not None else 0.0
        proj_w_b = float(projection_weights_bridging[i]) if projection_weights_bridging is not None else 0.0
        character = _detect_orbital_character(occ, block, proj_w, occ_active_lo, occ_active_hi)
        ao_str = ", ".join(f"{k}:{v:.3f}" for k, v in ao_contrib.items())
        passed_label = labels[i] if i < len(labels) else ""
        display_label = passed_label if passed_label else (_best_chemical_label(ao_contrib) or f"orb_{i}")

        orbital_rows.append(
            {
                "idx": i,
                "occ": round(occ, 6),
                "block": block,
                "label": display_label,
                "selected": is_selected,
                "character": character,
                "proj_weight": round(proj_w, 4),
                "proj_wt_metal": round(proj_w_m, 4),
                "proj_wt_bridging": round(proj_w_b, 4),
                "ao_contrib": ao_str,
            }
        )

    n_active = len(active_indices) if active_indices is not None else n_active_noon
    total_electrons = round(float(np.sum(occupations)), 2)

    lines = []
    lines.append("# Orbital Report")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("| key | value |")
    lines.append("|-----|-------|")
    lines.append(f"| cas_type | {cas_type} |")
    lines.append(f"| selection_method | {selection_method} |")
    lines.append(f"| n_total | {nmo} |")
    lines.append(f"| n_core | {n_core} |")
    lines.append(f"| n_active | {n_active} |")
    lines.append(f"| n_virtual | {n_virtual} |")
    lines.append(f"| threshold_active_lo | {occ_active_lo} |")
    lines.append(f"| threshold_active_hi | {occ_active_hi} |")
    lines.append(f"| total_electrons | {total_electrons} |")
    if cluster_info is not None:
        annotation_source = getattr(cluster_info, "annotation_source", "")
        cluster_info_path = getattr(cluster_info, "cluster_info_path", "")
        if annotation_source:
            lines.append(f"| atom_annotations | {annotation_source} |")
        if cluster_info_path:
            lines.append(f"| cluster_info_path | {cluster_info_path} |")
    lines.append("")

    if cluster_info is not None:
        lines.extend(_build_atom_roles_markdown(cluster_info))

    ao_analysis_md = _generate_ao_shell_markdown(mol, cluster_info)
    if ao_analysis_md:
        lines.append(ao_analysis_md)

    lines.append("## Orbitals")
    lines.append("")
    lines.append("<!-- Edit the 'selected' column to choose active orbitals, then run: apex-cas fcidump -->")
    lines.append("| idx | occ | block | label | selected | character | proj_weight | proj_wt_M | proj_wt_B | ao_contrib |")
    lines.append("|-----|------|-------|-------|----------|-----------|-------------|-----------|-----------|------------|")

    for row in orbital_rows:
        sel_str = "true" if row["selected"] else "false"
        lines.append(
            f"| {row['idx']} | {row['occ']:.6f} | {row['block']} | {row['label']} | {sel_str} | "
            f"{row['character']} | {row['proj_weight']:.4f} | {row['proj_wt_metal']:.4f} | "
            f"{row['proj_wt_bridging']:.4f} | {row['ao_contrib']} |"
        )

    lines.append("")
    lines.append("## Column Definitions")
    lines.append("")
    lines.append("| Column | Meaning |")
    lines.append("|--------|---------|")
    lines.append("| `idx` | MO index in the localized UNO basis. |")
    lines.append("| `occ` | UNO occupation number from the total natural-orbital density diagonalization. |")
    lines.append("| `block` | NOON-based classification: core / active / virtual. |")
    lines.append("| `label` | Chemical label from all-atom Mulliken AO contributions. |")
    lines.append("| `selected` | Current active-space choice; users may edit this manually. |")
    lines.append("| `character` | Heuristic chemical role inferred from occupation and projection weight. |")
    lines.append("| `proj_weight` | Total metal-d/f + bridging-p projection weight. |")
    lines.append("| `proj_wt_M` | Projection weight onto metal d/f shells only. |")
    lines.append("| `proj_wt_B` | Projection weight onto bridging p shells only. |")
    lines.append("| `ao_contrib` | Top Mulliken AO contributions, used to assign chemical labels. |")
    lines.append("")
    lines.append("`proj_weight = proj_wt_M + proj_wt_B`.")
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines))

    logger.info(
        "Orbital report written to %s (%d orbitals: %d core, %d active, %d virtual, method=%s)",
        output_path,
        nmo,
        n_core,
        n_active,
        n_virtual,
        selection_method or "noon",
    )
    return os.path.abspath(output_path)


def _generate_orbital_cubes(
    mol,
    mo_coeff: np.ndarray,
    indices: _Optional[list[int]] = None,
    labels: _Optional[list[str]] = None,
    output_dir: str = "cubes",
    prefix: str = "orb",
    nx: int = 80,
    ny: int = 80,
    nz: int = 80,
) -> list[str]:
    """Generate Gaussian cube files for selected orbitals."""
    os.makedirs(output_dir, exist_ok=True)
    if indices is None:
        indices = list(range(mo_coeff.shape[1]))

    paths = []
    for i in indices:
        if labels and i < len(labels) and labels[i]:
            safe_label = _sanitize_label(labels[i])
            filepath = os.path.join(output_dir, f"{prefix}_{i:04d}_{safe_label}.cube")
        else:
            filepath = os.path.join(output_dir, f"{prefix}_{i:04d}.cube")
        cubegen.orbital(mol, filepath, mo_coeff[:, i], nx=nx, ny=ny, nz=nz)
        paths.append(filepath)

    logger.info("Generated %d cube files in %s", len(paths), output_dir)
    return paths


def _generate_noon_plot(
    occupations: np.ndarray,
    labels: _Optional[list[str]] = None,
    output_path: str = "noon_plot.png",
    active_lo: float = 0.02,
    active_hi: float = 1.98,
    show_top_n: int = 50,
) -> str:
    """Generate a NOON dot-line plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping NOON plot")
        return ""

    nmo = len(occupations)
    x = np.arange(nmo)
    active_mask = np.array([(active_lo <= occ <= active_hi) for occ in occupations])
    inactive_mask = ~active_mask

    fig, ax = plt.subplots(figsize=(max(12, nmo * 0.06), 5))
    ax.plot(x, occupations, color="#333333", linewidth=0.8, zorder=1)
    ax.scatter(x[inactive_mask], occupations[inactive_mask], color="#aaaaaa", s=10, zorder=2)
    ax.scatter(x[active_mask], occupations[active_mask], color="#d62728", s=14, zorder=3)
    ax.axhline(y=active_hi, color="blue", linestyle="--", linewidth=0.8, label=f"occ = {active_hi}")
    ax.axhline(y=active_lo, color="blue", linestyle="--", linewidth=0.8, label=f"occ = {active_lo}")
    ax.set_xlabel("Orbital index")
    ax.set_ylabel("UNO occupation number")
    ax.set_title("Natural Orbital Occupation Numbers (NOON)")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 2.05)

    if labels:
        active_indices = [i for i in range(nmo) if active_lo <= occupations[i] <= active_hi]
        for i in active_indices[:show_top_n]:
            label = labels[i] if i < len(labels) else ""
            if label:
                ax.annotate(label, (i, occupations[i]), textcoords="offset points", xytext=(0, 5), fontsize=4, rotation=90, ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("NOON plot saved to %s", output_path)
    return os.path.abspath(output_path)


_GALLERY_SERVER_SCRIPT = """#!/usr/bin/env python3
\"\"\"Launch a local HTTP server to view the orbital gallery.

Usage:
    python {script_name}
\"\"\"
import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 0  # 0 = OS picks a free port
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

with socketserver.TCPServer((\"\", PORT), Handler) as httpd:
    port = httpd.server_address[1]
    # Find the gallery HTML file
    html_files = [f for f in os.listdir(DIRECTORY) if f.endswith(\"_orbital_gallery.html\")]
    if html_files:
        url = f\"http://localhost:{port}/{html_files[0]}\"
    else:
        url = f\"http://localhost:{port}/\"
    print(f\"Serving {DIRECTORY}\")
    print(f\"Open: {url}\")
    print(\"Press Ctrl+C to stop.\")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(\"\\nServer stopped.\")
"""


def _batch_render_orbital_html(
    cube_paths: list[str],
    labels: list[str],
    occupations: list[float],
    output_dir: str,
    isovalue: float = 0.04,
    opacity: float = 0.85,
    pos_color: str = "blue",
    neg_color: str = "red",
    bg_color: str = "black",
    gallery_name: str = "orbital_gallery.html",
) -> str:
    """Render multiple cube files into a single 3Dmol gallery HTML."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(cube_paths)
    if n == 0:
        logger.warning("No cube files provided for HTML gallery generation")
        return ""

    while len(labels) < n:
        labels.append(os.path.basename(cube_paths[len(labels)]))
    while len(occupations) < n:
        occupations.append(0.0)

    html_abs = os.path.abspath(os.path.join(output_dir, gallery_name))
    cube_rel_paths = [
        os.path.relpath(os.path.abspath(path), os.path.dirname(html_abs)).replace("\\", "/")
        for path in cube_paths
    ]
    cube_paths_js = "[" + ", ".join(f'"{path}"' for path in cube_rel_paths) + "]"
    nav_items = "\n".join(
        f'<button onclick="show({i})" id="btn_{i}" title="{labels[i]} (occ={occupations[i]:.4f})">'
        f'{labels[i]} <span style="color:#888;">occ={occupations[i]:.4f}</span></button>'
        for i in range(n)
    )
    js_cdn = "https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Orbital Gallery</title>
<script src="{js_cdn}"></script>
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; width:100%; height:100%; background:{bg_color}; font-family:monospace; overflow:hidden; }}
body {{ display:flex; }}
#sidebar {{ width:300px; min-width:300px; height:100vh; background:#1a1a1a; border-right:1px solid #333; display:flex; flex-direction:column; overflow:hidden; }}
#sidebar-title {{ padding:8px 10px; color:#aaa; font-size:12px; background:#222; border-bottom:1px solid #333; flex-shrink:0; }}
#btn-list {{ flex:1; overflow-y:auto; padding:4px; }}
#btn-list button {{ display:block; width:100%; text-align:left; margin:1px 0; padding:5px 8px; cursor:pointer; font-family:monospace; font-size:11px; line-height:1.4; border:1px solid #444; background:#2a2a2a; color:#ddd; border-radius:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
#btn-list button:hover {{ background:#444; }}
#btn-list button.active {{ background:#4477AA; color:white; border-color:#4477AA; }}
#main {{ flex:1; display:flex; flex-direction:column; height:100vh; overflow:hidden; }}
#info {{ padding:6px 12px; background:#1a1a1a; border-bottom:1px solid #333; color:#ccc; font-size:13px; text-align:center; flex-shrink:0; }}
#viewer-wrap {{ flex:1; display:flex; align-items:center; justify-content:center; position:relative; }}
#viewer {{ width:100%; height:100%; }}
#loading {{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#888; font-size:16px; z-index:200; display:none; pointer-events:none; }}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sidebar-title">Orbital Gallery</div>
  <div id="btn-list">{nav_items}</div>
</div>
<div id="main">
  <div id="info"></div>
  <div id="viewer-wrap">
    <div id="viewer"></div>
    <div id="loading">Loading...</div>
  </div>
</div>
<script>
var cubePaths = {cube_paths_js};
var currentViewer = null;
var n = {n};
var isoval = {isovalue};
var opacity = {opacity};
var posColor = "{pos_color}";
var negColor = "{neg_color}";
var bgColor = "{bg_color}";
function loadOrbital(idx) {{
    var viewer = document.getElementById("viewer");
    viewer.innerHTML = "";
    currentViewer = $3Dmol.createViewer(viewer, {{backgroundColor: bgColor}});
    var cubeUrl = cubePaths[idx];
    document.getElementById("loading").style.display = "block";
    fetch(cubeUrl).then(r => r.text()).then(data => {{
        currentViewer.addVolumetricData(data, "cube", {{isoval: isoval, color: posColor, opacity: opacity, smoothness: 5}});
        currentViewer.addVolumetricData(data, "cube", {{isoval: -isoval, color: negColor, opacity: opacity, smoothness: 5}});
        currentViewer.addModel(data, "cube");
        currentViewer.setStyle(
            {{}},
            {{
                stick: {{radius: 0.045, colorscheme: "Jmol"}},
            }}
        );
        currentViewer.addStyle({{elem: "H"}}, {{sphere: {{scale: 0.14, colorscheme: "Jmol"}}}});
        currentViewer.addStyle({{elem: "C"}}, {{sphere: {{scale: 0.20, colorscheme: "Jmol"}}}});
        currentViewer.addStyle({{elem: "S"}}, {{sphere: {{scale: 0.24, colorscheme: "Jmol"}}}});
        currentViewer.addStyle({{elem: "Fe"}}, {{sphere: {{scale: 0.27, colorscheme: "Jmol"}}}});
        currentViewer.zoomTo();
        currentViewer.render();
        document.getElementById("loading").style.display = "none";
    }}).catch(err => {{
        document.getElementById("loading").innerText = "Error: " + err.message;
    }});
}}
function show(idx) {{
    for (var i = 0; i < n; i++) {{
        var btn = document.getElementById("btn_" + i);
        if (btn) btn.className = (i === idx) ? "active" : "";
    }}
    var btn = document.getElementById("btn_" + idx);
    document.getElementById("info").innerText = btn.textContent;
    btn.scrollIntoView({{block: "nearest", behavior: "smooth"}});
    loadOrbital(idx);
}}
show(0);
document.addEventListener("keydown", function(e) {{
    var current = 0;
    for (var i = 0; i < n; i++) {{
        if (document.getElementById("btn_" + i).className === "active") {{ current = i; break; }}
    }}
    if (e.key === "ArrowRight" || e.key === "ArrowDown") {{
        show(Math.min(current + 1, n - 1)); e.preventDefault();
    }} else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {{
        show(Math.max(current - 1, 0)); e.preventDefault();
    }}
}});
</script>
</body>
</html>"""

    html_path = os.path.abspath(os.path.join(output_dir, gallery_name))
    with open(html_path, "w") as handle:
        handle.write(html)

    launcher_name = gallery_name.replace(".html", "_server.py")
    launcher_path = os.path.join(output_dir, launcher_name)
    with open(launcher_path, "w") as handle:
        handle.write(_GALLERY_SERVER_SCRIPT)

    logger.info("Orbital gallery HTML written to %s (%d orbitals)", html_path, n)
    return html_path


def plot_orbitals(
    cas,
    mol,
    output_dir: str,
    cluster_info=None,
    generate_cubes: bool = True,
    cube_grid: str = "80x80x80",
    occ_active_lo: float = 0.02,
    occ_active_hi: float = 1.98,
    stem: str = "",
    pw_plot_threshold: _Optional[float] = None,
    render_png: bool = True,
    png_isovalue: float = 0.05,
) -> dict:
    """Generate report, NOON plot, thresholded cubes, and gallery for a CAS."""
    mo_coeff_loc = cas.mo_coeff_full
    occ = cas.occupations_full
    labels = cas.orbital_labels_full
    if mo_coeff_loc is None or occ is None:
        raise ValueError("CAS.mo_coeff_full and CAS.occupations_full must be populated.")

    os.makedirs(output_dir, exist_ok=True)
    chemical_labels = _ensure_chemical_labels(mol, mo_coeff_loc, labels, cluster_info)
    report_path = os.path.join(output_dir, f"{stem}_orbital_report.md")
    _generate_orbital_report(
        mol,
        mo_coeff_loc,
        occ,
        chemical_labels,
        cluster_info=cluster_info,
        output_path=report_path,
        occ_active_lo=occ_active_lo,
        occ_active_hi=occ_active_hi,
        active_indices=cas.active_indices,
        cas_type=cas.cpt_cas_type or "",
        selection_method=cas.selection_method or "",
        projection_weights=cas.projection_weights,
        projection_weights_metal=cas.projection_weights_metal,
        projection_weights_bridging=cas.projection_weights_bridging,
    )

    noon_path = os.path.join(output_dir, f"{stem}_noon_plot.png")
    _generate_noon_plot(occ, chemical_labels, output_path=noon_path, active_lo=occ_active_lo, active_hi=occ_active_hi)

    proj_weights = cas.projection_weights
    if proj_weights is not None and pw_plot_threshold is not None:
        selected_indices = [
            i for i in range(mo_coeff_loc.shape[1])
            if proj_weights[i] >= pw_plot_threshold and occ[i] >= occ_active_lo
        ]
    else:
        selected_indices = [i for i in range(mo_coeff_loc.shape[1]) if occ_active_lo <= occ[i] <= occ_active_hi]

    cube_dir = ""
    cube_paths: list[str] = []
    if generate_cubes:
        nx, ny, nz = (int(part) for part in cube_grid.split("x"))
        cube_dir = os.path.join(output_dir, "cubes")
        cube_paths = _generate_orbital_cubes(
            mol,
            mo_coeff_loc,
            indices=selected_indices,
            labels=chemical_labels,
            output_dir=cube_dir,
            prefix=f"{stem}_orb",
            nx=nx,
            ny=ny,
            nz=nz,
        )

    html_gallery_path = ""
    if render_png and cube_paths:
        idx_to_path = {}
        for cube_path in cube_paths:
            fname = os.path.basename(cube_path)
            match = re.search(r"_(\d{4})(?:_|\.cube)", fname)
            if match:
                idx_to_path[int(match.group(1))] = cube_path

        render_cube_paths = [idx_to_path[i] for i in selected_indices if i in idx_to_path]
        render_labels = []
        render_occs = []
        for i in selected_indices:
            if i in idx_to_path:
                label = chemical_labels[i] if i < len(chemical_labels) else f"orb_{i}"
                render_labels.append(f"{i}: {label}")
                render_occs.append(float(occ[i]))

        if render_cube_paths:
            html_gallery_path = _batch_render_orbital_html(
                cube_paths=render_cube_paths,
                labels=render_labels,
                occupations=render_occs,
                output_dir=output_dir,
                isovalue=png_isovalue,
                gallery_name=f"{stem}_orbital_gallery.html" if stem else "orbital_gallery.html",
            )

    return {
        "report_path": report_path,
        "noon_path": noon_path,
        "cube_dir": cube_dir,
        "html_gallery_path": html_gallery_path,
        "chemical_labels": chemical_labels,
    }


def _detect_orbital_character(
    occ: float,
    block: str,
    proj_weight: float,
    occ_active_lo: float = 0.02,
    occ_active_hi: float = 1.98,
) -> str:
    if block == "active":
        return "strongly_correlated"
    if block == "core":
        if proj_weight > 0.3:
            return "bonding"
        if proj_weight > 0.05:
            return "weakly_bonding"
        return "inert"
    if proj_weight > 0.1:
        return "antibonding_virtual"
    return "virtual"


def _build_atom_labels(mol, cluster_info) -> _Optional[dict[int, str]]:
    if cluster_info is None:
        return None
    _require_authoritative_cluster_info(
        cluster_info,
        context="Orbital label construction",
    )
    label_map = {}
    for site_idx, metal in enumerate(cluster_info.metals):
        label_map[metal.index] = _resolve_metal_site_label(cluster_info, site_idx)
    for bridge in cluster_info.bridging_atoms:
        label_map[bridge.index] = _resolve_explicit_label(
            getattr(bridge, "label", ""),
            f"{bridge.element}{bridge.index + 1}",
            cluster_info=cluster_info,
            context=f"bridging atom {bridge.index}",
        )
    for ligand in cluster_info.terminal_ligands:
        if ligand.donor_atom_index >= 0:
            label_map[ligand.donor_atom_index] = _resolve_explicit_label(
                getattr(ligand, "label", ""),
                f"{mol.atom_symbol(ligand.donor_atom_index)}{ligand.donor_atom_index + 1}",
                cluster_info=cluster_info,
                context=f"terminal donor atom {ligand.donor_atom_index}",
            )
    return label_map if label_map else None


def _precompute_ao_contributions(
    mol,
    mo_coeff: np.ndarray,
    aoslices,
    ao_labels: list[str],
    atom_labels: _Optional[dict[int, str]] = None,
    top_n: int = 3,
) -> list[dict[str, float]]:
    nao, nmo = mo_coeff.shape
    natm = mol.natm
    ao_starts = np.array([aoslices[i][2] for i in range(natm)])

    overlap = mol.intor_symmetric("int1e_ovlp")
    overlap_coeff = np.dot(overlap, mo_coeff)
    mulliken = mo_coeff * overlap_coeff
    atom_contribs = np.add.reduceat(mulliken, ao_starts, axis=0)

    dominant_ao_global = np.empty((natm, nmo), dtype=int)
    for atom_idx in range(natm):
        ao_s, ao_e = int(aoslices[atom_idx][2]), int(aoslices[atom_idx][3])
        local_mull = mulliken[ao_s:ao_e, :]
        dominant_local = np.argmax(local_mull, axis=0)
        dominant_ao_global[atom_idx] = ao_s + dominant_local

    atom_names = []
    for atom_idx in range(natm):
        if atom_labels and atom_idx in atom_labels:
            atom_names.append(atom_labels[atom_idx])
        else:
            atom_names.append(f"{mol.atom_symbol(atom_idx)}{atom_idx + 1}")

    result: list[dict[str, float]] = []
    for mo_idx in range(nmo):
        contribs_i = atom_contribs[:, mo_idx]
        valid_indices = np.where(contribs_i > 1e-6)[0]
        if len(valid_indices) == 0:
            result.append({})
            continue
        sorted_idx = valid_indices[np.argsort(-contribs_i[valid_indices])][:top_n]
        row: dict[str, float] = {}
        for atom_idx in sorted_idx:
            dom_ao = dominant_ao_global[atom_idx, mo_idx]
            label = ao_labels[dom_ao] if dom_ao < len(ao_labels) else ""
            parts = label.split()
            ao_type = parts[-1] if len(parts) > 1 else ""
            key = f"{atom_names[atom_idx]}_{ao_type}" if ao_type else f"{atom_names[atom_idx]}_orb"
            row[key] = round(float(contribs_i[atom_idx]), 4)
        result.append(row)
    return result


def _best_chemical_label(ao_contrib: dict) -> str:
    if not ao_contrib:
        return ""
    return max(ao_contrib, key=ao_contrib.get)


def _sanitize_label(label: str) -> str:
    label = label.replace("^", "")
    label = label.replace("(", "_")
    label = label.replace(")", "")
    label = re.sub(r"_+", "_", label)
    return label.strip("_")


def _ensure_chemical_labels(mol, mo_coeff, labels, cluster_info) -> list[str]:
    """Recompute chemical labels from all-atom Mulliken analysis."""
    if cluster_info is not None:
        _require_authoritative_cluster_info(
            cluster_info,
            context="Chemical label assignment",
        )
    nmo = mo_coeff.shape[1]
    aoslices = mol.aoslice_by_atom()
    ao_labels_list = mol.ao_labels()
    atom_labels = _build_atom_labels(mol, cluster_info)
    all_ao_contribs = _precompute_ao_contributions(
        mol,
        mo_coeff,
        aoslices,
        ao_labels_list,
        atom_labels=atom_labels,
        top_n=3,
    )
    result = []
    for i in range(nmo):
        label = _best_chemical_label(all_ao_contribs[i])
        if not label and cluster_info is not None:
            raise ValueError(
                f"Failed to assign a chemical orbital label for MO {i} "
                "while using authoritative cluster_info.yaml."
            )
        result.append(label or f"orb_{i}")
    return result


def _build_atom_roles_markdown(cluster_info) -> list[str]:
    _require_authoritative_cluster_info(
        cluster_info,
        context="Atom role report generation",
    )
    lines = []
    lines.append("## Atom Roles")
    lines.append("")
    lines.append("| atom | element | index | role | bonded_to |")
    lines.append("|------|---------|-------|------|-----------|")

    bond_map: dict[str, set[str]] = {}

    def _ensure(label: str):
        bond_map.setdefault(label, set())

    for metal in cluster_info.metals:
        _ensure(metal.label)
    for bridge in cluster_info.bridging_atoms:
        bridge_label = _resolve_explicit_label(
            getattr(bridge, "label", ""),
            f"{bridge.element}{bridge.index + 1}",
            cluster_info=cluster_info,
            context=f"bridging atom {bridge.index}",
        )
        _ensure(bridge_label)
        for metal_idx in bridge.bridged_metals:
            if 0 <= metal_idx < len(cluster_info.metals):
                metal_label = _resolve_metal_site_label(cluster_info, metal_idx)
                bond_map[bridge_label].add(metal_label)
                bond_map[metal_label].add(bridge_label)

    for ligand in cluster_info.terminal_ligands:
        if ligand.donor_atom_index >= 0 and ligand.metal_index >= 0 and ligand.metal_index < len(cluster_info.metals):
            donor_label = _resolve_explicit_label(
                getattr(ligand, "label", ""),
                f"{cluster_info.all_elements[ligand.donor_atom_index]}{ligand.donor_atom_index + 1}",
                cluster_info=cluster_info,
                context=f"terminal donor atom {ligand.donor_atom_index}",
            )
            metal_label = _resolve_metal_site_label(cluster_info, ligand.metal_index)
            _ensure(donor_label)
            bond_map[donor_label].add(metal_label)
            bond_map[metal_label].add(donor_label)

    for site_idx, metal in enumerate(cluster_info.metals):
        metal_label = _resolve_metal_site_label(cluster_info, site_idx)
        bonded = ", ".join(sorted(bond_map.get(metal_label, set())))
        lines.append(f"| {metal_label} | {metal.element} | {metal.index} | metal | {bonded} |")

    for bridge in cluster_info.bridging_atoms:
        bridge_label = _resolve_explicit_label(
            getattr(bridge, "label", ""),
            f"{bridge.element}{bridge.index + 1}",
            cluster_info=cluster_info,
            context=f"bridging atom {bridge.index}",
        )
        bonded = ", ".join(sorted(bond_map.get(bridge_label, set())))
        lines.append(f"| {bridge_label} | {bridge.element} | {bridge.index} | bridging | {bonded} |")

    for ligand in cluster_info.terminal_ligands:
        if ligand.donor_atom_index >= 0 and ligand.metal_index >= 0 and ligand.metal_index < len(cluster_info.metals):
            donor_label = _resolve_explicit_label(
                getattr(ligand, "label", ""),
                f"{cluster_info.all_elements[ligand.donor_atom_index]}{ligand.donor_atom_index + 1}",
                cluster_info=cluster_info,
                context=f"terminal donor atom {ligand.donor_atom_index}",
            )
            bonded = ", ".join(sorted(bond_map.get(donor_label, set())))
            element = cluster_info.all_elements[ligand.donor_atom_index]
            lines.append(f"| {donor_label} | {element} | {ligand.donor_atom_index} | terminal | {bonded} |")

    lines.append("")
    return lines
