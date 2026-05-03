#!/usr/bin/env python3
"""Compare key Fe2S2 benchmark artifacts between two example directories.

Typical intended use after the next full rerun:

    python scripts/compare_fe2s2_runs.py \
      --current /Users/snh/Projects/APEX/examples/fe2s2 \
      --baseline /Users/snh/Projects/APEX_bk/examples/fe2s2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py

from shared.comparison import compare_density_matrices, compare_fcidumps


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _try_read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_records(path: Path) -> list[dict]:
    data = _read_json(path)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    return [data]


def _try_read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _read_records(path)


def _resolve_example_artifact(owner_root: Path, artifact_path: str | None) -> Path | None:
    if not artifact_path:
        return None
    raw = Path(artifact_path)
    if raw.exists():
        return raw
    text = str(artifact_path)
    for marker in ("/outputs/", "/filter_session/", "/chan_ref/"):
        if marker in text:
            rel = text.split(marker, 1)[1]
            return owner_root / marker.strip("/") / rel
    return None


def _resolve_testcas_dmrg_info(example_root: Path, stem: str) -> dict:
    dmrg_dir = example_root / "outputs" / "fcidump" / "dmrg"
    candidates = [
        dmrg_dir / f"{stem}_dmrg_info.json",
        dmrg_dir / f"{stem}_sz_M500_dmrg_info.json",
    ]
    for path in candidates:
        if path.exists():
            return _read_json(path)
    return {}


def _fmt_num(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12f}"
    return str(value)


def _fmt_delta(current: Any, baseline: Any) -> str:
    if isinstance(current, (int, float)) and isinstance(baseline, (int, float)):
        return f"{current - baseline:+.12f}"
    if current == baseline:
        return "match"
    return "DIFF"


def _cas_result(payload: dict, key: str) -> Any:
    if "results" in payload:
        return payload["results"].get(key)
    return payload.get(key)


def _get_one(records: list[dict], *, key: str, value: Any) -> dict | None:
    for row in records:
        if row.get(key) == value:
            return row
    return None


def _group_by_label(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in records:
        label = str(row.get("label", ""))
        grouped.setdefault(label, []).append(row)
    return grouped


def _first_common_labels(*record_sets: list[dict]) -> list[str]:
    label_sets = [{str(row.get("label", "")) for row in rows if row.get("label")} for rows in record_sets]
    if not label_sets:
        return []
    common = set.intersection(*label_sets)
    return sorted(common)


def _npz_to_h5(npz_path: str | None) -> Path | None:
    if not npz_path:
        return None
    path = Path(npz_path)
    if path.suffix != ".npz":
        return None
    return path.with_suffix(".h5")


def _load_h5_contract(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "groups": [],
            "theory": None,
            "control_source": None,
        }
    with h5py.File(path, "r") as h5:
        meta = h5["metadata"] if "metadata" in h5 else None
        settings = None
        if meta is not None and "settings_json" in meta.attrs:
            settings = json.loads(meta.attrs["settings_json"])
        return {
            "groups": sorted(h5.keys()),
            "theory": None if settings is None else settings.get("theory"),
            "control_source": None if settings is None else settings.get("control_source"),
        }


def _compare_dmrg_h5(current_path: Path | None, baseline_path: Path | None) -> dict[str, Any] | None:
    if current_path is None or baseline_path is None:
        return None
    if not current_path.exists() or not baseline_path.exists():
        return None

    with h5py.File(current_path, "r") as current_h5, h5py.File(baseline_path, "r") as baseline_h5:
        if "dmrg_1rdm" not in current_h5 or "dmrg_1rdm" not in baseline_h5:
            return None
        matrix_compare = compare_density_matrices(
            baseline_h5["dmrg_1rdm"][...],
            current_h5["dmrg_1rdm"][...],
        )
        noon_delta = None
        if "noon" in current_h5 and "noon" in baseline_h5:
            noon_current = current_h5["noon"][...]
            noon_baseline = baseline_h5["noon"][...]
            if noon_current.shape == noon_baseline.shape:
                noon_abs = abs(noon_current - noon_baseline)
                noon_delta = {
                    "max_abs": float(noon_abs.max()),
                    "frobenius": float((noon_abs**2).sum() ** 0.5),
                }
        return {
            "matrix": matrix_compare,
            "noon_delta": noon_delta,
        }


def _emit_section(title: str):
    print()
    print(f"## {title}")


def _emit_scalar_table(rows: list[tuple[str, Any, Any]]):
    print("| Metric | Current | Baseline | Delta |")
    print("|---|---:|---:|---:|")
    for metric, current, baseline in rows:
        print(
            f"| {metric} | {_fmt_num(current)} | {_fmt_num(baseline)} | {_fmt_delta(current, baseline)} |"
        )


def _emit_single_column_table(rows: list[tuple[str, Any]]):
    print("| Metric | Value |")
    print("|---|---:|")
    for metric, value in rows:
        print(f"| {metric} | {_fmt_num(value)} |")


def _compare_case(current: Path, baseline: Path) -> None:
    stem = "C4H12Fe2S6_uks_BP86_tzp-dkh"

    _emit_section("APEX_CAS JSON")
    current_scf = _try_read_json(current / "outputs" / "scf" / f"{stem}_scf_info.json")
    baseline_scf = _try_read_json(baseline / "outputs" / "scf" / f"{stem}_scf_info.json")
    current_cas = _try_read_json(current / "outputs" / "scf" / f"{stem}_cas_info.json")
    baseline_cas = _try_read_json(baseline / "outputs" / "scf" / f"{stem}_cas_info.json")
    current_fcid = _try_read_json(current / "outputs" / "fcidump" / f"{stem}_fcidump_info.json")
    baseline_fcid = _try_read_json(baseline / "outputs" / "fcidump" / f"{stem}_fcidump_info.json")
    current_dmrg_info = _resolve_testcas_dmrg_info(current, stem)
    baseline_dmrg_info = _resolve_testcas_dmrg_info(baseline, stem)
    _emit_scalar_table(
        [
            ("SCF energy", current_scf.get("energy"), baseline_scf.get("energy")),
            ("CAS E_core", _cas_result(current_cas, "E_core"), _cas_result(baseline_cas, "E_core")),
            ("CAS E_act", _cas_result(current_cas, "E_act"), _cas_result(baseline_cas, "E_act")),
            ("CAS E_tot", _cas_result(current_cas, "E_tot"), _cas_result(baseline_cas, "E_tot")),
            ("FCIDUMP n_electrons", current_fcid.get("n_electrons"), baseline_fcid.get("n_electrons")),
            ("FCIDUMP n_orbitals", current_fcid.get("n_orbitals"), baseline_fcid.get("n_orbitals")),
            ("testcas E_total", current_dmrg_info.get("e_total"), baseline_dmrg_info.get("e_total")),
        ]
    )

    _emit_section("APEX_CAS DMRG Matrix Compare")
    current_dmrg_h5 = None
    baseline_dmrg_h5 = None
    if current_dmrg_info:
        current_dmrg_h5 = _resolve_example_artifact(current, current_dmrg_info.get("results_h5"))
    if baseline_dmrg_info:
        baseline_dmrg_h5 = _resolve_example_artifact(baseline, baseline_dmrg_info.get("results_h5"))
    dmrg_matrix_compare = _compare_dmrg_h5(current_dmrg_h5, baseline_dmrg_h5)
    if dmrg_matrix_compare is None:
        print("| Metric | Value |")
        print("|---|---|")
        print("| dmrg_matrix_compare | skipped (missing DMRG HDF5) |")
    else:
        matrix = dmrg_matrix_compare["matrix"]
        noon_delta = dmrg_matrix_compare["noon_delta"] or {}
        _emit_single_column_table(
            [
                ("dmrg_1rdm_elementwise_frobenius", matrix["elementwise"]["frobenius"]),
                ("dmrg_1rdm_elementwise_rms", matrix["elementwise"]["rms"]),
                ("dmrg_1rdm_elementwise_max_abs", matrix["elementwise"]["max_abs"]),
                ("dmrg_1rdm_trace_diff", matrix["trace_diff"]),
                ("dmrg_1rdm_trace_square_diff", matrix["trace_square_diff"]),
                ("dmrg_1rdm_eigenvalue_frobenius", matrix["spectrum"]["eigenvalue_frobenius"]),
                ("dmrg_1rdm_eigenvalue_max_abs", matrix["spectrum"]["eigenvalue_max_abs"]),
                ("dmrg_1rdm_basis_rotation_likely", matrix["basis_rotation_likely"]),
                ("noon_frobenius", noon_delta.get("frobenius")),
                ("noon_max_abs", noon_delta.get("max_abs")),
            ]
        )

    _emit_section("APEX_CAS FCIDUMP Deep Compare")
    current_fcidump = current / "outputs" / "fcidump" / f"FCIDUMP.{stem}"
    baseline_fcidump = baseline / "outputs" / "fcidump" / f"FCIDUMP.{stem}"
    if current_fcidump.exists() and baseline_fcidump.exists():
        fcidump_compare = compare_fcidumps(
            str(baseline_fcidump),
            str(current_fcidump),
        )
        _emit_single_column_table(
            [
                ("match", fcidump_compare.get("match")),
                ("eigval_frobenius", fcidump_compare.get("eigval_frobenius")),
                ("eigval_max", fcidump_compare.get("eigval_max")),
                ("h1e_frobenius", fcidump_compare.get("h1e_frobenius")),
                ("h2e_rms", fcidump_compare.get("h2e_rms")),
                ("h2e_max", fcidump_compare.get("h2e_max")),
                ("ecore_diff", fcidump_compare.get("ecore_diff")),
                ("n_eigval_mismatch", fcidump_compare.get("n_eigval_mismatch")),
            ]
        )
    else:
        print("| Metric | Value |")
        print("|---|---|")
        print("| fcidump_compare | skipped (missing FCIDUMP file) |")

    _emit_section("APEX_CAS HDF5 Contract")
    current_cas_h5 = _load_h5_contract(current / "outputs" / "orbitals" / f"{stem}_cas_data.h5")
    baseline_cas_h5 = _load_h5_contract(baseline / "outputs" / "orbitals" / f"{stem}_cas_data.h5")
    print("| Artifact | Current groups | Baseline groups |")
    print("|---|---|---|")
    print(
        f"| cas_data.h5 | `{', '.join(current_cas_h5['groups'])}` | `{', '.join(baseline_cas_h5['groups'])}` |"
    )

    _emit_section("APEX_Filter Summaries")
    cur_step3_records = _try_read_records(current / "filter_session" / "step3_uhf" / "uhf_summary.json")
    bk_step3_records = _try_read_records(baseline / "filter_session" / "step3_uhf" / "uhf_summary.json")
    cur_step6_records = _try_read_records(current / "filter_session" / "step6_ccsdt" / "ccsdt_summary.json")
    bk_step6_records = _try_read_records(baseline / "filter_session" / "step6_ccsdt" / "ccsdt_summary.json")
    cur_step9_records = _try_read_records(current / "filter_session" / "step9_extrapolate" / "dmrg_extrapolation_summary.json")
    bk_step9_records = _try_read_records(baseline / "filter_session" / "step9_extrapolate" / "dmrg_extrapolation_summary.json")
    cur_step10_records = _try_read_records(current / "filter_session" / "step10_report" / "final_summary.json")
    bk_step10_records = _try_read_records(baseline / "filter_session" / "step10_report" / "final_summary.json")

    summary_rows: list[tuple[str, Any, Any]] = []
    for label in _first_common_labels(cur_step3_records, bk_step3_records):
        cur_step3 = _get_one(cur_step3_records, key="label", value=label)
        bk_step3 = _get_one(bk_step3_records, key="label", value=label)
        summary_rows.extend(
            [
                (f"step3 UHF energy [{label}]", cur_step3.get("energy"), bk_step3.get("energy")),
                (f"step3 UHF <S^2> [{label}]", cur_step3.get("s_squared"), bk_step3.get("s_squared")),
                (f"step3 two_sz_fe1 [{label}]", cur_step3.get("two_sz_fe1"), bk_step3.get("two_sz_fe1")),
                (f"step3 two_sz_fe2 [{label}]", cur_step3.get("two_sz_fe2"), bk_step3.get("two_sz_fe2")),
            ]
        )
    for label in _first_common_labels(cur_step6_records, bk_step6_records):
        cur_step6 = _get_one(cur_step6_records, key="label", value=label)
        bk_step6 = _get_one(bk_step6_records, key="label", value=label)
        summary_rows.extend(
            [
                (f"step6 CCSDT energy [{label}]", cur_step6.get("energy"), bk_step6.get("energy")),
                (f"step6 CCSDT <S^2> [{label}]", cur_step6.get("s_squared"), bk_step6.get("s_squared")),
            ]
        )
    for label in _first_common_labels(cur_step9_records, bk_step9_records):
        cur_step9 = _get_one(cur_step9_records, key="label", value=label)
        bk_step9 = _get_one(bk_step9_records, key="label", value=label)
        summary_rows.append(
            (f"step9 extrapolated energy [{label}]", cur_step9.get("energy"), bk_step9.get("energy"))
        )
    for label in _first_common_labels(cur_step10_records, bk_step10_records):
        cur_step10 = _get_one(cur_step10_records, key="label", value=label)
        bk_step10 = _get_one(bk_step10_records, key="label", value=label)
        summary_rows.append(
            (
                f"step10 consensus energy [{label}]",
                cur_step10.get("consensus_energy"),
                bk_step10.get("consensus_energy"),
            )
        )
    if summary_rows:
        _emit_scalar_table(summary_rows)
    else:
        print("| Metric | Value |")
        print("|---|---|")
        print("| filter_summary_compare | skipped (current filter_session not present yet) |")

    _emit_section("Step8 DMRG Ladder")
    cur_step8 = _try_read_records(current / "filter_session" / "step8_dmrg" / "dmrg_summary.json")
    bk_step8 = _try_read_records(baseline / "filter_session" / "step8_dmrg" / "dmrg_summary.json")
    cur_step8_by_label = _group_by_label(cur_step8)
    bk_step8_by_label = _group_by_label(bk_step8)
    print("| Label | Bond dim | Current energy | Baseline energy | Delta |")
    print("|---|---:|---:|---:|---:|")
    common_step8_labels = sorted(set(cur_step8_by_label) & set(bk_step8_by_label))
    if common_step8_labels:
        for label in common_step8_labels:
            cur_rows = cur_step8_by_label[label]
            bk_rows = bk_step8_by_label[label]
            for bond_dim in sorted({int(row["bond_dim"]) for row in cur_rows} | {int(row["bond_dim"]) for row in bk_rows}):
                cur = _get_one(cur_rows, key="bond_dim", value=bond_dim)
                bk = _get_one(bk_rows, key="bond_dim", value=bond_dim)
                cur_e = None if cur is None else cur.get("energy")
                bk_e = None if bk is None else bk.get("energy")
                print(f"| {label} | {bond_dim} | {_fmt_num(cur_e)} | {_fmt_num(bk_e)} | {_fmt_delta(cur_e, bk_e)} |")
    else:
        print("| (no common Step8 records yet) | N/A | N/A | N/A | N/A |")

    _emit_section("APEX_Filter HDF5 Provenance")
    step8_cur_ref = None
    step8_bk_ref = None
    if common_step8_labels:
        compare_label = common_step8_labels[0]
        step8_cur_ref = _get_one(cur_step8_by_label[compare_label], key="bond_dim", value=1000) or _get_one(
            cur_step8_by_label[compare_label], key="bond_dim", value=400
        )
        step8_bk_ref = _get_one(bk_step8_by_label[compare_label], key="bond_dim", value=1000) or _get_one(
            bk_step8_by_label[compare_label], key="bond_dim", value=400
        )
    stage_paths = [
        ("step3", current / "filter_session" / "step3_uhf" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_uhf.h5",
         baseline / "filter_session" / "step3_uhf" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_uhf.h5"),
        ("step6", current / "filter_session" / "step6_ccsdt" / "scripts" / "Fe1↑Fe2↓_2xFe(III)_d:none_ccsdt_results.h5",
         baseline / "filter_session" / "step6_ccsdt" / "scripts" / "Fe1↑Fe2↓_2xFe(III)_d:none_ccsdt_results.h5"),
        ("step7", current / "filter_session" / "step7_dmrg_basis" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_dmrg_basis.h5",
         baseline / "filter_session" / "step7_dmrg_basis" / "results" / "Fe1↑Fe2↓_2xFe(III)_d:none_dmrg_basis.h5"),
        (
            "step8",
            None if step8_cur_ref is None else _npz_to_h5(step8_cur_ref.get("result_path")),
            None if step8_bk_ref is None else _npz_to_h5(step8_bk_ref.get("result_path")),
        ),
    ]
    print("| Stage | Current theory | Baseline theory | Current control source | Baseline control source |")
    print("|---|---|---|---|---|")
    for stage, cur_path, bk_path in stage_paths:
        cur = _load_h5_contract(cur_path) if cur_path is not None else {"theory": None, "control_source": None}
        bk = _load_h5_contract(bk_path) if bk_path is not None else {"theory": None, "control_source": None}
        print(
            f"| {stage} | `{cur['theory']}` | `{bk['theory']}` | `{cur['control_source']}` | `{bk['control_source']}` |"
        )

    _emit_section("Chan Bundle Presence")
    required = [
        "fe2s2_chan2026_oxidized_benchmark.json",
        "fe2s2_oxidized_apex_vs_chan2026_energy_table.csv",
        "fe2s2_oxidized_apex_vs_chan2026_observables_table.csv",
        "fe2s2_oxidized_apex_vs_chan2026_tables.md",
    ]
    print("| File | Current | Baseline |")
    print("|---|---|---|")
    for name in required:
        print(
            f"| {name} | {'yes' if (current / 'chan_ref' / name).exists() else 'no'} | "
            f"{'yes' if (baseline / 'chan_ref' / name).exists() else 'no'} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, help="Current examples/fe2s2 directory")
    parser.add_argument("--baseline", required=True, help="Baseline examples/fe2s2 directory")
    args = parser.parse_args()

    _compare_case(Path(args.current).resolve(), Path(args.baseline).resolve())


if __name__ == "__main__":
    main()
