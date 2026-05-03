#!/usr/bin/env python3
"""Generic artifact comparison CLI for validation-side workflows.

Examples:
    python scripts/compare_artifacts.py ref.FCIDUMP new.FCIDUMP
    python scripts/compare_artifacts.py ref_uhf.h5 new_uhf.h5 --format summary
    python scripts/compare_artifacts.py ref_dmrg_basis.npz new_dmrg_basis.npz --format json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from shared.comparison import compare_artifacts


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _emit_summary(prefix: str, payload: Any) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _emit_summary(child, value)
        return
    if isinstance(payload, list):
        if payload and all(not isinstance(v, (dict, list, tuple)) for v in payload):
            if len(payload) <= 8:
                print(f"{prefix}: {payload}")
            else:
                print(f"{prefix}: list(len={len(payload)})")
        else:
            print(f"{prefix}: list(len={len(payload)})")
        return
    print(f"{prefix}: {payload}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Reference artifact path")
    parser.add_argument("current", help="Current artifact path")
    parser.add_argument(
        "--format",
        choices=("summary", "json"),
        default="summary",
        help="Output format",
    )
    args = parser.parse_args()

    result = compare_artifacts(args.reference, args.current)
    payload = _jsonable(result)

    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"reference: {Path(args.reference).resolve()}")
    print(f"current:   {Path(args.current).resolve()}")
    print(f"kind:      {payload.get('kind', 'unknown')}")
    print()
    _emit_summary("", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
