#!/usr/bin/env python3
"""Analyze seven centered finite-field ABACUS calculations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

AXES = "xyz"
POINTS = ("zero", "x_minus", "x_plus", "y_minus", "y_plus", "z_minus", "z_plus")
RY_TO_EV = 13.605693122994
BOHR_PER_ANGSTROM = 1.889725988579
ELECTRON_CHARGE_RY = math.sqrt(2.0)
FORCE_CONVERSION = RY_TO_EV * BOHR_PER_ANGSTROM * ELECTRON_CHARGE_RY
POLARIZABILITY_CONVERSION = 4.0 * math.pi * ELECTRON_CHARGE_RY
FLOAT = r"[-+0-9.eE]+"
FORCE_HEADER = re.compile(r"^\s*#?TOTAL-FORCE \(eV/Angstrom\)#?\s*$", re.MULTILINE)
FORCE_ROW = re.compile(rf"^\s*(\S+)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
POLARIZATION = re.compile(
    rf"Finite-field(?: LCAO)? polarization_e_per_bohr2 electronic=.*? total="
    rf"({FLOAT})\s+({FLOAT})\s+({FLOAT})"
)
WALL_TIME = re.compile(rf"^\s*total\s+({FLOAT})\s+", re.MULTILINE)
INPUT_VALUE = re.compile(r"^\s*({key})\s+([^#\s]+)", re.MULTILINE)


def input_value(path: Path, key: str) -> str:
    match = re.search(INPUT_VALUE.pattern.format(key=re.escape(key)), path.read_text(), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{key} is missing from {path}")
    return match.group(2)


def find_log(case: Path) -> Path:
    logs = sorted(case.glob("OUT.*/running_scf.log"))
    if len(logs) != 1:
        raise RuntimeError(f"expected exactly one OUT.*/running_scf.log below {case}, got {len(logs)}")
    return logs[0]


def parse_case(case: Path) -> dict:
    log = find_log(case)
    text = log.read_text(errors="replace")
    converged = "#SCF IS CONVERGED#" in text and "!!SCF IS NOT CONVERGED!!" not in text
    polarizations = POLARIZATION.findall(text)
    forces: list[list[float]] = []
    labels: list[str] = []
    headers = list(FORCE_HEADER.finditer(text))
    if headers:
        for line in text[headers[-1].end():].splitlines():
            match = FORCE_ROW.match(line)
            if match:
                labels.append(match.group(1))
                forces.append([float(value) for value in match.groups()[1:]])
            elif forces:
                break
    wall_times = WALL_TIME.findall(text)
    return {
        "log": str(log),
        "converged": converged,
        "polarization_e_per_bohr2": (
            [float(value) for value in polarizations[-1]] if polarizations else None
        ),
        "force_labels": labels,
        "forces_eV_per_A": forces,
        "wall_seconds": float(wall_times[-1]) if wall_times else None,
    }


def infer_field(root: Path) -> float:
    amplitudes = {}
    for axis in AXES:
        amplitudes[f"{axis}_minus"] = float(input_value(root / f"{axis}_minus" / "INPUT", "finite_field_amp"))
        amplitudes[f"{axis}_plus"] = float(input_value(root / f"{axis}_plus" / "INPUT", "finite_field_amp"))
    magnitudes = [abs(value) for value in amplitudes.values()]
    field = magnitudes[0]
    tolerance = max(1.0e-15, field * 1.0e-12)
    if field == 0.0 or any(abs(value - field) > tolerance for value in magnitudes):
        raise RuntimeError(f"nonzero field magnitudes are inconsistent: {amplitudes}")
    for axis in AXES:
        if amplitudes[f"{axis}_minus"] >= 0.0 or amplitudes[f"{axis}_plus"] <= 0.0:
            raise RuntimeError(f"invalid +/- field signs: {amplitudes}")
    return field


def apply_atom_map(values: list, mapping: list[int]) -> list:
    if sorted(mapping) != list(range(len(values))):
        raise RuntimeError("atom map must be a permutation of 0..N-1")
    mapped = [None] * len(values)
    for grouped_index, target_index in enumerate(mapping):
        mapped[target_index] = values[grouped_index]
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing zero and six +/- field cases")
    parser.add_argument("--field", type=float, help="assert the positive field magnitude")
    parser.add_argument("--atom-map", type=Path, help="JSON grouped-index to target-index permutation")
    parser.add_argument(
        "--polarizability-only", action="store_true",
        help="allow nonzero-field outputs without forces",
    )
    parser.add_argument("--output", type=Path, default=Path("response.json"))
    args = parser.parse_args()

    missing = [point for point in POINTS if not (args.root / point / "INPUT").is_file()]
    if missing:
        parser.error(f"missing case directories or INPUT files: {', '.join(missing)}")
    inferred_field = infer_field(args.root)
    field = args.field if args.field is not None else inferred_field
    if field <= 0.0 or not math.isclose(field, inferred_field, rel_tol=1.0e-12, abs_tol=1.0e-15):
        parser.error(f"--field {field} disagrees with INPUT magnitude {inferred_field}")

    cases = {point: parse_case(args.root / point) for point in POINTS}
    failed = [point for point, case in cases.items() if not case["converged"]]
    if failed:
        raise RuntimeError(f"SCF did not converge: {', '.join(failed)}")
    nonzero_points = [point for point in POINTS if point != "zero"]
    missing_polarization = [
        point for point in nonzero_points
        if cases[point]["polarization_e_per_bohr2"] is None
    ]
    if missing_polarization:
        raise RuntimeError(f"polarization is missing: {', '.join(missing_polarization)}")

    alpha: list[list[float]] = []
    delta_p: list[list[float]] = []
    for axis in AXES:
        plus = cases[f"{axis}_plus"]["polarization_e_per_bohr2"]
        minus = cases[f"{axis}_minus"]["polarization_e_per_bohr2"]
        delta = [plus[i] - minus[i] for i in range(3)]
        delta_p.append(delta)
        alpha.append([POLARIZABILITY_CONVERSION * value / (2.0 * field) for value in delta])

    force_counts = {point: len(case["forces_eV_per_A"]) for point, case in cases.items()}
    have_bec = all(force_counts[point] > 0 for point in nonzero_points)
    if not have_bec and not args.polarizability_only:
        raise RuntimeError(
            "BEC requires force output in all six nonzero-field cases; "
            "rerun them with cal_force 1 or pass --polarizability-only"
        )
    bec_grouped = None
    acoustic = None
    labels = None
    if have_bec:
        counts = {force_counts[point] for point in nonzero_points}
        if len(counts) != 1:
            raise RuntimeError(f"inconsistent nonzero-field force row counts: {force_counts}")
        natom = counts.pop()
        labels = cases["x_plus"]["force_labels"]
        if any(cases[point]["force_labels"] != labels for point in nonzero_points):
            raise RuntimeError("atom labels/order differ among nonzero-field force tables")
        bec_grouped = [[[0.0] * 3 for _ in range(3)] for _ in range(natom)]
        for beta, axis in enumerate(AXES):
            plus = cases[f"{axis}_plus"]["forces_eV_per_A"]
            minus = cases[f"{axis}_minus"]["forces_eV_per_A"]
            for atom in range(natom):
                for response in range(3):
                    bec_grouped[atom][beta][response] = (
                        plus[atom][response] - minus[atom][response]
                    ) / (2.0 * field * FORCE_CONVERSION)
        acoustic = [
            [sum(tensor[beta][response] for tensor in bec_grouped) for response in range(3)]
            for beta in range(3)
        ]

    warnings = []
    if max(abs(value) for row in delta_p for value in row) > 0.05:
        warnings.append("large +/- polarization jump; inspect Berry-phase branch continuity")
    antisymmetric = max(abs(alpha[i][j] - alpha[j][i]) for i in range(3) for j in range(3))
    bec_mapped = None
    if args.atom_map is not None:
        if bec_grouped is None:
            raise RuntimeError("--atom-map requires force output and a BEC calculation")
        mapping = json.loads(args.atom_map.read_text())
        bec_mapped = apply_atom_map(bec_grouped, mapping)

    report = {
        "conventions": {
            "tensor_rows": "applied field beta (x,y,z)",
            "tensor_columns": "polarization or force response alpha (x,y,z)",
            "field_unit": "ABACUS finite_field_amp in Ry/(e*bohr) convention",
            "polarizability_unit": "dimensionless",
            "bec_unit": "elementary charge",
        },
        "field_magnitude": field,
        "all_scf_converged": True,
        "polarization_difference_e_per_bohr2": delta_p,
        "polarizability_tensor": alpha,
        "max_abs_polarizability_antisymmetric": antisymmetric,
        "bec_available": have_bec,
        "force_labels_grouped_order": labels,
        "born_effective_charges_grouped_order": bec_grouped,
        "born_effective_charges_mapped_order": bec_mapped,
        "acoustic_sum_tensor": acoustic,
        "max_abs_acoustic_sum_e": (
            max(abs(value) for row in acoustic for value in row) if acoustic is not None else None
        ),
        "total_wall_seconds": (
            sum(case["wall_seconds"] for case in cases.values())
            if all(case["wall_seconds"] is not None for case in cases.values()) else None
        ),
        "warnings": warnings,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    alpha_csv = args.output.with_suffix(".polarizability.csv")
    with alpha_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["field_direction", "alpha_x", "alpha_y", "alpha_z"])
        writer.writerows([[AXES[i], *alpha[i]] for i in range(3)])
    if bec_grouped is not None:
        bec_csv = args.output.with_suffix(".bec.csv")
        output_bec = bec_mapped if bec_mapped is not None else bec_grouped
        with bec_csv.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["atom_index_0based", "Z_xx", "Z_xy", "Z_xz", "Z_yx", "Z_yy", "Z_yz", "Z_zx", "Z_zy", "Z_zz"])
            for atom, tensor in enumerate(output_bec):
                writer.writerow([atom, *(tensor[beta][response] for beta in range(3) for response in range(3))])
    print(json.dumps({key: report[key] for key in ("field_magnitude", "polarizability_tensor", "max_abs_polarizability_antisymmetric", "bec_available", "max_abs_acoustic_sum_e", "total_wall_seconds", "warnings")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
