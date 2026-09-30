#!/usr/bin/env python3
"""Prepare centered finite-field ABACUS cases for BEC/polarizability."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

AXES = {"x": 1, "y": 2, "z": 3}
REQUIRED_FILES = ("INPUT", "STRU", "KPT")


def set_keyword(text: str, keyword: str, value: object) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(keyword)}\s+).*$", re.MULTILINE)
    replacement = rf"\g<1>{value}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    if not text.endswith("\n"):
        text += "\n"
    return f"{text}{keyword:<27}{value}\n"


def parse_properties(raw: str) -> set[str]:
    properties = {item.strip().lower() for item in raw.split(",") if item.strip()}
    allowed = {"bec", "polarizability"}
    unknown = properties - allowed
    if not properties or unknown:
        raise argparse.ArgumentTypeError(
            "properties must be bec, polarizability, or bec,polarizability"
        )
    return properties


def write_case(
    template: Path, destination: Path, suffix: str, direction: int, amplitude: float,
    cal_force: bool, cal_stress: bool, overwrite: bool,
) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to replace existing case: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        shutil.copy2(template / name, destination / name)
    text = (destination / "INPUT").read_text()
    settings = {
        "suffix": suffix,
        "calculation": "scf",
        "cal_force": int(cal_force),
        "cal_stress": int(cal_stress),
        "finite_field": 1,
        "finite_field_dir": direction,
        "finite_field_amp": format(amplitude, ".16g"),
    }
    for keyword, value in settings.items():
        text = set_keyword(text, keyword, value)
    (destination / "INPUT").write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="directory containing INPUT, STRU, and KPT")
    parser.add_argument("output", type=Path, help="new seven-case directory")
    parser.add_argument("--field", type=float, required=True, help="positive finite_field_amp magnitude")
    parser.add_argument(
        "--properties", type=parse_properties, default=parse_properties("bec,polarizability"),
        help="comma-separated: bec,polarizability (default: both)",
    )
    parser.add_argument("--zero-force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-stress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suffix-prefix", default="finite-field-response")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.field <= 0.0:
        parser.error("--field must be positive")
    missing = [name for name in REQUIRED_FILES if not (args.template / name).is_file()]
    if missing:
        parser.error(f"template is missing: {', '.join(missing)}")

    nonzero_force = "bec" in args.properties
    cases = [("zero", 1, 0.0, args.zero_force, args.zero_stress)]
    for axis, direction in AXES.items():
        cases.extend(
            [
                (f"{axis}_minus", direction, -args.field, nonzero_force, False),
                (f"{axis}_plus", direction, args.field, nonzero_force, False),
            ]
        )
    for name, direction, amplitude, cal_force, cal_stress in cases:
        write_case(
            args.template, args.output / name, f"{args.suffix_prefix}-{name}",
            direction, amplitude, cal_force, cal_stress, args.overwrite,
        )
    print(f"prepared {len(cases)} cases below {args.output}")
    print(f"properties: {','.join(sorted(args.properties))}; field magnitude: {args.field:.16g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
