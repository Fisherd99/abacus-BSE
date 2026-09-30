#!/usr/bin/env python3
"""Lightweight unit tests for the finite-field response tools."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_cases", HERE / "prepare_cases.py")
analyze = load_module("analyze_response", HERE / "analyze_response.py")


class FiniteFieldResponseTest(unittest.TestCase):
    def test_set_keyword_replaces_or_appends(self):
        text = "INPUT_PARAMETERS\ncal_force 0\n"
        self.assertIn("cal_force 1", prepare.set_keyword(text, "cal_force", 1))
        self.assertIn("finite_field_amp", prepare.set_keyword(text, "finite_field_amp", 0.0005))

    def test_infer_field_checks_centered_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for axis, direction in prepare.AXES.items():
                for sign, amplitude in (("minus", -0.0005), ("plus", 0.0005)):
                    case = root / f"{axis}_{sign}"
                    case.mkdir()
                    case.joinpath("INPUT").write_text(
                        f"INPUT_PARAMETERS\nfinite_field_dir {direction}\nfinite_field_amp {amplitude}\n"
                    )
            self.assertEqual(analyze.infer_field(root), 0.0005)

    def test_atom_map(self):
        self.assertEqual(analyze.apply_atom_map(["a", "b", "c"], [2, 0, 1]), ["b", "c", "a"])
        with self.assertRaises(RuntimeError):
            analyze.apply_atom_map(["a", "b"], [0, 0])


if __name__ == "__main__":
    unittest.main()
