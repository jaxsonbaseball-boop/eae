"""
Verification tests for EAE Phase 2B-1 MiLB 4YR Draftee Pipeline & Blended Trainer.
"""

from __future__ import annotations

import ast
from pathlib import Path
import py_compile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMiLBPipelineSyntaxAndContracts(unittest.TestCase):
    def test_files_compile_and_parse(self) -> None:
        for rel in (
            "src/ingestion/fetch_milb_draft_arms.py",
            "src/models/train_blended_stuff_model.py",
        ):
            fpath = REPO_ROOT / rel
            self.assertTrue(fpath.exists(), f"Missing {rel}")
            py_compile.compile(str(fpath), doraise=True)
            ast.parse(fpath.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
