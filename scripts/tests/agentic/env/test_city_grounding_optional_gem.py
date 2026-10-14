"""Standalone inference must work without GEM and expose broken GEM installs."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("missing_module", ["gem", "broken_gem_dependency"])
def test_standalone_gem_import(tmp_path, missing_module):
    probe = """
import importlib.abc
import runpy
import sys
from pathlib import Path

missing = sys.argv[1]

class BlockGem(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "gem":
            raise ModuleNotFoundError("missing " + missing, name=missing)

sys.meta_path.insert(0, BlockGem())
if missing == "gem":
    checks = runpy.run_path(sys.argv[2])
    checks["test_multi_turn_tool_then_commit"](Path(sys.argv[3]))
else:
    try:
        from roll.pipeline.agentic.env.city_grounding import CityGroundingEnv
    except ModuleNotFoundError as exc:
        assert exc.name == missing
    else:
        raise AssertionError("Broken GEM dependencies must not be hidden")
"""
    protocol_tests = Path(__file__).with_name("test_city_grounding.py")
    completed = subprocess.run(
        [sys.executable, "-c", probe, missing_module, str(protocol_tests), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[4], capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
