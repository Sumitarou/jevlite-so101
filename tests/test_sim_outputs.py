"""scripts/sim/outputs.py: one new output folder per run."""

from datetime import datetime

from scripts.sim.outputs import new_run_dir


def test_new_run_dir_is_stamped_and_never_reused(tmp_path):
    now = datetime(2026, 10, 11, 12, 34, 56)
    a = new_run_dir("engine_compare", base=tmp_path, now=now)
    b = new_run_dir("engine_compare", base=tmp_path, now=now)
    assert a.name == "20261011_123456_engine_compare" and a.is_dir()
    assert b.name == "20261011_123456_engine_compare_2" and b.is_dir()
