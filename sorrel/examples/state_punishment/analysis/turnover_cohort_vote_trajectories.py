"""
Re-exports :mod:`cohort_turnover` for backward compatibility.

If ``from turnover_cohort_vote_trajectories import plot_turnover_encounter_cohorts``
fails in Jupyter, your kernel cached an older version of this file. Either restart
the kernel, or import from ``cohort_turnover`` instead (recommended).
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_path = Path(__file__).resolve().parent / "cohort_turnover.py"
_spec = spec_from_file_location("cohort_turnover", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load cohort_turnover from {_path}")

_cohort = module_from_spec(_spec)
_spec.loader.exec_module(_cohort)

for _name in _cohort.__all__:
    globals()[_name] = getattr(_cohort, _name)

__all__ = list(_cohort.__all__)
del _cohort, _name, _path, _spec
