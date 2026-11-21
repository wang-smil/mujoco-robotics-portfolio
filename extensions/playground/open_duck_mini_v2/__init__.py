"""Open Duck Mini environment and its supported task variants."""
from pathlib import Path as _Path

_package_dir = _Path(__file__).resolve().parent
# Keep existing module names usable by saved runs and external evaluation scripts.
__path__ += [str(_package_dir / "profiles"),
             str(_package_dir.parents[2] / "history" / "environment_profiles")]
