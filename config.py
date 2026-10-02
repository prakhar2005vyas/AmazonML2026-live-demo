import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
_default_data = ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
if not _default_data.exists():
    _default_data = ROOT / "student_resource" / "dataset"
DATA = Path(os.environ.get("ER_DATA", _default_data))
WORK = Path(os.environ.get("ER_WORK", Path.home() / "mlc26_work"))
OUTPUT = Path(os.environ.get("ER_OUTPUT", ROOT / "output"))
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
