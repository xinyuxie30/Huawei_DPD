import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "step40.1.15_bofir_eatcn1000.py"
DEFAULT_VARIANT = "p3m5_h11_fir47_1184"


def load_base_module():
    spec = importlib.util.spec_from_file_location("bofir_eatcn", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    if "--variant" not in sys.argv:
        sys.argv.extend(["--variant", DEFAULT_VARIANT])
    load_base_module().main()
