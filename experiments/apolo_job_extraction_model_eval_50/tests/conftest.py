import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
