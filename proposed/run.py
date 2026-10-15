from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.proposed import run
from model import DualBranchEnsemble

if __name__ == "__main__":
    run(DualBranchEnsemble, Path(__file__).with_name("config.json"))
