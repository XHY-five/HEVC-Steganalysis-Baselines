from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.runner import execute
from features import extract

if __name__=='__main__':
    execute(extract,Path(__file__).with_name('config.json'))
