from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.deep import run
from model import PUSN

if __name__=='__main__':
    run(PUSN,Path(__file__).with_name('config.json'))
