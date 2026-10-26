"""Checks that source text replacement is atomic and consistent with observed 4.1 anchors.
Does NOT claim full x265 compilation or true bitstream test.
"""
from pathlib import Path
import tempfile
import importlib.util

PATCH_FILE = Path(__file__).resolve().parents[1] / 'stego' / 'tar4' / 'TU' / 'x265_41_patch' / 'apply_patch.py'
SPEC = importlib.util.spec_from_file_location('tar4_tu_patch', PATCH_FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
main = MODULE.main

def test_fixture():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        src=base/'source'/'encoder'
        src.mkdir(parents=True)
        (src/'search.cpp').write_text('''#include "framedata.h"
    Cost fullCost;
    uint32_t bCBF = 0;

    pixel*   reconQt
        if (splitCost.rdcost < fullCost.rdcost)
        {
            outCost.rdcost     += splitCost.rdcost;
''')
        (src/'analysis.cpp').write_text('''#include "encoder.h"
    if (m_param->csvLogLevel >= 2)
        collectPUStatistics(ctu, cuGeom);

    return *m_modeDepth[0].bestMode;
}
''')
        main(base)
        text=(src/'search.cpp').read_text()
        assert 'xingChooseSplit' in text
        assert 'X265_TU_PLAN' not in text or 'XingTUBridge' in text
        assert 'tu_bridge.hpp' in text
        assert 'recordFinal' in (src/'analysis.cpp').read_text()
        main(base)
        assert text==(src/'search.cpp').read_text()
    print('PASS: source patch text replacements are idempotent')

if __name__=='__main__': test_fixture()
