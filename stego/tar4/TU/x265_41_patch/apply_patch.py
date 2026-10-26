#!/usr/bin/env python3
"""Apply Xing TU-only intra (8x8 CU, 2Nx2N PU) RDO hook to official x265 4.1.

This is a source patch, NOT a compiled encoder. Fails on unfamiliar x265 source.
"""
from pathlib import Path
import shutil
import sys

here = Path(__file__).resolve().parent

def replace_one(text: str, old: str, new: str, tag: str) -> str:
    if new in text:
        print(f'  already patched: {tag}')
        return text
    n=text.count(old)
    if n != 1:
        raise RuntimeError(f'{tag}: expected exactly 1 anchor, found {n}; check x265 version')
    print('  patch: '+tag)
    return text.replace(old,new,1)

def main(source: Path) -> None:
    root=source.resolve()
    if not (root/'source'/'encoder'/'search.cpp').exists():
        if (root/'encoder'/'search.cpp').exists():
            root=root.parent
        else:
            raise FileNotFoundError(f'Expected x265 root with source/encoder/search.cpp: {root}')
    search=root/'source'/'encoder'/'search.cpp'
    analysis=root/'source'/'encoder'/'analysis.cpp'
    s=search.read_text(encoding='utf-8')
    a=analysis.read_text(encoding='utf-8')
    s=replace_one(s, '#include "framedata.h"\n', '#include "framedata.h"\n#include "tu_bridge.hpp"\n', 'include search')
    a=replace_one(a, '#include "encoder.h"\n', '#include "encoder.h"\n#include "tu_bridge.hpp"\n', 'include analysis')
    # Insert before the 8x8 intra full/split evaluation, never directly patch m_tuDepth.
    s=replace_one(s,
'''    Cost fullCost;
    uint32_t bCBF = 0;

    pixel*   reconQt''',
'''    // Xing TU-only: 8x8 intra CU / SIZE_2Nx2N with an optional forced split.
    // Leave ordinary x265 decisions untouched when no matching plan key exists.
    const bool xingTU = cuGeom.log2CUSize == 3 && log2TrSize == 3 &&
                        tuDepth == 0 && cu.m_partSize[0] == SIZE_2Nx2N && cu.isIntra(0);
    int xingTarget = -1;
    if (xingTU && XingTUBridge::instance().active())
    {
        xingTarget = XingTUBridge::instance().target(m_frame->m_poc,
                         cu.m_cuAddr, (int)cu.m_cuPelX, (int)cu.m_cuPelY);
        if (xingTarget == 1 && mightNotSplit && log2TrSize > depthRange[0])
            mightSplit = true; // evaluate 8x8 -> four 4x4 TUs even in coarse intra trial
    }

    Cost fullCost;
    uint32_t bCBF = 0;

    pixel*   reconQt''', 'TU eligibility / force split')
    s=replace_one(s,
'''        if (splitCost.rdcost < fullCost.rdcost)
        {
            outCost.rdcost     += splitCost.rdcost;''',
'''        bool xingChooseSplit = splitCost.rdcost < fullCost.rdcost;
        if (xingTU && mightNotSplit && XingTUBridge::instance().active())
        {
            XingTUBridge::instance().observe(m_frame->m_poc, cu.m_cuAddr,
                  (int)cu.m_cuPelX, (int)cu.m_cuPelY,
                  (int)cu.m_lumaIntraDir[0], (double)fullCost.rdcost,
                  (double)splitCost.rdcost);
            if (xingTarget >= 0)
                xingChooseSplit = (xingTarget == 1);
        }
        if (xingChooseSplit)
        {
            outCost.rdcost     += splitCost.rdcost;''', 'select TU split with full RDO state')
    a=replace_one(a,
'''    if (m_param->csvLogLevel >= 2)
        collectPUStatistics(ctu, cuGeom);

    return *m_modeDepth[0].bestMode;
}''',
'''    if (m_param->csvLogLevel >= 2)
        collectPUStatistics(ctu, cuGeom);

    // Trace only the FINAL chosen CTU, not speculative 8x8 CU RDO candidates.
    // The trace is emitted before the normal x265 entropy writer codes this CTU.
    if (XingTUBridge::instance().active())
    {
        const CUData& finalCU = m_modeDepth[0].bestMode->cu;
        int nFourByFourPUs = 0;
        for (uint32_t zi=0; zi<finalCU.m_numPartitions; )
        {
            uint32_t d=finalCU.m_cuDepth[zi];
            uint32_t step=finalCU.m_numPartitions >> (2*d);
            if (step==0) step=1;
            if (finalCU.m_log2CUSize[zi]==3 && finalCU.m_partSize[zi]==SIZE_NxN && finalCU.isIntra(zi))
                nFourByFourPUs += 4;
            zi += step;
        }
        for (uint32_t zi=0; zi<finalCU.m_numPartitions; )
        {
            uint32_t d=finalCU.m_cuDepth[zi];
            uint32_t step=finalCU.m_numPartitions >> (2*d);
            if (step==0) step=1;
            if (finalCU.m_log2CUSize[zi]==3 &&
                finalCU.m_partSize[zi]==SIZE_2Nx2N && finalCU.isIntra(zi))
            {
                int bx=(int)finalCU.m_cuPelX + (int)g_zscanToPelX[zi];
                int by=(int)finalCU.m_cuPelY + (int)g_zscanToPelY[zi];
                XingTUBridge::instance().recordFinal(m_frame->m_poc,
                    finalCU.m_cuAddr, bx, by, (finalCU.m_tuDepth[zi]>0 ? 1 : 0), nFourByFourPUs);
            }
            zi += step;
        }
    }

    return *m_modeDepth[0].bestMode;
}''', 'final CTU TU trace')
    # Don't write partial patch if an anchor fails.
    shutil.copyfile(here/'tu_bridge.hpp', root/'source'/'encoder'/'tu_bridge.hpp')
    search.write_text(s,encoding='utf-8',newline='\n')
    analysis.write_text(a,encoding='utf-8',newline='\n')
    print('Patched x265 source at:',root)

if __name__=='__main__':
    if len(sys.argv)!=2:
        sys.exit('Usage: python apply_patch.py <path-to-extracted-x265-4.1>')
    main(Path(sys.argv[1]))
