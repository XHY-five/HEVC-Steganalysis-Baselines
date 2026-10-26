import csv
import random
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stego' / 'tar4'))

from TU.tu_core import stc_embed, stc_extract, read_candidates, groups, costs
from TU.tu_embed import embed
from TU.tu_extract import extract
from TU.tu_verify import verify
import itertools


class TestStc(unittest.TestCase):
    def test_randomized_roundtrips(self):
        rng = random.Random(2024)
        for n, m in ((3, 1), (9, 2), (20, 6), (32, 15), (64, 22), (200, 60)):
            for height in (3, 7):
                orig = [rng.randrange(2) for _ in range(n)]
                msg = [rng.randrange(2) for _ in range(m)]
                rho = [rng.random()*100 for _ in range(n)]
                stego = stc_embed(orig, msg, rho, height=height, seed=42)
                self.assertEqual(stc_extract(stego, m, height=height, seed=42), msg)
                self.assertEqual(len(stego), n)

    def test_stc_minimum_weight_small_problem(self):
        rng = random.Random(404)
        for n, m in ((5, 2), (8, 3)):
            orig = [rng.randrange(2) for _ in range(n)]
            msg = [rng.randrange(2) for _ in range(m)]
            rho = [rng.randint(1, 30) for _ in range(n)]
            chosen = stc_embed(orig, msg, rho, height=3, seed=7)
            actual = sum(r for x, y, r in zip(orig, chosen, rho) if x != y)
            optimal = min(sum(r for x, y, r in zip(orig, z, rho) if x != y)
                          for z in itertools.product((0, 1), repeat=n)
                          if stc_extract(z, m, height=3, seed=7) == msg)
            self.assertEqual(actual, optimal)

    def test_tu_complete_mock_codec(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            source = root/'source.csv'
            keys = ['frame','ctu','x','y','cu_size','pu_mode',
                    'pred_mode','tu_bit','j_single','j_split','n4x4']
            rng = random.Random(11)
            with source.open('w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f,fieldnames=keys); w.writeheader()
                for frame in range(2):
                    for ctu, n, n4 in ((0, 20, 24), (1, 14, 12), (2, 10, 5)):
                        for i in range(n):
                            w.writerow(dict(frame=frame,ctu=ctu,x=(i%8)*8,y=(i//8)*8,
                                            cu_size='8x8',pu_mode='2Nx2N',pred_mode='intra',
                                            tu_bit=rng.randrange(2),j_single=100+rng.random()*30,
                                            j_split=100+rng.random()*30,n4x4=n4))
            info = embed(source, root/'embed', ratio=.3,seed=42,
                         tb=8,tstc=16,height=7,cost_mode='paper')
            self.assertGreater(info['message_length'],0)
            self.assertEqual(len(info['groups']),4)  # 2 frames × 2 eligible CTUs
            # Mock only: emulate a correctly patched codec by applying the plan.
            plan = root/'embed'/'tu_plan.csv'
            decoded = root/'decoded.csv'
            with plan.open(newline='',encoding='utf-8') as src, \
                    decoded.open('w',newline='',encoding='utf-8') as dest:
                r=csv.DictReader(src)
                outcols=list(r.fieldnames)
                w=csv.DictWriter(dest,fieldnames=outcols);w.writeheader()
                for row in r:
                    row['tu_bit']=row['target_bit']
                    w.writerow(row)
            self.assertTrue(verify(plan, decoded, tb=8)['verified'])
            final=extract(decoded,root/'extracted.txt',ratio=.3,seed=42,
                          tb=8,tstc=16,height=7,
                          manifest=root/'embed'/'manifest.json')
            self.assertTrue(final['matched_manifest'])
            self.assertEqual((root/'embed'/'message_bits.txt').read_text(),
                             (root/'extracted.txt').read_text())
            # A bit flip must be detected with the manifest unless hidden by parity.
            self.assertEqual(sum(x['m'] for x in info['groups']),info['message_length'])

    def test_bad_fields(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'empty.csv';p.write_text('frame,ctu\n',encoding='utf-8')
            with self.assertRaises(ValueError):read_candidates(p)

if __name__=='__main__':
    unittest.main()
