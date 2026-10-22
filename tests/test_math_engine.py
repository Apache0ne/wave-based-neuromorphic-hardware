import json
import math
import pathlib
import tempfile
import unittest
import numpy as np
import sympy as sp
from math_engine.exact import expr, solve, MathInputError, solution_key
from math_engine.questions import questions,KINDS
from math_engine.benchmark import physical_readout,load_native_cube


class MathSafetyTests(unittest.TestCase):
    def test_reject_unsafe_ast(self):
        for inp in ['__import__("os").system("echo hi")', 'x.__class__','[1,2]',
                    'x[0]', 'unknown(5)', '2**200','3/0','x**x','lambda x:x']:
            with self.subTest(inp=inp),self.assertRaises(MathInputError):expr(inp)
    def test_domain_formula_examples(self):
        for prompt,expected in [('Solve for x: 3*x+7=22',5),('Evaluate: sin(pi/6)',sp.Rational(1,2)),
                                ('Factor: x**2-5*x+6',(sp.Symbol('x')-2)*(sp.Symbol('x')-3)),
                                ('Percent: 15% of 80',12),('Circle area: radius=3',9*sp.pi)]:
            with self.subTest(prompt=prompt):
                self.assertEqual(solution_key(solve(prompt)),solution_key(expected))
    def test_exact_600_seeded(self):
        for seed in (11,20261022):
            for q in questions(seed,30):
                with self.subTest(seed=seed,kind=q.kind,q=q.prompt):
                    self.assertEqual(solution_key(solve(q.prompt)),solution_key(q.expected))
    def test_generator_deterministic_and_diverse(self):
        a=questions(20261022,3);b=questions(20261022,3)
        self.assertEqual(a,b)
        self.assertEqual(set(KINDS),{q.kind for q in a})
    def test_group_split_and_folds(self):
        rng=np.random.default_rng(2)
        xx=np.asarray([(i,j) for i in range(1,7) for j in range(1,7) for _ in range(5)])
        wave=rng.normal(size=(len(xx),12)).astype(np.float32)
        rep=physical_readout(wave,xx,seed=27,max_voxels=6)
        self.assertEqual(set(rep),{'random_rows','unseen_operand_pairs'})
        self.assertEqual(len(rep['unseen_operand_pairs']['results']),5)
    def test_load_native_cube(self):
        with tempfile.TemporaryDirectory() as d:
            d=pathlib.Path(d)
            cube=np.arange(4*3*8*8,dtype=np.float32).reshape(4,3,8,8)
            np.save(d/'cube.npy',cube)
            (d/'index.json').write_text(json.dumps({str(i):{'X1':i//2,'X2':i%2} for i in range(4)}))
            f,coords=load_native_cube(d/'cube.npy',d/'index.json',frames=2,side=4)
            self.assertEqual(f.shape,(4,32))
            self.assertEqual(coords.shape,(4,2))
            self.assertAlmostEqual(float(f[0,0]),4.5)

if __name__ == '__main__':unittest.main()
