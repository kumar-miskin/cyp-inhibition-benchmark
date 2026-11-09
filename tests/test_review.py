"""Independent checks of scientific invariants, not model tuning."""
import unittest
import numpy as np
import pandas as pd
import torch
from benchmark import ENDPOINTS, MolecularGNN, batch_graph, graph, masked_mse, score, st_rae

class ReviewTests(unittest.TestCase):
    def test_interval_distance_scalar_reference(self):
        rng = np.random.default_rng(12)
        for _ in range(100):
            y, p = rng.normal(size=(2, 30))
            lo, hi = y - rng.random(30), y + rng.random(30)
            def distance(q, l, h):
                return l-q if q<l else q-h if q>h else 0.0
            denominator = sum(distance(float(y.mean()), l, h) for l,h in zip(lo,hi))
            expected = sum(distance(q,l,h) for q,l,h in zip(p,lo,hi))/denominator
            self.assertAlmostEqual(st_rae(y,p,lo,hi),expected,places=12)

    def test_missing_labels_excluded_from_score(self):
        df = pd.DataFrame({c: [0.,4.,np.nan] for c in ENDPOINTS})
        for c in ENDPOINTS:
            df[c+'_conf_low'] = [-1.,3.,np.nan]
            df[c+'_conf_high'] = [1.,5.,np.nan]
        pred = np.array([[0.]*4,[4.]*4,[1.e9]*4])
        result = score(df,pred)
        self.assertEqual(result['macro']['ST-RAE'],0.)
        self.assertTrue(all(result[c]['n']==2 for c in ENDPOINTS))

    def test_masked_gradients(self):
        p = torch.tensor([[2.,3.]],requires_grad=True)
        masked_mse(p,torch.tensor([[1.,float('nan')]])).backward()
        torch.testing.assert_close(p.grad,torch.tensor([[2.,0.]]))

    def test_edge_indices_and_disconnected_graphs(self):
        for s in ['[Na+]','[Na+].[Cl-]','C=C','c1ccccc1']:
            x,e = graph(s)
            pairs = set(map(tuple,e.T.tolist()))
            self.assertTrue(all((v,u) in pairs for u,v in pairs))
            self.assertTrue(not e.numel() or (e.min()>=0 and e.max()<len(x)))
            self.assertTrue(torch.isfinite(MolecularGNN()(*batch_graph([(x,e)]))).all())

    def test_zero_denominator_is_undefined(self):
        y=np.array([1.,1.])
        self.assertIsNone(st_rae(y,y,y,y))

    def test_reversed_intervals_rejected(self):
        df = pd.DataFrame({c:[0.,4.] for c in ENDPOINTS})
        for c in ENDPOINTS:
            df[c+'_conf_low']=[1.,5.]
            df[c+'_conf_high']=[-1.,3.]
        with self.assertRaisesRegex(ValueError,'Reversed'):
            score(df,np.zeros((2,4)))
