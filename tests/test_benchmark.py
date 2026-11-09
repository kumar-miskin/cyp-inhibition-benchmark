import sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from benchmark import *


class BenchmarkTests(unittest.TestCase):
    def test_equivalent_smiles(self):
        self.assertEqual(mol_key("CCO"), mol_key("OCC"))

    def test_invalid_smiles(self):
        with self.assertRaises(ValueError):
            mol_key("not-a-molecule")

    def test_scaffold_reproducible(self):
        ss = [
            "c1ccccc1",
            "Cc1ccccc1",
            "c1ccncc1",
            "Cc1ccncc1",
            "C1CCCCC1",
            "CC1CCCCC1",
            "CCO",
            "CCN",
        ]
        a, b = split_indices(ss)
        a2, b2 = split_indices(ss)
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(b, b2)
        self.assertFalse(
            set(scaffold_key(ss[i]) for i in a) & set(scaffold_key(ss[i]) for i in b)
        )

    def test_soft_error(self):
        y = np.array([0.0, 4.0])
        lo = np.array([-1.0, 3.0])
        hi = np.array([1.0, 5.0])
        self.assertEqual(st_rae(y, y, lo, hi), 0.0)
        self.assertEqual(st_rae(y, np.array([2.0, 2.0]), lo, hi), 1.0)

    def test_masked_loss(self):
        self.assertEqual(
            float(
                masked_mse(
                    torch.tensor([[1.0, 100.0]]), torch.tensor([[1.0, float("nan")]])
                )
            ),
            0.0,
        )

    def test_graph_permutation(self):
        torch.manual_seed(SEED)
        m = MolecularGNN().eval()
        a = m(*batch_graph([graph("CCO")]))
        b = m(*batch_graph([graph("OCC")]))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_batched_graph_independence(self):
        torch.manual_seed(SEED)
        m = MolecularGNN().eval()
        a = m(*batch_graph([graph("CCO")]))
        b = m(*batch_graph([graph("CCO"), graph("CCN")]))[:1]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
