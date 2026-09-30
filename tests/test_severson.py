"""The case study's headline numbers (Section 4), when severson_cells.npz is
available (skipped otherwise; see DATA.md)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiloss import datasets, direction  # noqa: E402

HAVE = os.path.exists(datasets.DATA["severson"])


@unittest.skipUnless(HAVE, "severson_cells.npz not found (set HI_DATA)")
class TestSeverson(unittest.TestCase):
    def test_arc_and_bound(self):
        units, _ = datasets.load_severson()
        self.assertEqual(len(units), 129)
        fd = direction.observed_direction(units, window=301)
        self.assertAlmostEqual(fd.arc_deg, 30.7, places=1)
        self.assertAlmostEqual(100 * (fd.bound() - 1), 7.5, places=1)
        self.assertAlmostEqual(100 * (np.sqrt(fd.bound()) - 1), 3.7, places=1)
        np.testing.assert_allclose(fd.indicator, [0.99, 0.10, 0.12], atol=0.005)
        self.assertEqual(round(fd.max_angle_deg()), 14)


if __name__ == "__main__":
    unittest.main()
