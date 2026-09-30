"""The closed forms of Sections 2 and 3, checked against direct computation.

    python -m unittest discover tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiloss import core  # noqa: E402


class TestLoss(unittest.TestCase):
    def test_variance_ratio_is_exp_two_loss(self):
        """eq. (8): Var[R | z] / Var[R | record] = e^{2 l}, by Gaussian
        conditioning on a whitened record y = d R + noise."""
        rng = np.random.default_rng(1)
        for _ in range(200):
            p = rng.integers(2, 7)
            d = rng.normal(size=p)
            v = rng.normal(size=p)
            v /= np.linalg.norm(v)
            V = 10 ** rng.uniform(-1, 3)
            post_rec = 1.0 / (1.0 / V + d @ d)       # all channels
            post_ind = 1.0 / (1.0 / V + (v @ d) ** 2)  # z = v' y
            psi = core.angle(v, d)
            gamma = (d @ d) * V
            np.testing.assert_allclose(post_ind / post_rec,
                                       core.variance_inflation(psi, gamma),
                                       rtol=1e-9)

    def test_two_forms_of_the_loss(self):
        psi = np.linspace(0, np.pi / 2, 50)
        for g in (0.1, 2.0, 1e3):
            a = 0.5 * np.log((1 + g) / (1 + g * np.cos(psi) ** 2))
            np.testing.assert_allclose(core.loss(psi, g), a, rtol=1e-12, atol=1e-15)

    def test_margin_is_square_root_of_variance(self):
        np.testing.assert_allclose(core.margin_inflation(0.4, 5.0) ** 2,
                                   core.variance_inflation(0.4, 5.0), rtol=1e-12)


class TestArcBound(unittest.TestCase):
    def test_values_quoted_in_section_3(self):
        """30 deg: at most 7.2 %; 60 deg: 33 %; 90 deg: a doubling."""
        b = [float(core.arc_bound(np.radians(x))) for x in (30, 60, 90)]
        self.assertAlmostEqual(100 * (b[0] - 1), 7.18, places=2)
        self.assertAlmostEqual(100 * (b[1] - 1), 33.33, places=2)
        self.assertAlmostEqual(b[2], 2.0, places=12)

    def test_gamma_bound_below_gamma_free_bound(self):
        for phi in np.radians([10, 45, 120]):
            self.assertLess(core.arc_bound(phi, 5.0), core.arc_bound(phi))

    def test_midpoint_is_within_half_the_arc(self):
        """The argument of Section 3.2 on random curved paths."""
        rng = np.random.default_rng(2)
        for _ in range(50):
            p = rng.integers(3, 6)
            a, b, c = rng.normal(size=(3, p))
            t = np.linspace(0, 1, 400)[:, None]
            P = a + t * b + t ** 2 * c
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            seg = np.arccos(np.clip((P[1:] * P[:-1]).sum(1), -1, 1))
            cum = np.r_[0, np.cumsum(seg)]
            mid = P[np.searchsorted(cum, cum[-1] / 2)]
            self.assertLessEqual(core.max_angle(mid, P), cum[-1] / 2 + seg.max())
            # hence the loss of an indicator there obeys eq. (12), up to the
            # sampling of the path (the midpoint is a sample, not exact)
            psi = core.angle(mid[None, :], P)
            self.assertTrue(np.all(core.variance_inflation(psi, 1e9)
                                   <= core.arc_bound(cum[-1] + 2 * seg.max())))

    def test_path_midpoint_is_always_covered(self):
        """core.path_midpoint interpolates the exact midpoint, so every
        sample of the path lies within arc/2 of it: the fallback of step 3
        always passes the check."""
        rng = np.random.default_rng(3)
        for _ in range(100):
            p = rng.integers(3, 6)
            a, b, c = rng.normal(size=(3, p))
            t = np.linspace(0, 1, rng.integers(20, 400))[:, None]
            P = a + t * b + t ** 2 * c
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            m = core.path_midpoint(P)
            self.assertLessEqual(core.max_angle(m, P),
                                 core.arc_length(P) / 2 + 1e-9)

    def test_arc_length_of_a_great_circle(self):
        th = np.linspace(0, np.pi / 3, 200)
        P = np.column_stack([np.cos(th), np.sin(th), 0 * th])
        self.assertAlmostEqual(core.arc_length(P), np.pi / 3, places=10)


if __name__ == "__main__":
    unittest.main()
