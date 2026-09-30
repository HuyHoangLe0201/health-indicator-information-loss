"""Steps 1-5 run end to end on the demo fleet (no data needed), and the
synthetic system reproduces the steering factor of Section 5.2."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))
from demo_fleet import demo_fleet  # noqa: E402
from hiloss import direction, policy, rul, synthetic  # noqa: E402


class TestProcedure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = demo_fleet(n_units=20, seed=3)
        cls.fd = direction.observed_direction(cls.units, window=101)

    def test_steps_1_to_3(self):
        s = self.fd.summary()
        self.assertGreater(s["arc_deg"], 1.0)
        # the barycentre stays within half the arc for this gently curved path
        self.assertLessEqual(s["max_angle_deg"], s["arc_deg"] / 2 + 1e-6)
        self.assertAlmostEqual(np.linalg.norm(self.fd.indicator), 1.0)

    def test_step_4_and_5(self):
        model = rul.FleetModel(self.units, window=101)
        res = rul.leave_one_out(model, ages=(0.7,),
                                names=("all", "barycentre", "channel:2"),
                                lmax=1400)
        ratios, _ = rul.mse_ratios(res)
        self.assertTrue(np.isfinite(ratios["barycentre"][0.7]))
        pred = policy.inspection_predictions(model, names=("all",), dn=50,
                                             horizon=900)
        out = policy.evaluate(pred, model.lives, thetas=range(0, 400, 10),
                              cost_ratios=(10.0,))
        self.assertGreater(out["all"][10.0]["rate"], 0.0)


class TestSyntheticSystem(unittest.TestCase):
    def test_steering_factor(self):
        """Section 5.2: the myopic policy at q* against the best constant input
        at the same indicator (about a minute). The unrounded factor is 68.59."""
        L_my = synthetic.myopic(synthetic.QS)[0]
        L_co = synthetic.best_constant(synthetic.QS)[0]
        self.assertAlmostEqual(L_my / 1e-4, 1.68, places=2)
        self.assertAlmostEqual(L_co / 1e-2, 1.15, places=2)
        self.assertAlmostEqual(L_co / L_my, 68.59, delta=0.01)

    def test_temperature_mapping(self):
        self.assertAlmostEqual(synthetic.celsius(0.0), 60.0, places=9)
        self.assertAlmostEqual(synthetic.celsius(3.0), 25.0, places=9)
        self.assertAlmostEqual(synthetic.celsius(1.5), 41.5, places=1)
        np.testing.assert_allclose([synthetic.activation_energy_eV(e) for e in synthetic.E],
                                   [0.44, 0.81, 0.59], atol=0.005)


if __name__ == "__main__":
    unittest.main()
