import unittest
from src.collect.plan import jitter_lateral_sign

class TestLateralConvention(unittest.TestCase):
    def test_rebased_side_and_original_reproduce_same_shift(self):
        self.assertEqual(jitter_lateral_sign('right', '77e745c37b0f'), jitter_lateral_sign('left', '08f1c9df8207'))
        self.assertEqual(jitter_lateral_sign('left', '77e745c37b0f'), jitter_lateral_sign('right', '08f1c9df8207'))
    def test_new_plans_preserve_fixture_side(self):
        for pid in ['08f1c9df8207','4ab360df2b71','4fa6496cd684','d8c7e569aa37',None]:
            self.assertEqual(jitter_lateral_sign('left',pid), 1)
            self.assertEqual(jitter_lateral_sign('right',pid), -1)
    def test_back_only_drawer_needs_no_lateral_side(self):
        for pid in ["77e745c37b0f", "08f1c9df8207"]:
            self.assertEqual(jitter_lateral_sign(None, pid, lateral=0), 0)
            with self.assertRaises(ValueError): jitter_lateral_sign(None, pid, lateral=.05)
    def test_invalid_side_rejected(self):
        with self.assertRaises(ValueError): jitter_lateral_sign('unknown','77e745c37b0f')
