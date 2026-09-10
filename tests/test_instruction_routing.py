import unittest
from scripts.serve.instruction_routing import validate_routing, resolve_routing

class RoutingTest(unittest.TestCase):
    def spec(self, mode):
        return {'version': 'instruction_setm_routing_v1', 'mode': mode}
    def test_instruction_only_ignores_cluster(self):
        s = validate_routing(self.spec('instruction_only'), ['instruction'])
        for c in ['c0', 'c7', None]:
            self.assertEqual(resolve_routing(s, c, False), ('instruction', None))
    def test_hybrid_preserves_registered_and_falls_back_missing(self):
        s = validate_routing(self.spec('cluster_instruction_fallback'), ['instruction', 'c0'])
        self.assertEqual(resolve_routing(s, 'c0', True), ('c0', None))
        self.assertEqual(resolve_routing(s, 'c1', False), ('instruction', 'instruction:phase_unregistered:c1'))
    def test_missing_instruction_rejected(self):
        with self.assertRaises(ValueError):
            validate_routing(self.spec('cluster_instruction_fallback'), ['c0'])
    def test_legacy_route_unchanged(self):
        self.assertEqual(resolve_routing(None, 'c2', False), ('c2', None))
