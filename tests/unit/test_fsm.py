import unittest

from research_pipeline.interaction.fsm import ContextAwarePolicy, ContextPolicyConfig
from research_pipeline.models.common import Prediction


class FSMTests(unittest.TestCase):
    def test_requires_stable_frames_and_cooldown(self):
        policy = ContextAwarePolicy(ContextPolicyConfig(activation_threshold=0.6, stable_frames=2, cooldown_ms=250))
        self.assertIsNone(policy.update(Prediction("swipe_right", 0.8, {}), 0))
        event = policy.update(Prediction("swipe_right", 0.8, {}), 100)
        self.assertIsNotNone(event)
        self.assertEqual(event.action, "navigate_next")
        self.assertIsNone(policy.update(Prediction("swipe_right", 0.8, {}), 200))


if __name__ == "__main__":
    unittest.main()

