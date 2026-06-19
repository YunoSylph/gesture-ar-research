import unittest

from research_pipeline.interaction.contract import (
    golden_validation_traces,
    run_validation_trace,
    validation_contract,
)
from research_pipeline.interaction.fsm import ContextPolicyConfig


class ValidationContractTests(unittest.TestCase):
    def test_contract_reflects_config(self):
        contract = validation_contract(ContextPolicyConfig())
        self.assertEqual(contract["config"]["activation_threshold"], 0.62)
        self.assertEqual(contract["config"]["stable_frames"], 2)
        self.assertEqual(contract["config"]["cooldown_ms"], 250)
        self.assertEqual(contract["action_by_label"]["click_2f"], "select_confirm")

    def test_stability_gate_and_acceptance(self):
        # one frame is ignored; two stable confident frames accept once
        self.assertEqual(run_validation_trace([(0, "click_2f", 0.9)]), [])
        events = run_validation_trace([(0, "click_2f", 0.9), (40, "click_2f", 0.9)])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["action"], "select_confirm")

    def test_below_threshold_is_rejected(self):
        self.assertEqual(run_validation_trace([(0, "click_2f", 0.5), (40, "click_2f", 0.5)]), [])

    def test_cooldown_debounces_repeats(self):
        frames = [(t, "zoom_in", 0.9) for t in (0, 33, 66, 99, 132)]
        events = run_validation_trace(frames)
        self.assertEqual(len(events), 1)  # one accept, rest within cooldown
        self.assertEqual(events[0]["timestamp_ms"], 33)

    def test_golden_traces_have_expected_event_counts(self):
        by_name = {trace["name"]: trace for trace in golden_validation_traces()}
        self.assertEqual(len(by_name["click_accept"]["events"]), 1)
        self.assertEqual(by_name["click_accept"]["events"][0]["action"], "select_confirm")
        self.assertEqual(len(by_name["single_frame_ignored"]["events"]), 0)
        self.assertEqual(len(by_name["below_threshold_rejected"]["events"]), 0)
        self.assertEqual(len(by_name["cooldown_debounces_repeat"]["events"]), 1)
        self.assertEqual(len(by_name["no_gesture_resets_candidate"]["events"]), 1)
        self.assertEqual(by_name["no_gesture_resets_candidate"]["events"][0]["action"], "navigate_next")


if __name__ == "__main__":
    unittest.main()
