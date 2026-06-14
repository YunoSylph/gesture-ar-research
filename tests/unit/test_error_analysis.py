import unittest

import numpy as np

from research_pipeline.evaluation.error_analysis import analyze_recognition_risk
from research_pipeline.labels import TARGET_LABELS, label_to_index


class ErrorAnalysisTests(unittest.TestCase):
    def test_no_gesture_false_action_rate(self):
        matrix = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=int)
        no_index = label_to_index("no_gesture")
        left_index = label_to_index("swipe_left")
        matrix[no_index, no_index] = 8
        matrix[no_index, left_index] = 2
        report = {"recognition": {"confusion_matrix": matrix.tolist()}}
        risk = analyze_recognition_risk(report)
        self.assertEqual(risk["no_gesture_false_action_total"], 2)
        self.assertAlmostEqual(risk["no_gesture_false_action_rate"], 0.2)
        self.assertEqual(risk["no_gesture_false_swipe_total"], 2)
        self.assertEqual(risk["no_gesture_false_actions"]["swipe_left"], 2)


if __name__ == "__main__":
    unittest.main()
