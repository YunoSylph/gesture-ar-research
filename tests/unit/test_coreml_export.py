import unittest

from research_pipeline.models.coreml_export import convert_tcn_artifact_to_coreml
from research_pipeline.utils.errors import PipelineError


class CoreMLExportGuardTests(unittest.TestCase):
    def test_rejects_non_tcn_artifact(self):
        # The model_type guard runs before any torch/coremltools import, so this
        # stays a fast unit test (the real conversion is verified end-to-end).
        with self.assertRaises(PipelineError):
            convert_tcn_artifact_to_coreml(
                {"model_type": "c0_rule"}, "/tmp/should-not-be-created.mlpackage"
            )


if __name__ == "__main__":
    unittest.main()
