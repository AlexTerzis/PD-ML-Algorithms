import unittest

from feature_schema import FEATURE_ORDER, MODEL_VERSION
from main import (
    METADATA,
    TRAIN_MEAN,
    TRAIN_STD,
    _input_assessment,
    _experimental_model_score,
    _prediction_assessment,
)
from praat_extractor import parse_voice_report


GOOD_QUALITY = {
    "duration_seconds": 5.0,
    "voiced_fraction": 0.95,
    "peak_amplitude": 0.5,
    "clipped_fraction": 0.0,
}


class VoicePipelineTest(unittest.TestCase):
    def test_model_and_feature_schema_are_versioned_together(self):
        self.assertEqual(METADATA["model_version"], MODEL_VERSION)
        self.assertEqual(METADATA["feature_order"], FEATURE_ORDER)

    def test_reference_mean_is_usable(self):
        assessment = _input_assessment(dict(TRAIN_MEAN), GOOD_QUALITY)
        self.assertTrue(assessment["usable"])
        self.assertEqual(assessment["status"], "ok")

    def test_unvalidated_model_never_authorizes_a_patient_score(self):
        assessment = _prediction_assessment(
            dict(TRAIN_MEAN), GOOD_QUALITY
        )
        self.assertFalse(assessment["usable"])
        self.assertEqual(
            assessment["status"], "research_model_not_validated"
        )

    def test_prototype_output_is_available_as_a_separate_number(self):
        score = _experimental_model_score(dict(TRAIN_MEAN))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_three_outlying_features_force_abstention(self):
        features = dict(TRAIN_MEAN)
        for name in FEATURE_ORDER[:3]:
            features[name] += 3.5 * TRAIN_STD[name]

        assessment = _input_assessment(features, GOOD_QUALITY)
        self.assertFalse(assessment["usable"])
        self.assertEqual(assessment["status"], "outside_reference_range")

    def test_capture_failure_force_abstention(self):
        quality = {**GOOD_QUALITY, "voiced_fraction": 0.2}
        assessment = _input_assessment(dict(TRAIN_MEAN), quality)
        self.assertFalse(assessment["usable"])
        self.assertEqual(assessment["status"], "capture_quality_failed")

    def test_incomplete_praat_report_fails_instead_of_inserting_zeros(self):
        with self.assertRaisesRegex(RuntimeError, "required fields"):
            parse_voice_report("Mean pitch: 120 Hz")


if __name__ == "__main__":
    unittest.main()
