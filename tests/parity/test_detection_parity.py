from __future__ import annotations

import unittest

from tests.parity._parity_utils import load_pair


class DetectionParityTest(unittest.TestCase):
    def test_detection_parity(self) -> None:
        python_payload, cpp_payload = load_pair("detect")
        if python_payload is None or cpp_payload is None:
            self.skipTest("run_parity.py has not generated detect parity JSON yet")

        self.assertEqual(len(python_payload["images"]), len(cpp_payload["images"]))
        for python_image, cpp_image in zip(
            python_payload["images"], cpp_payload["images"]
        ):
            self.assertLessEqual(
                abs(len(python_image["detections"]) - len(cpp_image["detections"])), 2
            )
            if not python_image["detections"] or not cpp_image["detections"]:
                continue

            python_top = python_image["detections"][0]
            cpp_top = cpp_image["detections"][0]
            self.assertEqual(python_top["class_id"], cpp_top["class_id"])
            self.assertAlmostEqual(python_top["score"], cpp_top["score"], delta=0.15)
            for python_value, cpp_value in zip(python_top["bbox"], cpp_top["bbox"]):
                self.assertAlmostEqual(python_value, cpp_value, delta=6.0)


if __name__ == "__main__":
    unittest.main()
