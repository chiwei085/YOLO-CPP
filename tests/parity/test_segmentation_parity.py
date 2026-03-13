from __future__ import annotations

import unittest

from tests.parity._parity_utils import load_pair, mask_iou


class SegmentationParityTest(unittest.TestCase):
    def test_segmentation_parity(self) -> None:
        python_payload, cpp_payload = load_pair("seg")
        if python_payload is None or cpp_payload is None:
            self.skipTest("run_parity.py has not generated segmentation parity JSON yet")

        self.assertEqual(len(python_payload["images"]), len(cpp_payload["images"]))
        for python_image, cpp_image in zip(
            python_payload["images"], cpp_payload["images"]
        ):
            self.assertLessEqual(
                abs(len(python_image["instances"]) - len(cpp_image["instances"])), 1
            )
            if not python_image["instances"] or not cpp_image["instances"]:
                continue

            python_top = python_image["instances"][0]
            cpp_top = cpp_image["instances"][0]
            self.assertEqual(python_top["class_id"], cpp_top["class_id"])
            self.assertAlmostEqual(python_top["score"], cpp_top["score"], delta=0.2)
            for python_value, cpp_value in zip(python_top["bbox"], cpp_top["bbox"]):
                self.assertAlmostEqual(python_value, cpp_value, delta=8.0)

            self.assertAlmostEqual(
                python_top["mask"]["area"], cpp_top["mask"]["area"], delta=2000
            )
            self.assertGreaterEqual(mask_iou(python_top["mask"], cpp_top["mask"]), 0.5)


if __name__ == "__main__":
    unittest.main()
