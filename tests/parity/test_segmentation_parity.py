from __future__ import annotations

import unittest

from tests.parity._parity_utils import load_pair, mask_iou, parity_skip
from tests.parity._shared import parity_tolerances


class SegmentationParityTest(unittest.TestCase):
    def test_segmentation_parity(self) -> None:
        tolerances = parity_tolerances("seg")
        python_payload, cpp_payload = load_pair("seg")
        if python_payload is None or cpp_payload is None:
            self.skipTest(parity_skip("seg"))

        self.assertEqual(len(python_payload["images"]), len(cpp_payload["images"]))
        for python_image, cpp_image in zip(
            python_payload["images"], cpp_payload["images"]
        ):
            self.assertLessEqual(
                abs(len(python_image["instances"]) - len(cpp_image["instances"])),
                tolerances["count_delta"],
            )
            if not python_image["instances"] or not cpp_image["instances"]:
                continue

            python_top = python_image["instances"][0]
            cpp_top = cpp_image["instances"][0]
            self.assertEqual(python_top["class_id"], cpp_top["class_id"])
            self.assertAlmostEqual(
                python_top["score"],
                cpp_top["score"],
                delta=tolerances["score_delta"],
            )
            for python_value, cpp_value in zip(python_top["bbox"], cpp_top["bbox"]):
                self.assertAlmostEqual(
                    python_value,
                    cpp_value,
                    delta=tolerances["bbox_delta"],
                )

            self.assertAlmostEqual(
                python_top["mask"]["area"],
                cpp_top["mask"]["area"],
                delta=tolerances["mask_area_delta"],
            )
            self.assertGreaterEqual(
                mask_iou(python_top["mask"], cpp_top["mask"]),
                tolerances["mask_iou_min"],
            )


if __name__ == "__main__":
    unittest.main()
