from __future__ import annotations

import unittest

from tests.parity._parity_utils import load_pair, parity_skip
from tests.parity._shared import parity_tolerances


class PoseParityTest(unittest.TestCase):
    def test_pose_parity(self) -> None:
        tolerances = parity_tolerances("pose")
        python_payload, cpp_payload = load_pair("pose")
        if python_payload is None or cpp_payload is None:
            self.skipTest(parity_skip("pose"))

        self.assertEqual(len(python_payload["images"]), len(cpp_payload["images"]))
        for python_image, cpp_image in zip(
            python_payload["images"], cpp_payload["images"]
        ):
            self.assertLessEqual(
                abs(len(python_image["poses"]) - len(cpp_image["poses"])),
                tolerances["count_delta"],
            )
            if not python_image["poses"] or not cpp_image["poses"]:
                continue

            python_top = python_image["poses"][0]
            cpp_top = cpp_image["poses"][0]
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

            self.assertEqual(
                len(python_top["keypoints"]), len(cpp_top["keypoints"])
            )
            for python_kp, cpp_kp in zip(
                python_top["keypoints"][:5], cpp_top["keypoints"][:5]
            ):
                for python_value, cpp_value in zip(
                    python_kp["point"], cpp_kp["point"]
                ):
                    self.assertAlmostEqual(
                        python_value,
                        cpp_value,
                        delta=tolerances["keypoint_point_delta"],
                    )
                self.assertAlmostEqual(
                    python_kp["score"],
                    cpp_kp["score"],
                    delta=tolerances["keypoint_score_delta"],
                )


if __name__ == "__main__":
    unittest.main()
