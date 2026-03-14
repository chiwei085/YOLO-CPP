from __future__ import annotations

import math
import unittest

from tests.parity._parity_utils import load_pair, parity_skip
from tests.parity._shared import parity_tolerances


def max_abs_diff(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs):
        return math.inf
    return max((abs(left - right) for left, right in zip(lhs, rhs)), default=0.0)


class ObbParityTest(unittest.TestCase):
    def test_obb_parity(self) -> None:
        tolerances = parity_tolerances("obb")
        python_payload, cpp_payload = load_pair("obb")
        if python_payload is None or cpp_payload is None:
            self.skipTest(parity_skip("obb"))

        self.assertEqual(len(python_payload["images"]), len(cpp_payload["images"]))
        for python_image, cpp_image in zip(
            python_payload["images"], cpp_payload["images"]
        ):
            self.assertLessEqual(
                abs(len(python_image["boxes"]) - len(cpp_image["boxes"])),
                tolerances["count_delta"],
            )
            if not python_image["boxes"] or not cpp_image["boxes"]:
                continue

            python_top = python_image["boxes"][0]
            cpp_top = cpp_image["boxes"][0]
            self.assertEqual(python_top["class_id"], cpp_top["class_id"])
            self.assertAlmostEqual(
                python_top["score"],
                cpp_top["score"],
                delta=tolerances["score_delta"],
            )
            self.assertLessEqual(
                max_abs_diff(python_top["center"], cpp_top["center"]),
                tolerances["center_delta"],
            )
            self.assertLessEqual(
                max_abs_diff(python_top["size"], cpp_top["size"]),
                tolerances["size_delta"],
            )
            self.assertAlmostEqual(
                python_top["angle_radians"],
                cpp_top["angle_radians"],
                delta=tolerances["angle_delta"],
            )
            for python_corner, cpp_corner in zip(
                python_top["corners"], cpp_top["corners"]
            ):
                self.assertLessEqual(
                    max_abs_diff(python_corner, cpp_corner),
                    tolerances["corner_delta"],
                )


if __name__ == "__main__":
    unittest.main()
