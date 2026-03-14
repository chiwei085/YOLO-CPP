from __future__ import annotations

import unittest

from tests.parity._parity_utils import load_pair, parity_skip
from tests.parity._shared import parity_tolerances


class ClassificationParityTest(unittest.TestCase):
    def test_classification_parity(self) -> None:
        tolerances = parity_tolerances("classify")
        python_payload, cpp_payload = load_pair("classify")
        if python_payload is None or cpp_payload is None:
            self.skipTest(parity_skip("classify"))

        self.assertEqual(len(python_payload["images"]), len(cpp_payload["images"]))
        for python_image, cpp_image in zip(
            python_payload["images"], cpp_payload["images"]
        ):
            python_top = python_image["classes"][0]
            cpp_top = cpp_image["classes"][0]
            self.assertEqual(python_top["class_id"], cpp_top["class_id"])

            python_topk = [item["class_id"] for item in python_image["classes"][:5]]
            cpp_topk = [item["class_id"] for item in cpp_image["classes"][:5]]
            self.assertEqual(python_topk, cpp_topk)

            for python_score, cpp_score in zip(
                python_image["scores"][:5], cpp_image["scores"][:5]
            ):
                self.assertAlmostEqual(
                    python_score,
                    cpp_score,
                    delta=tolerances["score_delta"],
                )


if __name__ == "__main__":
    unittest.main()
