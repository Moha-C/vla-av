import unittest

from scripts.pretrain_dreamer_rl_from_v1 import split_indices


class DistillationValidationTest(unittest.TestCase):
    def test_validation_holds_out_a_complete_route(self):
        samples = [
            {"route_id": "10", "category": "defer"},
            {"route_id": "10", "category": "recovery"},
            {"route_id": "12", "category": "defer"},
            {"route_id": "12", "category": "safety"},
            {"route_id": "13", "category": "recovery"},
        ]
        train, validation, held_out = split_indices(
            samples,
            seed=7,
            validation_route="12",
        )
        self.assertEqual(held_out, "12")
        self.assertEqual({samples[index]["route_id"] for index in validation}, {"12"})
        self.assertNotIn("12", {samples[index]["route_id"] for index in train})
        self.assertFalse(set(train) & set(validation))


if __name__ == "__main__":
    unittest.main()
