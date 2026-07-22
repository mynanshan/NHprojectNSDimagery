import unittest

import numpy as np

from nsdimagery.analysis import (
    balanced_split_identification,
    correlation_rdm,
    crossvalidated_dot_rdm,
    exact_class_label_permutation_test,
    leave_one_target_out_rdm_spearman,
    nearest_centroid_predict,
    rdm_noise_ceiling,
)


class MeasurementHelperTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.targets = np.repeat(np.arange(1, 4), 8)
        centers = np.eye(3, 12) * 4
        self.patterns = np.vstack([
            centers[target - 1] + rng.normal(scale=0.15, size=12)
            for target in self.targets
        ])

    def test_nearest_centroid_predicts_separated_targets(self):
        predicted = nearest_centroid_predict(
            self.patterns[::2], self.targets[::2], self.patterns[1::2]
        )
        np.testing.assert_array_equal(predicted, self.targets[1::2])

    def test_balanced_split_identification(self):
        values = balanced_split_identification(
            self.patterns, self.targets, n_splits=10, seed=5
        )
        np.testing.assert_allclose(values, 1)

    def test_exact_class_permutation(self):
        result = exact_class_label_permutation_test(self.targets, self.targets)
        self.assertEqual(result["n_permutations"], 6)
        self.assertAlmostEqual(result["accuracy"], 1)
        self.assertAlmostEqual(result["p_greater"], 1 / 6)

    def test_crossvalidated_dot_distance_is_positive(self):
        rdm, order = crossvalidated_dot_rdm(
            self.patterns[::2], self.targets[::2],
            self.patterns[1::2], self.targets[1::2],
        )
        np.testing.assert_array_equal(order, np.arange(1, 4))
        self.assertTrue(np.all(rdm[np.triu_indices(3, k=1)] > 0))

    def test_leave_one_target_out_and_noise_ceiling(self):
        base = correlation_rdm(np.asarray([
            [0, 1, 2, 0, 1, 3],
            [1, 1, 0, 2, 3, 0],
            [3, 0, 1, 1, 0, 2],
            [0, 2, 0, 3, 1, 1],
            [2, 3, 1, 0, 0, 1],
        ], dtype=float))
        perturbation = np.asarray([
            [0, .01, -.01, .02, 0],
            [.01, 0, .02, -.01, 0],
            [-.01, .02, 0, 0, .01],
            [.02, -.01, 0, 0, -.01],
            [0, 0, .01, -.01, 0],
        ])
        comparison = base + perturbation
        sensitivity = leave_one_target_out_rdm_spearman(base, comparison)
        self.assertEqual(sensitivity.shape, (5,))

        varied = np.stack([base, base + perturbation, base - perturbation])
        ceiling = rdm_noise_ceiling(varied)
        self.assertEqual(ceiling["lower_by_subject"].shape, (3,))
        self.assertGreater(ceiling["lower_mean"], .9)


if __name__ == "__main__":
    unittest.main()
