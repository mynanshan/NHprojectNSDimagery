import unittest

import numpy as np

from nsdimagery.encoding import (
    assign_image_splits,
    core_nsd_trial_ids,
    fit_ridge_weights,
    pool_transformer_hidden_state,
    voxelwise_prediction_metrics,
)


class EncodingHelperTests(unittest.TestCase):
    def test_core_design_is_beta_session_order(self):
        subjectim = np.tile(np.arange(1, 10001), (8, 1))
        masterordering = np.tile(np.arange(1, 10001), 3)[None, :]
        trial_10k, trial_73k, subject_73k = core_nsd_trial_ids(
            {"subjectim": subjectim, "masterordering": masterordering},
            1,
            n_sessions=1,
        )
        np.testing.assert_array_equal(trial_10k, np.arange(750))
        np.testing.assert_array_equal(trial_73k, np.arange(1, 751))
        np.testing.assert_array_equal(subject_73k, np.arange(1, 10001))

    def test_image_splits_are_unique_and_deterministic(self):
        indices = np.arange(1200)
        first = assign_image_splits(indices, seed=9)
        second = assign_image_splits(indices, seed=9)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:1000], "test")
        self.assertTrue(np.any(first == "validation"))
        self.assertTrue(np.any(first == "train"))

    def test_ridge_recovers_synthetic_voxels(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(80, 12)).astype(np.float32)
        true_weights = rng.normal(size=(12, 9)).astype(np.float32)
        targets = features @ true_weights
        weights = fit_ridge_weights(features, targets, 1e-4, device="cpu")
        predicted = features @ weights
        correlation, r_squared = voxelwise_prediction_metrics(targets, predicted)
        self.assertGreater(float(np.min(correlation)), 0.999)
        self.assertGreater(float(np.min(r_squared)), 0.999)

    def test_transformer_spatial_pyramid_shape(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed")
        hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        pooled = pool_transformer_hidden_state(
            hidden, grid_size=(2, 2), pyramid_levels=(1, 2), include_cls=True
        )
        self.assertEqual(tuple(pooled.shape), (2, 18))
        torch.testing.assert_close(pooled[:, :3], hidden[:, 0, :])


if __name__ == "__main__":
    unittest.main()
