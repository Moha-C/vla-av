import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.residual_dreamerv3.baselines import (
    Normalization,
    PersistenceBaseline,
    RidgeDynamicsBaseline,
    build_gate_report,
    evaluate_baseline,
)
from src.residual_dreamerv3.config import ResidualDreamerConfig
from src.residual_dreamerv3.data import Episode, SequenceDataset, stratified_seed_split
from src.residual_dreamerv3.model import ResidualDreamerV3
from src.residual_dreamerv3.runtime import (
    CheckpointNotPromotedError,
    ResidualDreamerRuntime,
)
from src.residual_dreamerv3.training import closed_loop_promotion_checks, lambda_returns
from src.residual_dreamerv3.transforms import symexp, symlog, two_hot, two_hot_mean


def tiny_config():
    config = ResidualDreamerConfig()
    config.data.sequence_length = 4
    config.data.minimum_train_seeds = 3
    config.data.minimum_validation_seeds = 2
    config.data.minimum_test_seeds = 2
    config.model.encoder_dim = 16
    config.model.hidden_dim = 24
    config.model.deterministic_size = 12
    config.model.stochastic_size = 3
    config.model.categorical_classes = 4
    config.model.reward_bins = 31
    config.model.value_bins = 31
    config.actor.hidden_dim = 24
    config.actor.imagination_horizon = 3
    config.gate.horizons = (1, 2, 4)
    config.validate()
    return config


def episode(seed, route="route", transitions=8, offset=0.0):
    rng = np.random.RandomState(int(seed) + 10)
    actions = rng.uniform(-0.2, 0.2, size=(transitions, 3)).astype(np.float32)
    actions[:, 1:] = np.abs(actions[:, 1:])
    observations = np.zeros((transitions + 1, 32), dtype=np.float32)
    observations[0] = offset
    for index in range(transitions):
        observations[index + 1] = observations[index]
        observations[index + 1, :3] += np.asarray(
            [actions[index, 1] - actions[index, 2], actions[index, 0], 0.1],
            dtype=np.float32,
        )
    rewards = actions[:, 1] - actions[:, 2]
    continuation = np.ones(transitions, dtype=np.float32)
    continuation[-1] = 0.0
    zeros = np.zeros(transitions, dtype=np.float32)
    return Episode(
        key="%s:%s" % (route, seed),
        seed=str(seed),
        path=Path("/%s/%s" % (route, seed)),
        metadata={
            "town": "Town12",
            "scenario": "Accident",
            "route_id": route,
            "trace_sha256": str(seed),
            "event_timing_quality": "none",
        },
        observations=observations,
        actions=actions,
        rewards=rewards.astype(np.float32),
        continuation=continuation,
        collision=zeros.copy(),
        offroad=zeros.copy(),
        risk=zeros.copy(),
        progress=np.maximum(rewards, 0.0).astype(np.float32),
    )


class ResidualDreamerV3Test(unittest.TestCase):
    def test_symlog_round_trip_and_two_hot(self):
        values = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        torch.testing.assert_close(symexp(symlog(values)), values)
        bins = torch.linspace(-5.0, 5.0, 31)
        labels = two_hot(values, bins)
        torch.testing.assert_close(labels.sum(dim=-1), torch.ones(len(values)))
        prediction = two_hot_mean(torch.log(labels + 1.0e-6), bins)
        self.assertTrue(torch.isfinite(prediction).all())

    def test_model_loss_and_actor_are_finite(self):
        config = tiny_config()
        model = ResidualDreamerV3(config)
        observations = torch.randn(2, 5, 32) * 0.1
        actions = torch.rand(2, 4, 3) * 0.2
        targets = {
            "rewards": torch.zeros(2, 4),
            "continuation": torch.ones(2, 4),
            "risk": torch.zeros(2, 4),
            "collision": torch.zeros(2, 4),
            "offroad": torch.zeros(2, 4),
        }
        loss, parts = model.world_model.loss(observations, actions, targets, config.loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
        state = model.world_model.observe_initial(observations[:, 0], deterministic=True)
        actor = model.actor(model.world_model.feature(state), observations[:, 0], deterministic=True)
        self.assertTrue(torch.allclose(actor.authority, torch.full((2,), 0.02), atol=1.0e-4))
        self.assertTrue(torch.all((actor.final_action[:, 0] >= -1.0) & (actor.final_action[:, 0] <= 1.0)))
        self.assertTrue(torch.all((actor.final_action[:, 1:] >= 0.0) & (actor.final_action[:, 1:] <= 1.0)))
        self.assertFalse(torch.any((actor.final_action[:, 1] > 0.0) & (actor.final_action[:, 2] > 0.0)))

    def test_zero_delta_decoder_is_exact_persistence(self):
        config = tiny_config()
        model = ResidualDreamerV3(config)
        for parameter in model.world_model.decoder.parameters():
            torch.nn.init.zeros_(parameter)
        reference = torch.randn(2, 32)
        state = model.world_model.observe_initial(reference, deterministic=True)
        prediction = model.world_model.prediction(state, reference)
        torch.testing.assert_close(prediction.observation, reference)

    def test_seed_split_is_disjoint_and_sequences_are_consecutive(self):
        config = tiny_config()
        episodes = [episode(index, route="route_%d" % (index % 3)) for index in range(1, 10)]
        splits = stratified_seed_split(episodes, config)
        seeds = {key: set(value) for key, value in splits.seed_sets().items()}
        self.assertFalse(seeds["train"] & seeds["validation"])
        self.assertFalse(seeds["train"] & seeds["test"])
        self.assertFalse(seeds["validation"] & seeds["test"])
        dataset = SequenceDataset([episodes[0]], 4)
        row = dataset[0]
        torch.testing.assert_close(row["observations"], torch.from_numpy(episodes[0].observations[:5]))
        torch.testing.assert_close(row["actions"], torch.from_numpy(episodes[0].actions[:4]))

    def test_real_event_windows_receive_higher_sampling_weight(self):
        item = episode(1)
        item.collision[-1] = 1.0
        dataset = SequenceDataset([item], 4)
        weights = dataset.sample_weights(32.0, 4.0, 0.65)
        self.assertEqual(float(weights[-1]), 32.0)
        self.assertEqual(float(weights[0]), 1.0)

    def test_ridge_baseline_beats_persistence_on_action_dynamics(self):
        train = [episode(index, offset=float(index)) for index in range(1, 8)]
        test = [episode(20, offset=20.0)]
        normalization = Normalization.fit(train)
        persistence = evaluate_baseline(PersistenceBaseline(train), test, normalization, (1, 2, 4))
        ridge = evaluate_baseline(RidgeDynamicsBaseline().fit(train), test, normalization, (1, 2, 4))
        self.assertLess(
            ridge["aggregate"]["observation_squared_error"],
            persistence["aggregate"]["observation_squared_error"],
        )

    def test_gate_rejects_collapsed_model(self):
        baseline = {
            "name": "persistence",
            "aggregate": {
                "observation_squared_error": 1.0,
                "reward_absolute_error": 1.0,
                "risk_brier": 1.0,
            },
        }
        model = {
            "name": "rssm",
            "aggregate": {"observation_squared_error": 0.5, "reward_absolute_error": 0.5, "risk_brier": 0.5},
        }
        report = build_gate_report(
            model,
            [baseline],
            {"mean_transition_spread": 0.0, "collapse_fraction": 1.0},
            tiny_config().gate,
            "validation",
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["action_not_collapsed"])

    def test_lambda_returns_respect_terminal_continuation(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        continuation = torch.tensor([[1.0, 0.0, 1.0]])
        values = torch.zeros_like(rewards)
        result = lambda_returns(rewards, continuation, values, torch.tensor([10.0]), 1.0, 1.0)
        torch.testing.assert_close(result, torch.tensor([[3.0, 2.0, 13.0]]))

    def test_runtime_refuses_unpromoted_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            torch.save({"metadata": {"status": "candidate"}}, str(checkpoint))
            with self.assertRaises(CheckpointNotPromotedError):
                ResidualDreamerRuntime(checkpoint)

    def test_candidate_shadow_can_never_override_native_action(self):
        config = tiny_config()
        model = ResidualDreamerV3(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            torch.save(
                {
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "metadata": {"status": "candidate"},
                },
                str(checkpoint),
            )
            runtime = ResidualDreamerRuntime(checkpoint, allow_candidate_shadow=True)
            observation = np.zeros(32, dtype=np.float32)
            observation[2:5] = np.asarray([0.1, 0.7, 0.0], dtype=np.float32)
            result = runtime.step(observation)
            np.testing.assert_allclose(result["action"], observation[2:5])
            self.assertTrue(result["shadow_only"])

    def test_candidate_control_requires_explicit_evaluation_mode(self):
        config = tiny_config()
        model = ResidualDreamerV3(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            torch.save(
                {
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "metadata": {"status": "candidate"},
                },
                str(checkpoint),
            )
            runtime = ResidualDreamerRuntime(
                checkpoint, allow_candidate_evaluation=True
            )
            observation = np.zeros(32, dtype=np.float32)
            observation[2:5] = np.asarray([0.1, 0.7, 0.0], dtype=np.float32)
            result = runtime.step(observation)
            self.assertFalse(result["shadow_only"])
            self.assertTrue(result["evaluation_only"])
            np.testing.assert_allclose(result["action"], result["dreamer_action"])

    def test_closed_loop_promotion_requires_six_seeds_and_no_regression(self):
        valid = {
            "schema_version": "residual_dreamerv3_closed_loop_eval_v1",
            "paired_evaluation": True,
            "seeds": list(range(6)),
            "baseline": {"driving_score": 70.0, "route_completion": 80.0, "collisions_per_km": 1.0, "offroad_rate": 0.1},
            "candidate": {"driving_score": 72.0, "route_completion": 82.0, "collisions_per_km": 0.8, "offroad_rate": 0.1},
        }
        self.assertTrue(all(closed_loop_promotion_checks(valid).values()))
        valid["candidate"]["collisions_per_km"] = 2.0
        self.assertFalse(closed_loop_promotion_checks(valid)["collisions_not_worse"])


if __name__ == "__main__":
    unittest.main()
