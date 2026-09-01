import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_report_dreamer import action_sensitivity_metrics
from scripts.promote_report_dreamer_checkpoint import (
    PREDICTION_LOSS_KEYS,
    validate_prediction_metrics,
)
from scripts.train_report_dreamer import load_pairwise_rows, loader_for
from scripts.verify_report_dreamer_shadow_trace import verify_shadow_rows
from src.world_model.agent import SimLingoDreamerAgent
from src.world_model.authority import LearnedAuthorityController
from src.world_model.config import DreamerConfig
from src.world_model.dataset import (
    DreamerEpisode,
    build_episode,
    policy_source,
    split_by_seed,
)
from src.world_model.observation import (
    DREAMER_OBSERVATION_FEATURES,
    DreamerObservationBuilder,
)
from src.world_model.pairwise import PairwiseCalibrator
from src.world_model.planning import CandidateEvaluator, CandidateGenerator
from src.world_model.policy import LatentCritic, ResidualActor
from src.world_model.rssm import CompactRSSM
from src.world_model.training import imagine_actor_critic


def small_config():
    config = DreamerConfig()
    config.rssm.encoder_dim = 16
    config.rssm.hidden_dim = 32
    config.rssm.deterministic_size = 16
    config.rssm.stochastic_size = 4
    config.rssm.categorical_classes = 4
    config.rssm.free_nats = 0.0
    config.rssm.imagination_horizon = 3
    config.policy.hidden_dim = 32
    config.pairwise.hidden_dim = 16
    config.training.sequence_length = 4
    config.training.batch_size = 4
    return config


def fake_targets(batch, time):
    return {
        "progress": torch.zeros(batch, time),
        "risk": torch.zeros(batch, time),
        "continuation": torch.ones(batch, time),
        "value": torch.zeros(batch, time),
        "collision": torch.zeros(batch, time),
        "offroad": torch.zeros(batch, time),
    }


class ReportAlignedDreamerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def test_01_rssm_forward_fake_sequence(self):
        config = small_config()
        model = CompactRSSM(config.rssm)
        observations = torch.randn(3, 6, 32).clamp(-1, 1)
        actions = torch.randn(3, 5, 3).clamp(-1, 1)
        loss, metrics = model.loss(
            observations, actions, fake_targets(3, 5), config.prediction_loss
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(("progress", "risk", "continuation", "value", "collision", "offroad")) - set(metrics), set())

    def test_02_world_model_learns_tiny_dataset(self):
        config = small_config()
        model = CompactRSSM(config.rssm)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
        actions = torch.zeros(4, 4, 3)
        observations = torch.zeros(4, 5, 32)
        observations[:, :, 0] = 0.2
        targets = fake_targets(4, 4)
        with torch.no_grad():
            initial = float(model.loss(observations, actions, targets, config.prediction_loss)[0])
        for _ in range(35):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(observations, actions, targets, config.prediction_loss)
            loss.backward()
            optimizer.step()
        final = float(model.loss(observations, actions, targets, config.prediction_loss)[0])
        self.assertLess(final, initial)

    def test_03_latent_candidate_rollout(self):
        config = small_config()
        config.candidates.emergency_brake_levels = (1.0,)
        model = CompactRSSM(config.rssm)
        actor = ResidualActor(model.feature_dim, config.policy)
        state = model.observe_initial(torch.zeros(1, 32), deterministic=True)
        native = torch.tensor([0.0, 0.4, 0.0])
        candidates = CandidateGenerator(config.candidates).generate(native, torch.zeros(32))
        self.assertEqual(candidates[0].kind, "native")
        self.assertTrue(torch.equal(candidates[0].proposal, native))
        emergency = next(
            item for item in candidates if item.kind == "emergency_brake_1.00"
        )
        torch.testing.assert_close(
            emergency.proposal, torch.tensor([0.0, 0.0, 1.0])
        )
        result = CandidateEvaluator(model, actor, config.candidates, config.evaluator).imagine(
            state, native, candidates, 3
        )
        self.assertEqual(result.world_actions.shape, (len(candidates), 3, 3))
        self.assertEqual(result.features.shape, (len(candidates), 5))

    def test_action_conditioned_prior_receives_training_gradient(self):
        config = small_config()
        config.prediction_loss.prior_prediction = 1.0
        config.prediction_loss.action_contrastive = 1.0
        config.prediction_loss.action_safety_monotonic = 1.0
        model = CompactRSSM(config.rssm)
        observations = torch.randn(4, 5, 32).clamp(-1, 1)
        actions = torch.rand(4, 4, 3)
        actions[:, :, 0] = actions[:, :, 0] * 2.0 - 1.0
        loss, metrics = model.loss(
            observations, actions, fake_targets(4, 4), config.prediction_loss
        )
        loss.backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.action_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)
        self.assertIn("prior_prediction_total", metrics)
        self.assertIn("action_contrastive", metrics)
        self.assertIn("action_safety_monotonic", metrics)
        self.assertGreater(float(metrics["action_safety_hazard_fraction"]), 0.0)

    def test_04_actor_critic_trains_in_imagination(self):
        config = small_config()
        model = CompactRSSM(config.rssm)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        actor = ResidualActor(model.feature_dim, config.policy)
        critic = LatentCritic(model.feature_dim, config.policy)
        state = model.observe_initial(torch.randn(4, 32)).detach()
        native = torch.tensor([[0.0, 0.3, 0.0]]).repeat(4, 1)
        result = imagine_actor_critic(model, actor, critic, state, native, config)
        result.actor_loss.backward()
        result.critic_loss.backward()
        self.assertIsNotNone(actor.mean.weight.grad)
        self.assertTrue(torch.isfinite(result.objective))

    def test_residual_actor_starts_as_a_low_authority_complement(self):
        config = small_config()
        model = CompactRSSM(config.rssm)
        actor = ResidualActor(model.feature_dim, config.policy)
        state = model.observe_initial(torch.zeros(2, 32), deterministic=True)
        native = torch.tensor([[0.0, 0.4, 0.0]]).repeat(2, 1)
        output = actor(model.feature(state), native, deterministic=True)
        torch.testing.assert_close(
            output.alpha,
            torch.full((2,), config.policy.initial_alpha),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_05_shadow_mode_never_controls(self):
        config = small_config()
        config.runtime.ablation = "D"
        config.runtime.shadow = True
        agent = SimLingoDreamerAgent(config, device="cpu")
        observation = DreamerObservationBuilder(config.observation).build(
            {"speed_mps": 4.0, "left_lane_available": True}, [0.1, 0.5, 0.0]
        )
        native = np.asarray([0.1, 0.5, 0.0], dtype=np.float32)
        result = agent.step(observation, native)
        np.testing.assert_array_equal(result.final_action, native)
        self.assertEqual(result.alpha, 0.0)

    def test_06_alpha_zero_is_bit_exact(self):
        config = small_config()
        controller = LearnedAuthorityController(config.authority)
        native = np.asarray([0.12345679, 0.7654321, 0.0], dtype=np.float32)
        proposal = np.asarray([-0.5, 0.0, 1.0], dtype=np.float32)
        result = controller.blend(native, proposal, 0.0)
        np.testing.assert_array_equal(result, native)

    def test_07_fixed_low_authority(self):
        config = small_config()
        config.authority.learned = False
        config.authority.fixed_alpha = 0.2
        config.authority.smoothing = 0.0
        config.authority.max_delta_per_step = 1.0
        controller = LearnedAuthorityController(config.authority)
        decision = controller.decide(learned_alpha=0.9)
        self.assertAlmostEqual(decision.alpha, 0.2)
        result = controller.blend([0.0, 0.5, 0.0], [0.2, 0.3, 0.0], decision.alpha)
        np.testing.assert_allclose(result, [0.04, 0.46, 0.0], atol=1.0e-6)

    def test_08_learned_authority_is_bounded_and_smoothed(self):
        config = small_config()
        config.authority.max_alpha = 0.7
        config.authority.smoothing = 0.5
        config.authority.max_delta_per_step = 0.1
        controller = LearnedAuthorityController(config.authority)
        first = controller.decide(1.0)
        second = controller.decide(1.0)
        self.assertAlmostEqual(first.alpha, 0.1)
        self.assertAlmostEqual(second.alpha, 0.2)
        self.assertLessEqual(second.alpha, config.authority.max_alpha)

    def test_09_pairwise_and_seed_split(self):
        calibrator = PairwiseCalibrator(5, 8)
        first = torch.tensor([[1.0, 0.1, 1.0, 0.0, 0.1]])
        second = torch.tensor([[0.0, 0.9, 1.0, 0.0, 0.5]])
        probability = calibrator(first, second)
        self.assertEqual(probability.shape, (1,))
        config = small_config()
        episodes = []
        for index in range(6):
            length = 5
            episodes.append(
                DreamerEpisode(
                    key=str(index),
                    seed=str(index),
                    metadata={"route_id": "r%d" % index, "town": "Town12", "scenario": "test"},
                    observations=np.zeros((length + 1, 32), np.float32),
                    actions=np.zeros((length, 3), np.float32),
                    alpha=np.zeros(length, np.float32),
                    progress=np.zeros(length, np.float32),
                    risk=np.zeros(length, np.float32),
                    continuation=np.ones(length, np.float32),
                    collision=np.zeros(length, np.float32),
                    offroad=np.zeros(length, np.float32),
                    reward=np.zeros(length, np.float32),
                    value=np.zeros(length, np.float32),
                )
            )
        splits = split_by_seed(episodes, config)
        splits.verify()
        self.assertFalse(set(splits.seed_sets["train"]) & set(splits.seed_sets["test"]))

    def test_native_trace_provenance_and_ground_truth_are_explicit(self):
        native = {
            "mode": "simlingo_native",
            "variant": "simlingo_native_report_collect",
            "applied": False,
            "alpha": 0.0,
        }
        guarded = dict(native, mode="apply", variant="dreamer_guard_v1", applied=True)
        self.assertEqual(policy_source([native, native]), "simlingo_native")
        self.assertEqual(policy_source([native, guarded]), "non_native_or_unknown")

        config = small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            rows = []
            for index in range(6):
                action = {"steer": 0.0, "throttle": 0.4, "brake": 0.0}
                stored_observation = {
                    name: float(feature_index + 1) / 100.0 + index / 1000.0
                    for feature_index, name in enumerate(
                        DREAMER_OBSERVATION_FEATURES
                    )
                }
                status = {
                    **native,
                    "timestamp": float(index),
                    "base_action": action,
                    "final_action": action,
                    "state_vector": [float(index), 0.0, 4.0] + [0.0] * 25,
                    "front_vehicle_m": 80.0,
                }
                rows.append(
                    {
                        "collector_time": float(index),
                        "route_id": "57",
                        "seed": "101",
                        "town": "Town12",
                        "observation": stored_observation,
                        "status": status,
                    }
                )
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            episode = build_episode(trace, config)
            self.assertIsNotNone(episode)
            self.assertEqual(
                episode.metadata["policy_source"], "simlingo_native"
            )
            self.assertFalse(episode.metadata["event_ground_truth"])
            self.assertEqual(episode.actions.shape[1], 3)
            np.testing.assert_array_equal(episode.alpha, 0.0)
            np.testing.assert_allclose(
                episode.observations[0],
                np.asarray(list(rows[0]["observation"].values()), dtype=np.float32),
            )
            (root / "episode.json").write_text(
                json.dumps(
                    {
                        "bench2drive_ground_truth": True,
                        "metrics": {
                            "bench2drive_ground_truth": True,
                            "collisions": 1,
                            "offroad": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            finalized = build_episode(trace, config)
            self.assertTrue(finalized.metadata["event_ground_truth"])
            self.assertEqual(float(finalized.collision.sum()), 1.0)

    def test_collision_event_truncates_at_impact_timestamp(self):
        config = small_config()
        config.reward.collision_terminal = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            rows = []
            for index in range(8):
                action = {"steer": 0.0, "throttle": 0.5, "brake": 0.0}
                rows.append(
                    {
                        "collector_time": float(index),
                        "route_id": "148",
                        "seed": "202",
                        "town": "Town10HD",
                        "observation": {
                            name: 0.0 for name in DREAMER_OBSERVATION_FEATURES
                        },
                        "status": {
                            "mode": "simlingo_native",
                            "variant": "simlingo_native_report_collect",
                            "timestamp": float(index),
                            "base_action": action,
                            "final_action": action,
                            "state_vector": [float(index), 0.0, 2.0]
                            + [0.0] * 25,
                        },
                    }
                )
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (root / "collision_events.jsonl").write_text(
                json.dumps({"timestamp": 4.2, "event": "collision"}) + "\n",
                encoding="utf-8",
            )
            episode = build_episode(trace, config)
            self.assertIsNotNone(episode)
            self.assertEqual(episode.transitions, 5)
            np.testing.assert_array_equal(
                episode.collision, np.asarray([0, 0, 0, 0, 1], np.float32)
            )

    def test_transition_risk_uses_post_action_world_state(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            rows = []
            for index, clearance in enumerate((80.0, 1.0, 80.0)):
                action = {"steer": 0.0, "throttle": 0.4, "brake": 0.0}
                rows.append(
                    {
                        "collector_time": float(index),
                        "route_id": "148",
                        "seed": "303",
                        "town": "Town10HD",
                        "observation": {
                            name: 0.0 for name in DREAMER_OBSERVATION_FEATURES
                        },
                        "status": {
                            "mode": "simlingo_native",
                            "variant": "simlingo_native_report_collect",
                            "timestamp": float(index),
                            "base_action": action,
                            "final_action": action,
                            "front_vehicle_clearance_m": clearance,
                            "current_oncoming_ttc_s": 99.0,
                            "left_oncoming_ttc_s": 99.0,
                            "right_oncoming_ttc_s": 99.0,
                            "left_ttc_s": 99.0,
                            "right_ttc_s": 99.0,
                            "state_vector": [float(index), 0.0, 2.0]
                            + [0.0] * 25,
                        },
                    }
                )
            trace.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            episode = build_episode(trace, config)
            self.assertIsNotNone(episode)
            self.assertGreater(episode.risk[0], 0.9)
            self.assertLess(episode.risk[1], 0.1)

    def test_action_sensitivity_diagnostic_is_finite(self):
        config = small_config()
        episodes = []
        for seed in ("1", "2"):
            length = 6
            observations = np.zeros((length + 1, 32), np.float32)
            observations[:, 15] = 0.1
            episodes.append(
                DreamerEpisode(
                    key=seed,
                    seed=seed,
                    metadata={"route_id": "148", "town": "Town10HD"},
                    observations=observations,
                    actions=np.zeros((length, 3), np.float32),
                    alpha=np.zeros(length, np.float32),
                    progress=np.zeros(length, np.float32),
                    risk=np.zeros(length, np.float32),
                    continuation=np.ones(length, np.float32),
                    collision=np.zeros(length, np.float32),
                    offroad=np.zeros(length, np.float32),
                    reward=np.zeros(length, np.float32),
                    value=np.zeros(length, np.float32),
                )
            )
        loader = loader_for(episodes, config, False)
        metrics = action_sensitivity_metrics(
            CompactRSSM(config.rssm), loader, torch.device("cpu"), max_states=8
        )
        self.assertEqual(metrics["schema_version"], "report_action_sensitivity_v1")
        self.assertGreater(metrics["states"], 0)
        self.assertGreater(metrics["hazard_states"], 0)
        self.assertTrue(np.isfinite(metrics["mean_output_spread"]))

    def test_pairwise_loader_rejects_nonbinary_or_nonfinite_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairwise.jsonl"
            rows = [
                {
                    "seed": "1",
                    "candidate_a_features": [0, 0, 0, 0, 0],
                    "candidate_b_features": [1, 1, 1, 1, 1],
                    "label": 1,
                },
                {
                    "seed": "2",
                    "candidate_a_features": [0, 0, 0, 0, float("nan")],
                    "candidate_b_features": [1, 1, 1, 1, 1],
                    "label": 0,
                },
                {
                    "seed": "3",
                    "candidate_a_features": [0, 0, 0, 0, 0],
                    "candidate_b_features": [1, 1, 1, 1, 1],
                    "label": 0.5,
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            valid = load_pairwise_rows(path)
            self.assertEqual(len(valid), 1)
            self.assertEqual(valid[0]["seed"], "1")

    def test_shadow_trace_verifier_proves_native_control_invariance(self):
        observation = {name: 0.0 for name in DREAMER_OBSERVATION_FEATURES}
        rows = []
        for index in range(3):
            rows.append(
                {
                    "map": "Town12",
                    "route": "route.xml",
                    "scenario": "CrossingBicycleFlow",
                    "seed": "101",
                    "ablation": "D",
                    "shadow": True,
                    "applied": False,
                    "alpha": 0.0,
                    "native_action": [0.1, 0.4, 0.0],
                    "final_action": [0.1, 0.4, 0.0],
                    "observation": observation,
                    "native_predicted_progress": 0.1,
                    "native_predicted_risk": 0.2,
                    "selected_predicted_progress": 0.1,
                    "selected_predicted_risk": 0.2,
                    "selected_predicted_continuation": 0.9,
                    "selected_predicted_value": 0.3,
                    "inference_latency_ms": 2.0,
                    "candidate_kinds": ["native", "slow"],
                    "candidate_features": [[0.1] * 5, [0.2] * 5],
                    "candidate_utilities": [0.0, -0.1],
                    "selected_index": index % 2,
                }
            )
        summary = verify_shadow_rows(rows, minimum_ticks=3)
        self.assertTrue(summary["native_final_bit_exact"])
        self.assertEqual(summary["nonnative_proposal_ticks"], 1)
        broken = [dict(row) for row in rows]
        broken[1]["final_action"] = [0.2, 0.4, 0.0]
        with self.assertRaisesRegex(ValueError, "differs from native"):
            verify_shadow_rows(broken, minimum_ticks=3)

    def test_promotion_requires_complete_finite_prediction_heads(self):
        per_seed = {
            seed: {key: 1.0 for key in PREDICTION_LOSS_KEYS}
            for seed in ("101", "202")
        }
        payload = {
            "test_seed_count": 2,
            "aggregate_prediction_losses": {
                key: 1.0 for key in PREDICTION_LOSS_KEYS
            },
            "dispersion": {
                key: {
                    "mean_across_seeds": 1.0,
                    "std_across_seeds": 0.0,
                    "seed_count": 2,
                }
                for key in PREDICTION_LOSS_KEYS
            },
            "per_seed": per_seed,
            "action_sensitivity": {
                "schema_version": "report_action_sensitivity_v1",
                "states": 256,
                "hazard_states": 64,
                "mean_transition_spread": 0.1,
                "mean_output_spread": 0.02,
                "mean_progress_spread": 0.02,
                "mean_risk_spread": 0.01,
                "mean_collision_spread": 0.01,
                "collapsed_output_fraction_1e-4": 0.0,
                "hazard_brake_risk_advantage": 0.01,
                "hazard_brake_collision_advantage": 0.01,
                "hazard_throttle_progress_advantage": 0.01,
            },
        }
        validate_prediction_metrics(payload)
        payload["aggregate_prediction_losses"]["risk"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            validate_prediction_metrics(payload)

    def test_promotion_rejects_inverted_braking_semantics(self):
        per_seed = {
            seed: {key: 1.0 for key in PREDICTION_LOSS_KEYS}
            for seed in ("101", "202")
        }
        payload = {
            "test_seed_count": 2,
            "aggregate_prediction_losses": {
                key: 1.0 for key in PREDICTION_LOSS_KEYS
            },
            "dispersion": {
                key: {
                    "mean_across_seeds": 1.0,
                    "std_across_seeds": 0.0,
                    "seed_count": 2,
                }
                for key in PREDICTION_LOSS_KEYS
            },
            "per_seed": per_seed,
            "action_sensitivity": {
                "schema_version": "report_action_sensitivity_v1",
                "states": 256,
                "hazard_states": 64,
                "mean_transition_spread": 0.1,
                "mean_output_spread": 0.02,
                "mean_progress_spread": 0.02,
                "mean_risk_spread": 0.01,
                "mean_collision_spread": 0.01,
                "collapsed_output_fraction_1e-4": 0.0,
                "hazard_brake_risk_advantage": -0.001,
                "hazard_brake_collision_advantage": 0.01,
                "hazard_throttle_progress_advantage": 0.01,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "brake_risk_advantage"):
            validate_prediction_metrics(payload)


if __name__ == "__main__":
    unittest.main()
