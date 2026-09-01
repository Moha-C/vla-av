import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from external.simlingo.team_code.dreamer_guard import (
    ActorCritic,
    DreamerGuard,
    GuardConfig,
    WorldModel,
    _is_opposing_vehicle_heading,
    _scale_control_authority,
    rssm_authority_confidence,
)
from external.simlingo.team_code.dreamer_world_models import (
    POLICY_MODEL_TYPE,
    UTILITY_CONTEXT_OBSERVATION_INDICES,
    UTILITY_MODEL_TYPE,
    WORLD_MODEL_TYPE,
    PairwiseUtilityCalibrator,
    RSSMConfig,
    RSSMState,
    TemporalRSSMWorldModel,
    discounted_feature_pool,
    expand_actor_input_state_dict,
)
from scripts.dreamer_online_rl_update import (
    MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
    upgrade_policy_observation_checkpoint,
)
from scripts.finetune_dreamer_rssm_stationary_oncoming import (
    behavior_preservation_gate,
    blended_risk_head_state,
    stationary_risk_margin,
)
from scripts.simlingo_dashboard import (
    checkpoint_for_rssm_v2,
    dreamer_comparison_payload,
    dreamer_group_for_variant,
    payload_enabled,
)
from scripts.train_dreamer_rssm_v2 import (
    calibrated_arbitration,
    geometric_targets,
    rssm_quality_gate,
    world_model_loss,
)


class TemporalRSSMTest(unittest.TestCase):
    def test_rssm_probability_state_preserves_soft_categorical_information(self):
        config = RSSMConfig(
            observation_dim=4,
            action_dim=2,
            encoder_dim=8,
            hidden_dim=8,
            deter_dim=4,
            stoch_dim=2,
            classes=3,
            deterministic_state_mode="probabilities",
        )
        model = TemporalRSSMWorldModel(config)
        logits = torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]])
        state = model._sample(logits, deterministic=True).reshape(1, 2, 3)
        torch.testing.assert_close(
            state.sum(dim=-1), torch.ones(1, 2)
        )
        self.assertTrue(torch.all((state > 0.0) & (state < 1.0)))

    def test_rssm_sigma_shooting_explores_positive_longitudinal_action(self):
        guard = object.__new__(DreamerGuard)
        guard.is_temporal_world_model = True
        guard.rssm_actor_sigma_shooting = True
        guard.policy_action_semantics = (
            "simlingo_signed_longitudinal_target_with_learned_gate_v3"
        )
        guard.config = GuardConfig(checkpoint="unused", rl_action_space="absolute")
        guard.policy = ActorCritic(4, 4, 8)
        guard.policy.log_std.data.fill_(-1.8)
        base = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        raw = np.asarray([-0.4, -0.8, -0.8, 8.0], dtype=np.float32)
        actor_policy = guard._decode_policy_raw_action(raw)
        actor_control = guard._policy_to_control_action(base, actor_policy)
        actions, metadata = guard._rssm_policy_shooting_actions(base, {
            "action": actor_control,
            "raw_action": raw,
        })
        self.assertEqual(len(actions), len(metadata))
        self.assertGreater(len(actions), 4)
        self.assertTrue(any(float(action[1]) > 0.25 for action in actions))
        self.assertTrue(all(
            not (float(action[1]) > 0.0 and float(action[2]) > 0.0)
            for action in actions
        ))
        self.assertTrue(any(
            "longitudinal_positive" in row["kind"] for row in metadata
        ))

    def test_rssm_sigma_shooting_is_checkpoint_opt_in(self):
        guard = object.__new__(DreamerGuard)
        guard.is_temporal_world_model = True
        guard.rssm_actor_sigma_shooting = False
        guard.policy_action_semantics = (
            "simlingo_signed_longitudinal_target_with_learned_gate_v3"
        )
        guard.config = GuardConfig(checkpoint="unused", rl_action_space="absolute")
        guard.policy = ActorCritic(4, 4, 8)
        base = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        raw = np.asarray([-0.4, -0.8, -0.8, 8.0], dtype=np.float32)
        actor_policy = guard._decode_policy_raw_action(raw)
        actor_control = guard._policy_to_control_action(base, actor_policy)
        actions, metadata = guard._rssm_policy_shooting_actions(base, {
            "action": actor_control,
            "raw_action": raw,
        })
        self.assertEqual(len(actions), 2)
        self.assertEqual(
            [row["kind"] for row in metadata],
            ["rssm_simlingo_base", "rssm_actor_mean"],
        )

    def test_dashboard_keeps_known_good_rssm_candidate_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate_model.pt"
            candidate.touch()
            checkpoint, source = checkpoint_for_rssm_v2(root)
            self.assertEqual(checkpoint, candidate)
            self.assertIn("known_good", source)
            calibrated = root / "utility_calibrator_candidate_pre_ab.pt"
            calibrated.touch()
            checkpoint, source = checkpoint_for_rssm_v2(root)
            self.assertEqual(checkpoint, candidate)
            self.assertIn("known_good", source)
            checkpoint, source = checkpoint_for_rssm_v2(
                root, prefer_calibrated=True
            )
            self.assertEqual(checkpoint, calibrated)
            self.assertIn("calibrated_action_shooting", source)

    def test_dashboard_uses_calibrated_rssm_only_as_missing_candidate_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibrated = root / "utility_calibrator_candidate_pre_ab.pt"
            calibrated.touch()
            checkpoint, source = checkpoint_for_rssm_v2(root)
            self.assertEqual(checkpoint, calibrated)
            self.assertIn("fallback", source)

    def test_stationary_opposite_heading_remains_oncoming(self):
        self.assertTrue(_is_opposing_vehicle_heading(-0.99))
        self.assertFalse(_is_opposing_vehicle_heading(0.99))

    def test_stationary_opposing_vehicle_in_selected_lane_is_a_risk_target(self):
        state = [0.0] * 28
        next_state = [0.0] * 28
        status = {
            "state_vector": state,
            "base_action": {
                "steer": 0.0,
                "throttle": 0.0,
                "brake": 1.0,
                "intervention": 0.0,
            },
            "front_vehicle_m": 80.0,
            "nearest_walker_m": 80.0,
            "nearest_bike_m": 80.0,
            "left_clear_m": 80.0,
            "left_ttc_s": 99.0,
            "left_oncoming_m": 10.0,
            "left_oncoming_ttc_s": 99.0,
            "left_lane_available": True,
            "traffic_light": "none",
        }
        risk, _, _ = geometric_targets(
            status,
            {"state_vector": next_state},
            np.asarray([-0.4, 0.4, 0.0, 1.0], dtype=np.float32),
        )
        self.assertGreater(risk, 0.70)

    def test_stationary_risk_margin_is_continuous_without_a_veto(self):
        far = stationary_risk_margin(34.0)
        middle = stationary_risk_margin(20.0)
        near = stationary_risk_margin(5.0)
        self.assertLess(far, middle)
        self.assertLess(middle, near)
        self.assertLess(near, 0.02)

    def test_risk_head_blend_keeps_parent_exactly_at_zero(self):
        parent = {"weight": torch.tensor([1.0, 2.0])}
        calibrated = {"weight": torch.tensor([3.0, -2.0])}
        blended = blended_risk_head_state(parent, calibrated, 0.0)
        torch.testing.assert_close(blended["weight"], parent["weight"])
        halfway = blended_risk_head_state(parent, calibrated, 0.5)
        torch.testing.assert_close(
            halfway["weight"], torch.tensor([2.0, 0.0])
        )

    def test_behavior_gate_rejects_loss_of_useful_overtakes(self):
        common = {"risk_increase_selected": 0}
        passed, details = behavior_preservation_gate(
            {"selected_candidates": 20, **common},
            {"selected_candidates": 5, **common},
            {"selected_candidates": 30, **common},
            {"selected_candidates": 10, **common},
            minimum_preservation=0.70,
            maximum_blocked_selection_ratio=0.50,
        )
        self.assertFalse(passed)
        self.assertEqual(details["minimum_preserved_selected"], 21)

    def test_behavior_gate_accepts_safer_preserving_candidate(self):
        common = {"risk_increase_selected": 0}
        passed, details = behavior_preservation_gate(
            {"selected_candidates": 20, **common},
            {"selected_candidates": 8, **common},
            {"selected_candidates": 30, **common},
            {"selected_candidates": 24, **common},
            minimum_preservation=0.70,
            maximum_blocked_selection_ratio=0.50,
        )
        self.assertTrue(passed)
        self.assertEqual(details["maximum_blocked_selected"], 10)

    def test_arbitration_calibration_is_continuous_and_has_no_guard(self):
        calibration = calibrated_arbitration({
            "5": {"risk_mae": 0.14, "progress_mae_m": 0.12},
        })
        self.assertGreater(calibration["risk_curvature"], 2.0)
        self.assertGreater(calibration["action_penalty"], 0.18)
        self.assertEqual(calibration["candidate_commit_horizon"], 1)
        self.assertAlmostEqual(calibration["authority_temperature"], 0.40)
        self.assertEqual(
            calibration["actor_gate_role"],
            "upper_bound_scaled_by_model_confidence",
        )
        self.assertFalse(calibration["hard_thresholds"])

    def test_rssm_authority_grows_continuously_with_utility_margin(self):
        self.assertEqual(rssm_authority_confidence(0.0, 0.4), 0.0)
        small = rssm_authority_confidence(0.02, 0.4)
        medium = rssm_authority_confidence(0.20, 0.4)
        large = rssm_authority_confidence(1.60, 0.4)
        self.assertGreater(small, 0.0)
        self.assertLess(small, medium)
        self.assertLess(medium, large)
        self.assertGreater(large, 0.98)

    def test_authority_scaling_keeps_longitudinal_control_exclusive(self):
        base = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        proposed = np.asarray([0.8, 1.0, 0.0, 0.95], dtype=np.float32)
        blended = _scale_control_authority(base, proposed, 0.25, 0.2375)
        self.assertAlmostEqual(float(blended[0]), 0.2)
        self.assertEqual(float(blended[1]), 0.0)
        self.assertGreater(float(blended[2]), 0.0)
        self.assertFalse(float(blended[1]) > 0.0 and float(blended[2]) > 0.0)

    def test_fresh_pairwise_calibrator_is_behavior_preserving(self):
        torch.manual_seed(11)
        calibrator = PairwiseUtilityCalibrator(12, hidden_dim=8)
        base = torch.randn(5, 12)
        candidate = torch.randn(5, 12)
        residual = calibrator(
            base,
            candidate,
            torch.randn(5),
            torch.randn(5),
            torch.rand(5, 3),
        )
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_discounted_feature_pool_uses_continuation_prefix(self):
        features = torch.tensor([[[1.0], [3.0], [9.0]]])
        continuation = torch.tensor([[1.0, 1.0, 0.0]])
        pooled = discounted_feature_pool(features, continuation, 1.0)
        torch.testing.assert_close(pooled, torch.tensor([[2.0]]))

    def test_temporal_score_adds_learned_residual_without_changing_base(self):
        guard = object.__new__(DreamerGuard)
        guard.is_temporal_world_model = True
        guard.rssm_progress_weight = 1.0
        guard.rssm_risk_weight = 2.0
        guard.rssm_risk_curvature = 0.0
        guard.rssm_action_penalty = 0.0
        guard._candidate_meta = []
        actions = np.asarray([
            [0.0, 0.0, 1.0, 0.0],
            [-0.3, 0.6, 0.0, 0.8],
        ], dtype=np.float32)
        without = guard._scored_predictions(
            actions,
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([0.2, 0.2], dtype=np.float32),
        )
        with_residual = guard._scored_predictions(
            actions,
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([0.2, 0.2], dtype=np.float32),
            utility_residual_np=np.asarray([0.0, 0.4], dtype=np.float32),
        )
        self.assertEqual(with_residual[0]["score"], without[0]["score"])
        self.assertAlmostEqual(
            with_residual[1]["score"] - without[1]["score"], 0.4
        )
        self.assertEqual(with_residual[0]["learned_utility_residual"], 0.0)

    def test_quality_gate_rejects_bad_forced_safety_validation(self):
        family = {
            "ego": {"persistence_ratio": 1.0},
            "decision": {"observation_mae_normalized": 0.10},
        }
        validation = {
            "1": {
                "families": family,
                "idle_decision_noise_mae_normalized": 0.01,
            },
            "5": {
                "families": family,
                "risk_mae": 0.10,
                "event_brier": 0.02,
            },
            "15": {"risk_mae": 0.15},
        }
        passed, details = rssm_quality_gate(
            validation,
            {"5": {"risk_mae": 0.40, "event_brier": 0.02}},
        )
        self.assertFalse(passed)
        self.assertFalse(details["forced_validation_passed"])

    def test_dashboard_string_zero_is_disabled(self):
        self.assertFalse(payload_enabled("0"))
        self.assertFalse(payload_enabled(False))
        self.assertTrue(payload_enabled("1"))

    def test_dashboard_rssm_has_an_isolated_kpi_family(self):
        self.assertEqual(
            dreamer_group_for_variant("dreamer_ppo_rssm_v2"),
            "dreamer_rssm",
        )
        self.assertEqual(
            dreamer_group_for_variant("dreamer_ppo_rl_noguard"),
            "dreamer_ppo_rl",
        )
        self.assertEqual(
            dreamer_group_for_variant("dreamer_sdbs_rl_noguard"),
            "dreamer_sdbs_rl",
        )

    def test_dashboard_comparison_exposes_report_ablation_families(self):
        payload = dreamer_comparison_payload()
        expected = [
            "simlingo",
            "dreamer_ppo",
            "report_rssm_fixed",
            "report_rssm_learned",
            "report_rssm_pairwise",
        ]
        self.assertEqual(
            [column["id"] for column in payload["columns"]],
            expected,
        )
        self.assertEqual(
            [card["id"] for card in payload["cards"]],
            expected,
        )

    def test_sequence_shapes_and_balanced_kl_are_finite(self):
        torch.manual_seed(4)
        config = RSSMConfig(
            observation_dim=12,
            action_dim=4,
            encoder_dim=16,
            hidden_dim=24,
            deter_dim=10,
            stoch_dim=4,
            classes=5,
            event_dim=5,
        )
        model = TemporalRSSMWorldModel(config)
        output = model.observe_sequence(
            torch.randn(3, 8, 12),
            torch.randn(3, 7, 4),
            deterministic=True,
        )
        self.assertEqual(output["observation_delta"].shape, (3, 7, 12))
        self.assertEqual(output["event_logits"].shape, (3, 7, 5))
        self.assertEqual(output["posterior_features"].shape, (3, 8, config.feature_dim))
        kl = model.kl_loss(output["posterior_logits"], output["prior_logits"])
        self.assertTrue(torch.isfinite(kl["loss"]))

    def test_deterministic_posterior_is_reproducible(self):
        torch.manual_seed(5)
        model = TemporalRSSMWorldModel(RSSMConfig(observation_dim=9, action_dim=4))
        observation = torch.randn(2, 9)
        first = model.observe_initial(observation, deterministic=True)
        second = model.observe_initial(observation, deterministic=True)
        torch.testing.assert_close(first.deter, second.deter)
        torch.testing.assert_close(first.stoch, second.stoch)
        torch.testing.assert_close(first.logits, second.logits)

    def test_static_observation_slots_cannot_accumulate_rollout_noise(self):
        torch.manual_seed(6)
        model = TemporalRSSMWorldModel(RSSMConfig(observation_dim=49, action_dim=4))
        state = model.observe_initial(torch.randn(2, 49), deterministic=True)
        _, heads = model.imagine_step(state, torch.randn(2, 4), deterministic=True)
        static = [0, 1, 3, 5, 7, 9, 11, 12, 15, 17, 19, 20, 22, 24, 25, 27, 45, 46, 47, 48]
        torch.testing.assert_close(
            heads["observation_delta"][:, static],
            torch.zeros(2, len(static)),
        )

    def test_multistep_training_loss_is_finite_and_backpropagates(self):
        torch.manual_seed(7)
        config = RSSMConfig(
            observation_dim=12,
            action_dim=4,
            encoder_dim=16,
            hidden_dim=24,
            deter_dim=10,
            stoch_dim=4,
            classes=5,
        )
        model = TemporalRSSMWorldModel(config)
        batch = {
            "observations": torch.randn(3, 9, 12),
            "actions": torch.randn(3, 8, 4),
            "rewards": torch.randn(3, 8),
            "continuation": torch.ones(3, 8),
            "risks": torch.rand(3, 8),
            "progress": torch.rand(3, 8),
            "events": torch.randint(0, 2, (3, 8, 5)).float(),
        }
        loss, metrics = world_model_loss(
            model,
            batch,
            torch.device("cpu"),
            overshoot_horizon=4,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(metrics["overshooting"], 0.0)
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_actor_migration_is_exact_with_zero_latent(self):
        torch.manual_seed(8)
        old_actor = ActorCritic(46, 4, 32)
        old_world_model = WorldModel(28, 4, 32)
        checkpoint = {
            "policy": old_actor.state_dict(),
            "world_model": old_world_model.state_dict(),
            "policy_state_mean": np.zeros(46, dtype=np.float32),
            "policy_state_std": np.ones(46, dtype=np.float32),
            "policy_input_semantics": "world_state_plus_simlingo_temporal_context_v4",
        }
        checkpoint, _ = upgrade_policy_observation_checkpoint(checkpoint)
        config = RSSMConfig(observation_dim=49, action_dim=4)
        migrated_state = expand_actor_input_state_dict(
            checkpoint["policy"], 49 + config.feature_dim
        )
        migrated_actor = ActorCritic(49 + config.feature_dim, 4, 32)
        migrated_actor.load_state_dict(migrated_state)

        old_input = torch.randn(6, 46)
        upgraded_input = torch.zeros(6, 49)
        upgraded_input[:, :42] = old_input[:, :42]
        upgraded_input[:, 45:49] = old_input[:, 42:46]
        temporal_input = torch.zeros(6, 49 + config.feature_dim)
        temporal_input[:, :49] = upgraded_input
        old_output, _, old_value = old_actor(old_input)
        new_output, _, new_value = migrated_actor(temporal_input)
        torch.testing.assert_close(old_output, new_output)
        torch.testing.assert_close(old_value, new_value)

    def test_runtime_loads_isolated_temporal_checkpoint(self):
        torch.manual_seed(9)
        config = RSSMConfig(
            observation_dim=49,
            action_dim=4,
            encoder_dim=24,
            hidden_dim=32,
            deter_dim=16,
            stoch_dim=4,
            classes=4,
        )
        world_model = TemporalRSSMWorldModel(config)
        actor = ActorCritic(49 + config.feature_dim, 4, 32)
        checkpoint = {
            "world_model_type": WORLD_MODEL_TYPE,
            "world_model_config": config.to_dict(),
            "policy_model_type": POLICY_MODEL_TYPE,
            "base_world_state_dim": 28,
            "policy_observation_dim": 49,
            "world_model": world_model.state_dict(),
            "policy": actor.state_dict(),
            "policy_input_semantics": MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
            "policy_action_semantics": "simlingo_signed_longitudinal_target_with_learned_gate_v3",
            "policy_state_mean": np.zeros(49, dtype=np.float32),
            "policy_state_std": np.ones(49, dtype=np.float32),
            "action_mean": np.zeros(4, dtype=np.float32),
            "action_std": np.ones(4, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rssm.pt"
            torch.save(checkpoint, path)
            guard = DreamerGuard(
                GuardConfig(
                    checkpoint=str(path),
                    mode="rl_noguard",
                    device="cpu",
                )
            )
        self.assertTrue(guard.is_temporal_world_model)
        self.assertEqual(guard.policy_observation_dim, 49)
        self.assertEqual(guard.policy_state_dim, 49 + config.feature_dim)
        state = np.zeros(28, dtype=np.float32)
        base = np.asarray([0.0, 0.4, 0.0, 0.0], dtype=np.float32)
        context = {"left_lane_available": 1.0, "right_lane_available": 1.0}
        actor_input, observation = guard._state_for_policy(state, base, context)
        self.assertEqual(observation.shape, (49,))
        self.assertEqual(actor_input.shape, (1, 49 + config.feature_dim))
        scores = guard.predict(state, [base, np.asarray([0.1, 0.3, 0.0, 0.5])])
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(math_value == math_value for row in scores for math_value in (row["risk"], row["progress"])))

    def test_runtime_loads_pairwise_utility_calibrator(self):
        torch.manual_seed(12)
        config = RSSMConfig(
            observation_dim=49,
            action_dim=4,
            encoder_dim=24,
            hidden_dim=32,
            deter_dim=16,
            stoch_dim=4,
            classes=4,
        )
        world_model = TemporalRSSMWorldModel(config)
        actor = ActorCritic(49 + config.feature_dim, 4, 32)
        calibrator = PairwiseUtilityCalibrator(
            config.feature_dim,
            observation_dim=len(UTILITY_CONTEXT_OBSERVATION_INDICES),
            hidden_dim=8,
            output_scale=1.25,
        )
        checkpoint = {
            "world_model_type": WORLD_MODEL_TYPE,
            "world_model_config": config.to_dict(),
            "policy_model_type": POLICY_MODEL_TYPE,
            "utility_model_type": UTILITY_MODEL_TYPE,
            "utility_calibrator": calibrator.state_dict(),
            "base_world_state_dim": 28,
            "policy_observation_dim": 49,
            "world_model": world_model.state_dict(),
            "policy": actor.state_dict(),
            "policy_input_semantics": MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
            "policy_action_semantics": "simlingo_signed_longitudinal_target_with_learned_gate_v3",
            "policy_state_mean": np.zeros(49, dtype=np.float32),
            "policy_state_std": np.ones(49, dtype=np.float32),
            "action_mean": np.zeros(4, dtype=np.float32),
            "action_std": np.ones(4, dtype=np.float32),
            "rssm_v2": {
                "utility_calibrator": {
                    "model_type": UTILITY_MODEL_TYPE,
                    "observation_dim": len(UTILITY_CONTEXT_OBSERVATION_INDICES),
                    "observation_indices": list(
                        UTILITY_CONTEXT_OBSERVATION_INDICES
                    ),
                    "hidden_dim": 8,
                    "output_scale": 1.25,
                    "blend": 0.65,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rssm_calibrated.pt"
            torch.save(checkpoint, path)
            guard = DreamerGuard(GuardConfig(
                checkpoint=str(path), mode="rl_noguard", device="cpu"
            ))
            self.assertIsNotNone(guard.rssm_utility_calibrator)
            self.assertAlmostEqual(guard.rssm_utility_blend, 0.65)
            state = np.zeros(28, dtype=np.float32)
            base = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            context = {"left_lane_available": 1.0, "right_lane_available": 1.0}
            guard._state_for_policy(state, base, context)
            scores = guard.predict(
                state,
                [base, np.asarray([-0.2, 0.5, 0.0, 0.8], dtype=np.float32)],
            )
        self.assertEqual(scores[0]["learned_utility_residual"], 0.0)
        self.assertTrue(all(
            np.isfinite(row["learned_utility_residual"]) for row in scores
        ))

    def test_temporal_score_penalizes_risk_increase_more_at_high_risk(self):
        guard = object.__new__(DreamerGuard)
        guard.is_temporal_world_model = True
        guard.rssm_progress_weight = 1.0
        guard.rssm_risk_weight = 2.0
        guard.rssm_risk_curvature = 3.0
        guard.rssm_action_penalty = 0.25
        guard._candidate_meta = []
        actions = np.asarray([
            [0.0, 0.4, 0.0, 0.0],
            [0.4, 0.8, 0.0, 0.8],
        ], dtype=np.float32)
        scored = guard._scored_predictions(
            actions,
            np.asarray([0.80, 0.81], dtype=np.float32),
            np.asarray([0.50, 0.60], dtype=np.float32),
        )
        self.assertGreater(scored[0]["score"], scored[1]["score"])

    def test_temporal_rollout_returns_to_simlingo_after_one_candidate_step(self):
        class RecordingModel:
            def __init__(self):
                self.actions = []

            def imagine_step(self, state, action, deterministic=True):
                self.actions.append(action.detach().cpu().clone())
                count = action.shape[0]
                return state, {
                    "risk_logit": torch.zeros(count),
                    "progress_symlog": torch.zeros(count),
                    "continuation_logit": torch.full((count,), 8.0),
                }

        guard = object.__new__(DreamerGuard)
        guard.is_temporal_world_model = True
        guard.device = torch.device("cpu")
        guard.rssm_state = RSSMState(
            deter=torch.zeros(1, 2),
            stoch=torch.zeros(1, 2),
            logits=torch.zeros(1, 1, 2),
        )
        guard.action_mean = torch.zeros(4)
        guard.action_std = torch.ones(4)
        guard.rssm_planning_horizon = 3
        guard.rssm_candidate_commit_horizon = 1
        guard.rssm_planning_discount = 0.95
        guard.rssm_progress_weight = 1.0
        guard.rssm_risk_weight = 2.0
        guard.rssm_risk_curvature = 2.0
        guard.rssm_action_penalty = 0.2
        guard._candidate_meta = []
        guard.model = RecordingModel()
        base = np.asarray([0.0, 0.4, 0.0, 0.0], dtype=np.float32)
        candidate = np.asarray([0.6, 0.8, 0.0, 0.9], dtype=np.float32)
        guard.predict(np.zeros(28, dtype=np.float32), [base, candidate])
        np.testing.assert_allclose(guard.model.actions[0][1].numpy(), candidate)
        np.testing.assert_allclose(guard.model.actions[1][1].numpy(), base)
        np.testing.assert_allclose(guard.model.actions[2][1].numpy(), base)

    def test_closed_loop_rollout_keeps_reference_simlingo_and_replans_candidates(self):
        class RecordingModel:
            def __init__(self):
                self.actions = []

            @staticmethod
            def feature(state):
                return torch.cat([state.deter, state.stoch], dim=-1)

            def imagine_step(self, state, action, deterministic=True):
                self.actions.append(action.detach().cpu().clone())
                count = action.shape[0]
                return state, {
                    "observation_delta": torch.zeros(count, 49),
                    "risk_logit": torch.zeros(count),
                    "progress_symlog": torch.zeros(count),
                    "continuation_logit": torch.full((count,), 8.0),
                }

        guard = object.__new__(DreamerGuard)
        guard.is_temporal_world_model = True
        guard.device = torch.device("cpu")
        guard.rssm_closed_loop_actor_rollout = True
        guard.rssm_state = RSSMState(
            deter=torch.zeros(1, 2),
            stoch=torch.zeros(1, 2),
            logits=torch.zeros(1, 1, 2),
        )
        guard.rssm_current_observation_normalized = torch.zeros(1, 49)
        guard.rssm_current_observation_normalized[0, 28:31] = torch.tensor(
            [0.1, 0.2, 0.0]
        )
        guard.action_mean = torch.zeros(4)
        guard.action_std = torch.ones(4)
        guard.policy_state_mean = torch.zeros(49)
        guard.policy_state_std = torch.ones(49)
        guard.world_observation_mean = torch.zeros(49)
        guard.world_observation_std = torch.ones(49)
        guard.rssm_planning_horizon = 3
        guard.rssm_candidate_commit_horizon = 1
        guard.rssm_planning_discount = 0.95
        guard.rssm_progress_weight = 1.0
        guard.rssm_risk_weight = 2.0
        guard.rssm_risk_curvature = 2.0
        guard.rssm_action_penalty = 0.2
        guard.rssm_utility_calibrator = None
        guard.policy_action_semantics = (
            "simlingo_signed_longitudinal_target_with_learned_gate_v3"
        )
        guard.config = GuardConfig(checkpoint="unused", rl_action_space="absolute")
        guard._candidate_meta = []
        guard.model = RecordingModel()
        guard.policy = ActorCritic(53, 4, 8)
        for parameter in guard.policy.parameters():
            parameter.data.zero_()
        guard.policy.actor_mean.bias.data.copy_(
            torch.tensor([-0.7, 1.0, -1.0, 8.0])
        )
        base = np.asarray([0.1, 0.2, 0.0, 0.0], dtype=np.float32)
        candidate = np.asarray([-0.2, 0.5, 0.0, 0.8], dtype=np.float32)

        guard.predict(np.zeros(28, dtype=np.float32), [base, candidate])

        np.testing.assert_allclose(guard.model.actions[1][0].numpy(), base)
        np.testing.assert_allclose(guard.model.actions[2][0].numpy(), base)
        self.assertLess(float(guard.model.actions[1][1, 0]), -0.5)
        self.assertGreater(float(guard.model.actions[1][1, 1]), 0.8)

    def test_temporal_arbiter_can_keep_simlingo_without_a_hard_guard(self):
        torch.manual_seed(10)
        config = RSSMConfig(
            observation_dim=49,
            action_dim=4,
            encoder_dim=24,
            hidden_dim=32,
            deter_dim=16,
            stoch_dim=4,
            classes=4,
        )
        checkpoint = {
            "world_model_type": WORLD_MODEL_TYPE,
            "world_model_config": config.to_dict(),
            "policy_model_type": POLICY_MODEL_TYPE,
            "base_world_state_dim": 28,
            "policy_observation_dim": 49,
            "world_model": TemporalRSSMWorldModel(config).state_dict(),
            "policy": ActorCritic(49 + config.feature_dim, 4, 32).state_dict(),
            "policy_input_semantics": MAP_INVARIANT_CURRENT_ONCOMING_POLICY_INPUT_SEMANTICS,
            "policy_action_semantics": "simlingo_signed_longitudinal_target_with_learned_gate_v3",
            "policy_state_mean": np.zeros(49, dtype=np.float32),
            "policy_state_std": np.ones(49, dtype=np.float32),
            "world_observation_mean": np.zeros(49, dtype=np.float32),
            "world_observation_std": np.ones(49, dtype=np.float32),
            "action_mean": np.zeros(4, dtype=np.float32),
            "action_std": np.ones(4, dtype=np.float32),
        }

        class Control:
            def __init__(self, steer=0.0, throttle=0.0, brake=0.0):
                self.steer = steer
                self.throttle = throttle
                self.brake = brake

        class Agent:
            step = 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rssm.pt"
            torch.save(checkpoint, path)
            guard = DreamerGuard(GuardConfig(
                checkpoint=str(path), mode="rl_noguard", device="cpu"
            ))
            guard.build_state = lambda agent, tick: (
                np.zeros(28, dtype=np.float32),
                {"front_vehicle_m": 80.0, "traffic_light": "none"},
            )
            proposal = np.asarray([0.4, 0.8, 0.0, 0.9], dtype=np.float32)
            guard.rl_policy_action = lambda state, base, context: {
                "action": proposal.copy(),
                "policy_action": proposal.copy(),
                "raw_action": proposal.copy(),
                "intervention_strength": 0.9,
                "log_prob": 0.0,
                "value": 0.0,
                "deterministic": True,
                "policy_observation": np.zeros(49, dtype=np.float32),
                "previous_policy_action": np.zeros(4, dtype=np.float32),
            }
            guard.predict = lambda state, actions: [
                {"candidate_index": 0, "action": actions[0], "risk": 0.1, "progress": 1.0, "score": 0.7},
                {"candidate_index": 1, "action": actions[1], "risk": 0.9, "progress": 1.1, "score": -2.0},
            ]
            original = Control(steer=0.05, throttle=0.3, brake=0.0)
            chosen, info = guard.maybe_override(Agent(), {}, original)

        self.assertIs(chosen, original)
        self.assertEqual(info["chosen_kind"], "rssm_simlingo_base")
        self.assertEqual(info["candidate_index"], 0)
        self.assertEqual(info["dreamer_weight"], 0.0)
        np.testing.assert_allclose(
            guard.previous_rssm_action,
            np.asarray([0.05, 0.3, 0.0, 0.0], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
