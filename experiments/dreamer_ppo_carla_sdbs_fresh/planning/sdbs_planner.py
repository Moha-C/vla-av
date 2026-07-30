"""Multi-step dreaming with Serendipitous Diverse Beam Search (S-DBS).

Replaces the greedy one-step dreaming of ``training/dreamer_ppo.py`` with a
budget-aware, diverse, multi-step lookahead:

  * scene difficulty (VRU count, TTC urgency, occlusion, risk density,
    world-model disagreement) scales the search width B, horizon H, and
    number of diverse groups G;
  * a hard safety layer (``evaluate_mandated_safety``) can clamp execution to
    a stop/yield regardless of what the search prefers;
  * group 1 exploits (pure imagined return); groups 2..G are pushed apart by a
    serendipity bonus (novelty + surprise + gain) and a diversity penalty so
    the agent considers non-obvious safe maneuvers around occluded VRUs.

The single ``WorldModel`` used here returns ``(next_state, risk, progress)``;
imagined per-step reward is synthesized as ``w_progress*progress -
w_risk*risk`` to match the original greedy dreaming score.
"""
import math
import time

import numpy as np
import torch

from rewards.vru_reward import EGO_SPEED, resolve_layout
from env.carla_env import (
    VEHICLE_AHEAD_DIST, VEHICLE_LEFT_DIST, VEHICLE_RIGHT_DIST,
    validate_state_vector,
)
from planning.sdbs_core import (
    Plan, BeamState, compute_conflict_cells, jaccard_diversity,
)
from planning.defensive_driving import DefensiveDrivingController
from planning.lane_change_explainer import LaneChangeExplainer

# State indices not re-exported by the reward module.
TRAFFIC_LIGHT_STATE = 10
DIST_TO_LIGHT = 11
LANE_CHANGE_STEER_THRESHOLD = 0.3   # |steering| above this = a lane-change attempt
LANE_CHANGE_STEER_CLAMP = 0.2       # clamp steering to +-this when staying in lane


class SDBSPlanner:
    def __init__(self, policy, world_model, critic, config, device="cpu"):
        self.policy = policy
        self.world_model = world_model
        self.critic = critic if critic is not None else policy
        self.config = config
        self.layout = resolve_layout(config)
        self.device = torch.device(device)
        self.beam = None        # most recent BeamState (handy for inspection/tests)
        self.defensive_controller = DefensiveDrivingController(config)
        self.explainer = LaneChangeExplainer()

    def _infer_action_reason(self, action, state):
        """Heuristic explanation for why the planner chose this action."""
        steering, throttle, brake = float(action[0]), float(action[1]), float(action[2])
        vehicle_ahead_dist = float(state[VEHICLE_AHEAD_DIST])
        if abs(steering) > 0.3:
            return "avoid_front_vehicle" if vehicle_ahead_dist < 10 else "reach_goal"
        if brake > 0.5:
            return "safety_brake"
        if throttle > 0.5:
            return "accelerate_progress"
        return "maintain_course"

    # ------------------------------------------------------------------ #
    # Defensive-mode action filtering
    # ------------------------------------------------------------------ #
    def filter_risky_actions(self, candidate_actions, state, info):
        """Drop risky candidates when defensive mode is active.

        Returns the safe subset; if every candidate is risky, returns the first
        one so planning always has something to fall back on.
        """
        if not self.defensive_controller.defensive_mode_active:
            return list(candidate_actions)
        safe = [a for a in candidate_actions
                if not self.defensive_controller.is_risky_action(a, state, info)]
        if not safe:
            return [candidate_actions[0]]
        return safe

    # ------------------------------------------------------------------ #
    # Scene difficulty -> search budget
    # ------------------------------------------------------------------ #
    def estimate_scene_difficulty(self, state, info):
        """Combine danger cues into a scalar difficulty in [0, 1] via sigmoid."""
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        cfg = self.config
        info = info or {}

        vru_indices = self.layout.vru_indices
        n_vru = info.get("n_vru")
        if n_vru is None:
            n_vru = sum(1 for idx in vru_indices if 0 < float(s[idx]) < 40.0)

        ego_speed = float(s[EGO_SPEED])
        ttcs = [float(s[idx]) / (ego_speed + 1e-6)
                for idx in vru_indices if float(s[idx]) > 0]
        min_ttc = min(ttcs) if ttcs else cfg.tau_ttc * 5.0
        ttc_deficit = max(0.0, cfg.tau_ttc - min_ttc)

        occlusion = float(info.get("occlusion_flag", info.get("occlusion", 0.0)))
        risk_density = float(info.get("risk_density", info.get("vru_risk", 0.0)))
        disagreement = float(info.get("world_model_disagreement", 0.0))

        z = (cfg.w_vru_count * n_vru
             + cfg.w_ttc_deficit * ttc_deficit
             + cfg.w_occlusion * occlusion
             + cfg.w_risk_density * risk_density
             + cfg.w_uncertainty * disagreement)
        # Bias so a quiet scene maps low and a couple of close VRUs maps high.
        return float(1.0 / (1.0 + math.exp(-(z - 1.0))))

    def get_search_params(self, difficulty):
        """Scale (beam width B, horizon H, groups G) by difficulty in [0, 1].

        When ``config.sdbs_force_fixed_params`` is set, difficulty-based scaling
        is bypassed and the fixed (beam_width, horizon, groups) from config are
        returned. This lets an integration start at horizon=1, groups=1 —
        behaviourally equivalent to greedy one-step dreaming — and increase the
        horizon gradually to isolate exactly where degradation starts.
        """
        cfg = self.config
        if getattr(cfg, "sdbs_force_fixed_params", False):
            return (cfg.sdbs_fixed_beam_width,
                    cfg.sdbs_fixed_horizon,
                    cfg.sdbs_fixed_groups)
        d = float(np.clip(difficulty, 0.0, 1.0))
        B = int(round(cfg.beam_width_min + d * (cfg.beam_width_max - cfg.beam_width_min)))
        H = int(round(cfg.horizon_min + d * (cfg.horizon_max - cfg.horizon_min)))
        G = int(round(cfg.num_groups_min + d * (cfg.num_groups_max - cfg.num_groups_min)))
        B = max(cfg.beam_width_min, min(cfg.beam_width_max, B))
        H = max(cfg.horizon_min, min(cfg.horizon_max, H))
        G = max(cfg.num_groups_min, min(cfg.num_groups_max, G))
        G = min(G, B)           # never more groups than total beam width
        return B, H, G

    # ------------------------------------------------------------------ #
    # Hard safety layer
    # ------------------------------------------------------------------ #
    def evaluate_mandated_safety(self, state, info):
        """Check hard constraints; return a mandated control clamp if any.

        Returns ``{'mandate': 'stop'/'yield'/None, 'clamped_controls': action}``.
        Action layout: [steering, throttle, brake, stop_continue].
        """
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        cfg = self.config
        info = info or {}

        # [steer=0, throttle=0, brake=1, stop_continue=0] -> full hard stop.
        stop = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        yield_ctrl = np.array([0.0, 0.0, 0.7, 0.0], dtype=np.float32)

        ego_speed = float(s[EGO_SPEED])
        ttcs = [float(s[idx]) / (ego_speed + 1e-6)
                for idx in self.layout.vru_indices if float(s[idx]) > 0]
        min_ttc = min(ttcs) if ttcs else float("inf")
        if min_ttc < cfg.tau_safe:
            return {"mandate": "stop", "clamped_controls": stop, "min_ttc": min_ttc}

        light_red = (float(s[TRAFFIC_LIGHT_STATE]) == 0.0
                     and float(s[DIST_TO_LIGHT]) < 10.0)
        if info.get("red_light_violation") or info.get("stop_sign_violation") or light_red:
            return {"mandate": "stop", "clamped_controls": stop}

        if info.get("right_of_way_violation"):
            return {"mandate": "yield", "clamped_controls": yield_ctrl}

        # ---- Lane-change blind-spot safety (Tier 1) ------------------- #
        # A lane change is requested when |steering| exceeds the threshold;
        # steering < 0 turns left, > 0 turns right. If a vehicle occupies the
        # target side within the (speed-scaled) clearance, block the change by
        # clamping steering to keep the car in its lane. Side-vehicle awareness
        # only exists with Tier-1 state, so the check is skipped otherwise.
        requested_steer = float(info.get("requested_action_steering", 0.0))
        if self.layout.tier1 and abs(requested_steer) > LANE_CHANGE_STEER_THRESHOLD:
            if requested_steer < 0:
                target = s[VEHICLE_LEFT_DIST:VEHICLE_LEFT_DIST + 5]
                direction = "left"
            else:
                target = s[VEHICLE_RIGHT_DIST:VEHICLE_RIGHT_DIST + 5]
                direction = "right"
            target_dist, target_speed = float(target[0]), float(target[1])
            speed_buffer = 0.5 * abs(target_speed - ego_speed)
            min_safe = cfg.min_lane_change_clearance + speed_buffer
            if target_dist < min_safe and target_dist < 100.0:
                return {
                    "mandate": "stay_in_lane",
                    "clamped_controls": None,   # plan() clamps steering in place
                    "direction": direction,
                    "reason": (f"unsafe {direction} lane change "
                               f"(vehicle at {target_dist:.1f}m)"),
                }

        return {"mandate": None, "clamped_controls": None}

    # ------------------------------------------------------------------ #
    # World-model imagination
    # ------------------------------------------------------------------ #
    def _dream_one(self, state_t, action):
        """Imagine one step. Returns (next_state_np, imagined_reward_float)."""
        if not torch.is_tensor(action):
            action = torch.as_tensor(np.asarray(action, dtype=np.float32),
                                     device=self.device)
        st = state_t.reshape(1, -1).float().to(self.device)
        ac = action.reshape(1, -1).float().to(self.device)
        out = self.world_model(st, ac)
        next_state, risk, progress = out[0], out[1], out[2]
        r = (self.config.w_progress * float(progress.reshape(-1)[0])
             - self.config.w_risk * float(risk.reshape(-1)[0]))
        return next_state.reshape(-1).cpu().numpy(), float(r)

    def _predict_risk(self, state_np, action):
        """World-model risk_hat for one (state, action) — SAFE-DREAM diagnostics.

        Additive helper used only to populate per-candidate risk predictions in
        plan()'s metadata; it does not affect scoring or selection.
        """
        st = torch.as_tensor(np.asarray(state_np, dtype=np.float32),
                             device=self.device).reshape(1, -1)
        ac = torch.as_tensor(np.asarray(action, dtype=np.float32),
                             device=self.device).reshape(1, -1)
        out = self.world_model(st, ac)
        return float(out[1].reshape(-1)[0])

    def _dream_forward(self, state, actions_seq, world_model=None):
        """Unroll the world model for a fixed action sequence.

        Returns ``(final_state, rewards, imagined_trajectory)`` where the
        trajectory includes the starting state followed by every imagined state.
        """
        world_model = world_model or self.world_model
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32),
                                  device=self.device)
        traj = [state_t.reshape(-1).cpu().numpy()]
        rewards = []
        cur = state_t
        for a in actions_seq:
            ns, r = self._dream_one(cur, a)
            rewards.append(r)
            traj.append(ns)
            cur = torch.as_tensor(ns, device=self.device)
        return traj[-1], rewards, traj

    def _value(self, state_np):
        st = torch.as_tensor(np.asarray(state_np, dtype=np.float32),
                             device=self.device)
        _, _, value = self.critic.forward(st)
        return float(value.reshape(-1)[0])

    def _g_value(self, plan):
        """Discounted imagined return + bootstrapped terminal critic value."""
        g = 0.0
        for t, r in enumerate(plan.imagined_rewards):
            g += (self.config.gamma ** t) * r
        h = len(plan.imagined_rewards)
        if plan.imagined_states:
            g += (self.config.gamma ** h) * self._value(plan.imagined_states[-1])
        return g

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score_plan(self, plan, group_id, all_groups, beam):
        """Composite score: exploitation for group 0, serendipity for the rest."""
        plan.g_value = self._g_value(plan)
        g_value = plan.g_value
        cfg = self.config

        if group_id == 0:
            plan.serendipity_value = 0.0
            return g_value

        my_cells = beam.conflict_cells.get(plan)
        if my_cells is None:
            my_cells = compute_conflict_cells(plan.imagined_states)
            beam.conflict_cells[plan] = my_cells

        # Novelty: 1 - max overlap with any plan from the other groups.
        max_overlap = 0.0
        all_g_values = []
        diversity_penalty = 0.0
        for og_idx, og in enumerate(all_groups):
            for other in og:
                all_g_values.append(other.g_value)
                oc = beam.conflict_cells.get(other)
                if oc is None:
                    continue
                overlap = jaccard_diversity(my_cells, oc)
                if og_idx != group_id:
                    max_overlap = max(max_overlap, overlap)
                if og_idx < group_id:
                    diversity_penalty += overlap
        novelty = 1.0 - max_overlap

        # Surprise: negative log-likelihood of the plan's first action.
        surprise = -float(plan.first_log_prob)

        # Gain: how far above the beam's mean imagined return this plan sits.
        mean_g = float(np.mean(all_g_values)) if all_g_values else g_value
        gain = g_value - mean_g

        serendipity = (cfg.alpha_novelty * novelty
                       + cfg.beta_surprise * surprise
                       + cfg.gamma_gain * gain)
        plan.serendipity_value = serendipity

        return (g_value
                + cfg.eta_serendipity * serendipity
                - cfg.lambda_g * diversity_penalty)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def plan(self, state, info=None, compute_budget=None):
        """Run S-DBS. Returns (best_action_tensor, best_plan, metadata)."""
        validate_state_vector(state, self.policy.state_dim)
        t0 = time.perf_counter()
        info = info or {}
        if compute_budget is None:
            compute_budget = self.config.compute_budget

        state_np = np.asarray(state, dtype=np.float32).reshape(-1) \
            if not torch.is_tensor(state) else state.detach().cpu().numpy().reshape(-1)

        difficulty = self.estimate_scene_difficulty(state_np, info)
        B, H, G = self.get_search_params(difficulty)

        # The policy's intended steering (squashed mean) drives the lane-change
        # blind-spot check, unless the caller already supplied one.
        if "requested_action_steering" not in info:
            mean, _std, _value = self.policy.forward(
                torch.as_tensor(state_np, device=self.device))
            info = dict(info)
            info["requested_action_steering"] = float(
                torch.tanh(mean.reshape(-1)[0]).item())
        mandate = self.evaluate_mandated_safety(state_np, info)

        beam = BeamState(depth=0, num_groups=G)
        for g in range(G):
            beam.add_plan(
                Plan(actions=[], imagined_states=[state_np.copy()],
                     imagined_rewards=[], group_id=g),
                g,
            )

        b_per_group = max(1, B // G)
        budget = 0
        best_searched = None

        # SAFE-DREAM per-decision diagnostics (additive; no effect on search).
        unsafe_thr = getattr(self.config, "unsafe_score_threshold", -5.0)
        n_candidates_evaluated = 0
        n_candidates_rejected_unsafe = 0

        for depth in range(H):
            new_groups = []
            for g_idx, group in enumerate(beam.groups):
                expanded = []
                for parent in group:
                    last_state = torch.as_tensor(
                        parent.imagined_states[-1], device=self.device
                    )
                    for _ in range(b_per_group):
                        if budget >= compute_budget:
                            break
                        action, log_prob, value, raw = self.policy.act(
                            last_state.unsqueeze(0)
                        )
                        action = action.squeeze(0)
                        ns, r = self._dream_one(last_state, action)
                        budget += 1

                        child = Plan(
                            actions=parent.actions + [action.cpu().numpy()],
                            imagined_states=parent.imagined_states + [ns],
                            imagined_rewards=parent.imagined_rewards + [r],
                            group_id=g_idx,
                        )
                        if parent.actions:        # inherit the root proposal
                            child.first_raw_action = parent.first_raw_action
                            child.first_log_prob = parent.first_log_prob
                            child.first_value = parent.first_value
                        else:                     # this child holds the first action
                            child.first_raw_action = raw.squeeze(0).cpu().numpy()
                            child.first_log_prob = float(log_prob.item())
                            child.first_value = float(value.item())

                        beam.conflict_cells[child] = compute_conflict_cells(
                            child.imagined_states
                        )
                        child.score = self._score_plan(
                            child, g_idx, beam.groups, beam
                        )
                        # SAFE-DREAM: tally every candidate and flag unsafe ones.
                        n_candidates_evaluated += 1
                        if child.score < unsafe_thr:
                            n_candidates_rejected_unsafe += 1
                        expanded.append(child)
                expanded.sort(key=lambda p: p.score, reverse=True)
                new_groups.append(expanded[:b_per_group])

            beam.groups = new_groups
            beam.depth = depth + 1
            beam.get_top_per_group(b_per_group)

            for grp in beam.groups:
                for p in grp:
                    if best_searched is None or p.score > best_searched.score:
                        best_searched = p
            beam.set_incumbent(best_searched)
            if budget >= compute_budget:
                break

        self.beam = beam

        # Fallback if the search produced nothing (e.g. zero budget).
        if best_searched is None or not best_searched.actions:
            action, log_prob, value, raw = self.policy.act(
                torch.as_tensor(state_np, device=self.device).unsqueeze(0)
            )
            best_searched = Plan(
                actions=[action.squeeze(0).cpu().numpy()],
                imagined_states=[state_np.copy()],
                imagined_rewards=[], group_id=0,
            )
            best_searched.first_raw_action = raw.squeeze(0).cpu().numpy()
            best_searched.first_log_prob = float(log_prob.item())
            best_searched.first_value = float(value.item())

        # Execution action: a safety mandate overrides the searched preference.
        if mandate["mandate"] in ("stop", "yield"):
            exec_action_np = np.asarray(mandate["clamped_controls"], dtype=np.float32)
            best_searched.maneuver = mandate["mandate"]
        elif mandate["mandate"] == "stay_in_lane":
            # Keep the planned action but clamp steering so no lane change happens.
            exec_action_np = np.asarray(best_searched.actions[0], dtype=np.float32).copy()
            exec_action_np[0] = float(np.clip(
                exec_action_np[0], -LANE_CHANGE_STEER_CLAMP, LANE_CHANGE_STEER_CLAMP))
            best_searched.maneuver = "stay_in_lane"
        else:
            exec_action_np = np.asarray(best_searched.actions[0], dtype=np.float32)

        # Defensive mode: soften a risky chosen action (gentle steer, ease off).
        if (self.defensive_controller.defensive_mode_active
                and self.defensive_controller.is_risky_action(
                    exec_action_np, state_np, info)):
            exec_action_np = np.asarray(exec_action_np, dtype=np.float32).copy()
            exec_action_np[0] = float(np.clip(exec_action_np[0], -0.2, 0.2))
            exec_action_np[1] = float(min(exec_action_np[1], 0.3))
            best_searched.maneuver = "defensive"

        best_action = torch.as_tensor(exec_action_np, device=self.device)

        # Tier-3: record lane-change decisions (executed or mandate-blocked).
        requested_steer = float(info.get("requested_action_steering",
                                         exec_action_np[0]))
        blocked = mandate["mandate"] == "stay_in_lane"
        if abs(requested_steer) > 0.3 or blocked:
            direction = (mandate.get("direction") if blocked
                         else ("left" if requested_steer < 0 else "right"))
            reason = ("blocked_unsafe_lane_change" if blocked
                      else self._infer_action_reason(exec_action_np, state_np))
            self.explainer.record_decision(
                timestep=int(info.get("timestep", 0)),
                action=exec_action_np,
                state=state_np,
                info={**info, "scene_difficulty": difficulty,
                      "defensive_mode": self.defensive_controller.defensive_mode_active},
                mandate=(mandate if blocked else None),
                is_safe=(not blocked),
                reason=reason,
                direction=direction,
            )

        serendipity_used = bool(
            best_searched.group_id > 0
            and mandate["mandate"] is None
            and best_searched.serendipity_value != 0.0
        )

        # ---- SAFE-DREAM per-decision diagnostics (additive) ------------- #
        # Final per-group top candidates (before selecting the single best):
        # their scores drive diversity/entropy and their WM risk_hat drives
        # risk-prediction metrics. Risk is predicted from the root state for
        # each candidate's first action so it is comparable to chosen_risk.
        final_candidates = [p for grp in beam.groups for p in grp if p.actions]
        candidate_scores = [float(p.score) for p in final_candidates]
        candidate_risk_predictions = [
            self._predict_risk(state_np, p.actions[0]) for p in final_candidates
        ]
        chosen_risk_prediction = self._predict_risk(state_np, exec_action_np)

        metadata = {
            "lookahead": H,
            "beam_width": B,
            "groups": G,
            "difficulty": difficulty,
            "conflict_cells_best": sorted(
                beam.conflict_cells.get(best_searched, set())
            ),
            "serendipity_bonus_used": serendipity_used,
            "serendipity_score": float(best_searched.serendipity_value),
            "defensive_mode": self.defensive_controller.defensive_mode_active,
            "planning_latency_ms": (time.perf_counter() - t0) * 1000.0,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "mandate": mandate["mandate"],
            # PPO-consistency triple (the policy proposal behind the first action)
            "first_raw_action": best_searched.first_raw_action,
            "first_log_prob": best_searched.first_log_prob,
            "first_value": best_searched.first_value,
            # SAFE-DREAM beam diagnostics (additive keys only).
            "n_candidates_evaluated": int(n_candidates_evaluated),
            "n_candidates_rejected_unsafe": int(n_candidates_rejected_unsafe),
            "candidate_scores": candidate_scores,
            "candidate_risk_predictions": candidate_risk_predictions,
            "chosen_risk_prediction": chosen_risk_prediction,
            "prediction_timestamp": int(info.get("timestep", 0)),
        }
        return best_action, best_searched, metadata
