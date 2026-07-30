"""Step 5: full Dreamer-PPO training with world-model dreaming.

The policy is trained with PPO. Once the world model is ready
(``wm_trainer.is_ready()``), action selection switches from direct policy
sampling to *dreaming*: sample ``k`` candidate actions, imagine each one
step ahead with the world model, and execute the best-scored candidate.
Before the world model is ready, the gate falls back to ``policy.act``.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from configs.config import Config
from env.carla_env import CarlaEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from training.rollout_buffer import RolloutBuffer
from training.ppo import update_ppo
from training.logger import Logger
from training.wm_buffer import WorldModelBuffer
from training.world_model_trainer import WorldModelTrainer
from training.evaluator import Evaluator
from training.safe_dream_accumulator import EpisodeSafetyAccumulator
from utils.progress_monitor import ProgressMonitor
from rewards.vru_reward import ROUTE_PROGRESS


# ---------------------------------------------------------------------- #
# Dreaming action selection
# ---------------------------------------------------------------------- #
@torch.no_grad()
def select_action_with_dreaming(policy, world_model, state, k=5, w_progress=1.0,
                                w_risk=2.0, w_value=0.5, device="cpu"):
    """Sample k candidate actions, score each with the world model, pick best.

    Returns ``(best_action, best_raw_action, best_log_prob, best_value, scores)``:
      - ``best_action``     bounded action tensor, shape (action_dim,)
      - ``best_raw_action`` pre-squash sample (kept so the PPO ratio stays
                            consistent between rollout collection and update)
      - ``best_log_prob``   scalar tensor
      - ``best_value``      scalar tensor
      - ``scores``          list of the k candidate scores (for logging)
    """
    if not torch.is_tensor(state):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device)

    candidates = []
    for _ in range(k):
        action, log_prob, value, raw_action = policy.act(state.unsqueeze(0))
        action = action.squeeze(0)
        raw_action = raw_action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        value = value.squeeze(0)

        # World model imagines the immediate future of this candidate.
        next_state_hat, risk_hat, progress_hat = world_model(
            state.unsqueeze(0), action.unsqueeze(0)
        )
        risk = risk_hat.item()
        progress = progress_hat.item()
        val = value.item()

        score = w_progress * progress - w_risk * risk + w_value * val
        candidates.append((score, action, raw_action, log_prob, value))

    best = max(candidates, key=lambda c: c[0])
    scores = [c[0] for c in candidates]
    return best[1], best[2], best[3], best[4], scores


def _select_action(policy, world_model, wm_trainer, state, config, device):
    """Gate: dream if the world model is ready, otherwise sample the policy.

    Returns (action_np, raw_action_np, log_prob, value, scores, dreaming_used).
    """
    if wm_trainer.is_ready():
        action, raw_action, log_prob, value, scores = select_action_with_dreaming(
            policy, world_model, state, k=config.dream_k,
            w_progress=config.w_progress, w_risk=config.w_risk,
            w_value=config.w_value, device=device,
        )
        dreaming_used = True
    else:
        action, log_prob, value, raw_action = policy.act(state.unsqueeze(0))
        action = action.squeeze(0)
        raw_action = raw_action.squeeze(0)
        log_prob = log_prob.squeeze(0)
        value = value.squeeze(0)
        scores = None
        dreaming_used = False

    return (action.cpu().numpy(), raw_action.cpu().numpy(),
            float(log_prob.item()), float(value.item()), scores, dreaming_used)


def _save_checkpoint(path, episode, policy, world_model, optimizer_pi, eval_return):
    torch.save({
        "episode": episode,
        "policy": policy.state_dict(),
        "world_model": world_model.state_dict(),
        "optimizer_pi": optimizer_pi.state_dict(),
        "eval_return": eval_return,
    }, path)


def _load_world_model_checkpoint(path, world_model, device):
    """Warm-start the world model from a guarded or RL checkpoint."""
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model") or ckpt.get("world_model")
    if state_dict is None:
        raise KeyError(f"{path} must contain either 'model' or 'world_model'.")
    world_model.load_state_dict(state_dict, strict=True)
    print(f"Loaded initial world model from {path}", flush=True)


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #
def train(config=None, mock=False, num_episodes=None, verbose=True, log_dir=None,
          device="cpu", eval_interval=50, ckpt_dir="checkpoints",
          log_name="training_log.csv", init_world_model_path=None):
    config = config or Config()
    if num_episodes is not None:
        config.num_episodes = num_episodes
    device = torch.device(device)

    env = CarlaEnv(mock=mock, config=config)
    policy = ActorCritic(config.state_dim, config.action_dim, config.hidden).to(device)
    world_model = WorldModel(config.state_dim, config.action_dim,
                             config.wm_hidden).to(device)
    _load_world_model_checkpoint(init_world_model_path, world_model, device)
    optimizer_pi = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)

    buffer = RolloutBuffer(
        config.rollout_size, config.state_dim, config.action_dim,
        gamma=config.gamma, lam=config.lam,
    )
    wm_buffer = WorldModelBuffer(capacity=50_000, state_dim=config.state_dim,
                                 action_dim=config.action_dim)
    wm_trainer = WorldModelTrainer(world_model, config)
    evaluator = Evaluator(env, policy, world_model, config, device=device)

    logger = Logger(log_dir, log_name) if log_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    best_eval_return = -float("inf")

    # Flags a route_progress signal that never advances (env/reward wiring bug)
    # rather than letting it masquerade as poor policy performance.
    progress_monitor = ProgressMonitor()
    global_step = 0

    history = []
    try:
        for episode in range(config.num_episodes):
            buffer.clear()
            obs = env.reset()
            ep_return = 0.0
            ep_stats = defaultdict(float)
            ep_collisions = 0
            ep_lane_departures = 0
            route_completion = float(obs[ROUTE_PROGRESS])
            safe_acc = EpisodeSafetyAccumulator(config)
            unsafe_thr = getattr(config, "unsafe_score_threshold", -5.0)
            done = False
            info = {}

            # ---- collect a rollout (may span several env episodes) ---- #
            while not buffer.is_full():
                state = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_np, raw_action_np, log_prob, value, scores, dreaming_used = \
                    _select_action(policy, world_model, wm_trainer, state,
                                   config, device)
                if dreaming_used:
                    ep_stats["dreaming_steps"] += 1

                next_obs, reward, done, info = env.step(action_np)

                # SAFE-DREAM telemetry. Greedy dreaming's k imagined candidates
                # are this variant's "counterfactuals": their scores drive
                # coverage / diversity / rejection metrics (risk_hat isn't
                # surfaced here, so risk-prediction fields stay 0 for dreamer).
                safe_acc.record_step(info, next_obs)
                if dreaming_used and scores:
                    safe_acc.record_plan_meta({
                        "n_candidates_evaluated": len(scores),
                        "n_candidates_rejected_unsafe":
                            sum(1 for sc in scores if sc < unsafe_thr),
                        "candidate_scores": list(scores),
                        "candidate_risk_predictions": [],
                        "chosen_risk_prediction": None,
                    })
                risk_target = float(info.get("vru_risk", 0.0))
                progress_target = float(info.get("progress", 0.0))

                buffer.store(obs, raw_action_np, reward, done, value, log_prob,
                             next_obs, risk_target, progress_target)
                wm_buffer.add(obs, action_np, next_obs, risk_target, progress_target)

                for comp_key, comp_val in info.get("reward_components", {}).items():
                    ep_stats[f"r_{comp_key}"] += comp_val

                ep_return += reward
                ep_collisions += int(info.get("vru_collisions", 0))
                ep_lane_departures += int(info.get("lane_departures", 0))
                route_completion = float(next_obs[ROUTE_PROGRESS])
                progress_monitor.record(global_step, route_completion)
                global_step += 1
                obs = next_obs

                if done:
                    buffer.finish_path(last_value=0.0)
                    obs = env.reset()

            # Bootstrap the value of the final (unfinished) state.
            if buffer.path_start < buffer.ptr:
                with torch.no_grad():
                    _, _, last_v = policy.forward(
                        torch.as_tensor(obs, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    )
                buffer.finish_path(last_value=float(last_v.item()))

            # ---- PPO update ---- #
            batch = buffer.get()
            n = batch["states"].shape[0]
            ppo_stats = {}
            for _ in range(config.update_epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, config.batch_size):
                    mb_idx = idx[start:start + config.batch_size]
                    mb = {key: val[mb_idx] for key, val in batch.items()}
                    ppo_stats = update_ppo(
                        policy, optimizer_pi, mb, clip_eps=config.clip_eps,
                        ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                        max_grad_norm=config.max_grad_norm,
                    )

            # ---- world-model update (from its own replay) ---- #
            if wm_buffer.is_ready(min_size=1000):
                wm_stats = wm_trainer.update(wm_buffer.sample(config.wm_batch_size))
                wm_eval = wm_trainer.evaluate(wm_buffer.sample(256))
            else:
                wm_stats = {"loss_wm": 0.0, "loss_state": 0.0,
                            "loss_risk": 0.0, "loss_progress": 0.0}
                wm_eval = {"state_pred_error": 0.0, "risk_pred_error": 0.0,
                           "progress_pred_error": 0.0}

            if verbose and episode % 10 == 0 and wm_buffer.is_ready(1000):
                print(f"  WM eval | state_err={wm_eval['state_pred_error']:.4f} "
                      f"risk_err={wm_eval['risk_pred_error']:.4f} "
                      f"progress_err={wm_eval['progress_pred_error']:.4f}")

            dreaming_active = wm_trainer.is_ready()

            # ---- periodic greedy evaluation + best-model checkpoint ---- #
            eval_stats = {}
            if eval_interval and episode % eval_interval == 0:
                eval_stats = evaluator.evaluate(n_episodes=5)
                if verbose:
                    print("\n" + "=" * 50)
                    print(f"EVAL episode {episode}")
                    print(f"  return:          {eval_stats['eval_return']:.2f}")
                    print(f"  vru_collisions:  {eval_stats['eval_vru_collisions']:.2f}")
                    print(f"  near_misses:     {eval_stats['eval_near_misses']:.2f}")
                    print(f"  route_completion:{eval_stats['eval_route_completion']:.2f}")
                    print("=" * 50 + "\n")
                if ckpt_dir and eval_stats["eval_return"] > best_eval_return:
                    best_eval_return = eval_stats["eval_return"]
                    _save_checkpoint(os.path.join(ckpt_dir, "best_model.pt"),
                                     episode, policy, world_model, optimizer_pi,
                                     best_eval_return)
                    if verbose:
                        print(f"  💾 New best model saved "
                              f"(return={best_eval_return:.2f})")

            # ---- periodic checkpoint regardless of performance ---- #
            if ckpt_dir and episode % 100 == 0:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"episode_{episode:04d}.pt"),
                    episode, policy, world_model, optimizer_pi,
                    eval_stats.get("eval_return", best_eval_return),
                )

            if verbose:
                print(f"Episode {episode:04d} | return={ep_return:.2f} | "
                      f"dreaming={'ON' if dreaming_active else 'OFF'} | "
                      f"ppo_loss={ppo_stats.get('loss', 0.0):.4f} | "
                      f"wm_loss={wm_stats['loss_wm']:.4f} | "
                      f"vru_collisions={ep_collisions}")

            record = {
                "episode": episode,
                "return": ep_return,
                "r_progress": ep_stats.get("r_progress", 0.0),
                "r_vru": ep_stats.get("r_vru_risk", 0.0),
                "r_collision": ep_stats.get("r_collision", 0.0),
                "r_comfort": ep_stats.get("r_comfort", 0.0),
                "r_rules": ep_stats.get("r_rules", 0.0),
                "ppo_loss": ppo_stats.get("loss", 0.0),
                "vf_loss": ppo_stats.get("value_loss", 0.0),
                "entropy": ppo_stats.get("entropy", 0.0),
                "loss_wm": wm_stats.get("loss_wm", 0.0),
                "wm_state_err": wm_eval.get("state_pred_error", 0.0),
                "wm_risk_err": wm_eval.get("risk_pred_error", 0.0),
                "dreaming_active": int(dreaming_active),
                "dreaming_steps": ep_stats.get("dreaming_steps", 0.0),
                "vru_collisions": ep_collisions,
                "lane_departures": ep_lane_departures,
                "route_completion": route_completion,
            }
            record.update(safe_acc.summarize())
            record.update(eval_stats)
            history.append(record)
            if logger is not None:
                logger.log(episode, record)

            if episode % 50 == 0:
                stall_check = progress_monitor.check_stalled()
                if stall_check["stalled"]:
                    print(f"WARNING: {stall_check['reason']}")
    finally:
        env.close()
        if logger is not None:
            logger.close()

    return history


# ---------------------------------------------------------------------- #
# S-DBS training (Serendipitous Diverse Beam Search extension)
# ---------------------------------------------------------------------- #
def _occupancy_pairs(config):
    """(rel_x, rel_y) index pairs of every agent block for the chosen layout.

    Vehicle "ahead" and the two VRUs are always present; the rear/side/nearest
    vehicle blocks only exist with Tier-1 state. Derived from the layout so the
    occupancy target stays valid at 28/48/55 dims.
    """
    from rewards.vru_reward import (
        resolve_layout, VEHICLE_AHEAD_DIST, VEHICLE_BEHIND_DIST,
        VEHICLE_LEFT_DIST, VEHICLE_RIGHT_DIST, VEHICLE_NEAREST_DIST,
    )
    lay = resolve_layout(config)
    bases = [VEHICLE_AHEAD_DIST]
    if lay.tier1:
        bases += [VEHICLE_BEHIND_DIST, VEHICLE_LEFT_DIST,
                  VEHICLE_RIGHT_DIST, VEHICLE_NEAREST_DIST]
    bases += [lay.vru0, lay.vru1]
    return [(b + 3, b + 4) for b in bases]     # rel_x, rel_y within each block


def _build_occupancy_targets(states, pairs, grid=16, extent=20.0):
    """Rasterize vehicle + VRU relative positions into a BEV occupancy grid
    target (B, grid, grid) for the scene-reconstruction head.
    """
    states = states.detach().cpu().numpy()
    b = states.shape[0]
    target = np.zeros((b, grid, grid), dtype=np.float32)
    for i in range(b):
        for xi, yi in pairs:
            rx, ry = float(states[i, xi]), float(states[i, yi])
            gx = int((rx + extent) / (2 * extent) * grid)
            gy = int((ry + extent) / (2 * extent) * grid)
            if 0 <= gx < grid and 0 <= gy < grid:
                target[i, gx, gy] = 1.0
    return torch.as_tensor(target)


def train_sdbs(config=None, mock=False, num_episodes=None, verbose=True,
               log_dir=None, device="cpu", ckpt_dir="checkpoints",
               log_name="training_log.csv", init_world_model_path=None):
    """Dreamer-PPO with S-DBS planning, risk-aware curriculum, and grounding.

    Replaces greedy one-step dreaming with multi-step diverse beam search,
    trains a world-model ensemble + auxiliary heads alongside PPO, and draws
    scenarios from a prioritized curriculum. Falls back to the base ``Config``
    behaviour for everything not specific to S-DBS.
    """
    from configs.sdbs_config import SDBSConfig
    from planning.sdbs_planner import SDBSPlanner
    from planning.curriculum import RiskAwareCurriculum
    from models.auxiliary_heads import (
        SceneReconstructionHead, RiskDensityHead, WorldModelEnsemble,
    )
    from utils.safety_tracker import SafetyTracker

    config = config or SDBSConfig()
    if num_episodes is not None:
        config.num_episodes = num_episodes
    device = torch.device(device)

    # The env now emits the full state for the enabled tiers (base 28, +20 with
    # Tier-1, +7 with Tier-2), so policy/world-model/buffers all use
    # config.state_dim directly — no separate augmentation step.
    obs_dim = config.state_dim
    occ_pairs = _occupancy_pairs(config)

    env = CarlaEnv(mock=mock, config=config)
    policy = ActorCritic(obs_dim, config.action_dim, config.hidden).to(device)
    world_model = WorldModel(obs_dim, config.action_dim,
                             config.wm_hidden).to(device)
    _load_world_model_checkpoint(init_world_model_path, world_model, device)
    optimizer_pi = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)

    buffer = RolloutBuffer(
        config.rollout_size, obs_dim, config.action_dim,
        gamma=config.gamma, lam=config.lam,
    )
    wm_buffer = WorldModelBuffer(capacity=50_000, state_dim=obs_dim,
                                 action_dim=config.action_dim)
    wm_trainer = WorldModelTrainer(world_model, config)

    sdbs_planner = SDBSPlanner(policy, world_model, policy, config, device=device)
    # Tier-2: enter defensive mode up front (e.g. unknown map / manual request).
    if config.defensive_mode:
        sdbs_planner.defensive_controller.activate_defensive_mode()
    curriculum = RiskAwareCurriculum(config)
    scenario_replayer = curriculum.replayer

    world_model_ensemble = WorldModelEnsemble(
        type(world_model), obs_dim, config.action_dim,
        n_models=config.world_model_ensemble_size, device=device,
    )
    wm_ensemble_opt = torch.optim.Adam(world_model_ensemble.parameters(),
                                       lr=config.lr_wm)
    recon_head = SceneReconstructionHead(config.hidden).to(device)
    risk_density_head = RiskDensityHead(config.hidden).to(device)
    aux_opt = torch.optim.Adam(
        list(recon_head.parameters()) + list(risk_density_head.parameters()),
        lr=config.lr_wm,
    )

    logger = Logger(log_dir, log_name) if log_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    best_eval_return = -float("inf")
    aux_ready = max(32, min(config.wm_batch_size, 64))
    safety_tracker = SafetyTracker()

    history = []
    try:
        for episode in range(config.num_episodes):
            scenario_ids = curriculum.get_active_scenarios()
            if len(scenario_ids) > config.max_scenarios_per_episode:
                # Prioritized replay decides which scenarios to revisit.
                scenario_ids = scenario_replayer.sample_scenarios(
                    config.max_scenarios_per_episode
                )

            safety_tracker.reset()
            safe_acc = EpisodeSafetyAccumulator(config)
            ep_return = 0.0
            ep_collisions = 0
            ep_planning_steps = 0
            ep_completion = 0.0
            ep_latency_sum = 0.0
            ep_reward_components = defaultdict(float)
            last_meta = {"latency_ms": 0.0, "difficulty": 0.0}
            ppo_stats, wm_stats = {}, {"loss_wm": 0.0}
            aux_loss_val = 0.0
            wm_ens_loss_val = 0.0

            for scenario_id in scenario_ids:
                buffer.clear()
                info = {}
                obs = env.reset_to_scenario(scenario_id)

                while not buffer.is_full():
                    state = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    best_action, plan, meta = sdbs_planner.plan(state, info)
                    last_meta = meta
                    ep_planning_steps += 1
                    action_np = best_action.cpu().numpy()

                    safety_tracker.step()
                    ep_latency_sum += meta.get("latency_ms", 0.0)
                    # SAFE-DREAM: fold this decision's beam diagnostics in.
                    safe_acc.record_plan_meta(meta)
                    # Lane-change accounting from the planner's mandate decision.
                    if meta.get("mandate") == "stay_in_lane":
                        safety_tracker.record_lane_change(
                            is_safe=False, blocked_by_mandate=True)
                        # A blocked unsafe lane change is a disengagement-equivalent.
                        safe_acc.record_disengagement()
                    elif abs(float(action_np[0])) > 0.3:
                        safety_tracker.record_lane_change(is_safe=True)

                    next_obs, reward, done, info = env.step(action_np)
                    safe_acc.record_step(info, next_obs)

                    # Tier-3: accumulate per-step safety metrics.
                    for d, ttc in zip(info.get("vru_distance_list", []),
                                      info.get("vru_ttc_list", [])):
                        safety_tracker.record_vru_observation(d, 0.0, ttc)
                    for d, ttc, dirn in zip(info.get("vehicle_distance_list", []),
                                            info.get("vehicle_ttc_list", []),
                                            info.get("vehicle_directions", [])):
                        safety_tracker.record_vehicle_observation(d, 0.0, ttc, dirn)
                    if info.get("collision_with_vehicle"):
                        safety_tracker.record_vehicle_collision()
                    if info.get("vru_collisions", 0):
                        safety_tracker.record_vru_collision()
                    for ck, cv in info.get("reward_components", {}).items():
                        ep_reward_components[ck] += cv

                    # Intrinsic reward from serendipitous discoveries.
                    if meta.get("serendipity_bonus_used"):
                        reward += config.eta_s * meta.get("serendipity_score", 0.0)

                    risk_target = float(info.get("vru_risk", 0.0))
                    progress_target = float(info.get("progress", 0.0))

                    # Train PPO on the policy proposal behind the first action,
                    # even when execution was overridden by a safety mandate.
                    buffer.store(obs, meta["first_raw_action"], reward, done,
                                 meta["first_value"], meta["first_log_prob"],
                                 next_obs, risk_target, progress_target)
                    wm_buffer.add(obs, action_np, next_obs,
                                  risk_target, progress_target)

                    ep_return += reward
                    ep_collisions += int(info.get("vru_collisions", 0))
                    ep_completion = float(next_obs[ROUTE_PROGRESS])
                    obs = next_obs

                    if done:
                        buffer.finish_path(last_value=0.0)
                        scenario_replayer.record_episode(
                            scenario_id=scenario_id,
                            n_collisions=int(info.get("vru_collisions", 0)),
                            n_near_misses=int(info.get("near_misses", 0)),
                            n_ttc_violations=int(info.get("ttc_violations", 0)),
                            progress_deficit=max(
                                0.0, config.goal_distance
                                - float(info.get("route_completion", ep_completion))
                            ),
                        )
                        curriculum.record_rollout(scenario_id, {
                            "vru_collisions": int(info.get("vru_collisions", 0)),
                            "near_misses": int(info.get("near_misses", 0)),
                            "route_completion": ep_completion,
                        })
                        safety_tracker.record_episode_completion(ep_completion)
                        obs = env.reset_to_scenario(scenario_id)

                if buffer.path_start < buffer.ptr:
                    with torch.no_grad():
                        _, _, last_v = policy.forward(
                            torch.as_tensor(obs, dtype=torch.float32,
                                            device=device).unsqueeze(0)
                        )
                    buffer.finish_path(last_value=float(last_v.item()))

                # ---- PPO update ---- #
                batch = buffer.get()
                n = batch["states"].shape[0]
                for _ in range(config.update_epochs):
                    idx = np.random.permutation(n)
                    for start in range(0, n, config.batch_size):
                        mb_idx = idx[start:start + config.batch_size]
                        mb = {key: val[mb_idx] for key, val in batch.items()}
                        ppo_stats = update_ppo(
                            policy, optimizer_pi, mb, clip_eps=config.clip_eps,
                            ent_coef=config.ent_coef, vf_coef=config.vf_coef,
                            max_grad_norm=config.max_grad_norm,
                        )

                # ---- world-model (single + ensemble) update ---- #
                if wm_buffer.is_ready(min_size=aux_ready):
                    wm_stats = wm_trainer.update(wm_buffer.sample(config.wm_batch_size))

                    wm_batch = wm_buffer.sample(config.wm_batch_size)
                    wm_ens_loss = world_model_ensemble.loss(wm_batch)
                    wm_ensemble_opt.zero_grad()
                    wm_ens_loss.backward()
                    wm_ensemble_opt.step()
                    wm_ens_loss_val = float(wm_ens_loss.item())

                    # ---- auxiliary grounding heads ---- #
                    feats = policy.trunk(wm_batch["states"].float().to(device)).detach()
                    recon_logits = recon_head(feats)
                    occ_target = _build_occupancy_targets(
                        wm_batch["states"], occ_pairs).to(device)
                    recon_loss = recon_head.loss(recon_logits, occ_target)

                    risk_pred = risk_density_head(feats)
                    risk_loss = risk_density_head.loss(
                        risk_pred, wm_batch["risk_targets"]
                    )

                    aux_loss = (config.lambda_recon * recon_loss
                                + config.lambda_risk_density * risk_loss)
                    aux_opt.zero_grad()
                    aux_loss.backward()
                    aux_opt.step()
                    aux_loss_val = float(aux_loss.item())

                # ---- curriculum advancement ---- #
                if curriculum.should_advance_stage():
                    curriculum.advance_to_next_stage()
                    if verbose:
                        print(f"CURRICULUM: Advanced to stage "
                              f"{curriculum.current_stage()}")

            defensive_on = sdbs_planner.defensive_controller.defensive_mode_active
            if verbose:
                print(f"Episode {episode:04d} | stage={curriculum.current_stage()} | "
                      f"state_dim={obs_dim} | "
                      f"defensive={'ON' if defensive_on else 'OFF'} | "
                      f"return={ep_return:.2f} | difficulty={last_meta['difficulty']:.2f} "
                      f"| planning_latency={last_meta['latency_ms']:.1f}ms | "
                      f"vru_collisions={ep_collisions}")

            safety_summary = safety_tracker.summarize()
            avg_latency = (ep_latency_sum / ep_planning_steps
                           if ep_planning_steps else 0.0)
            record = {
                "episode": episode,
                "return": ep_return,
                "ppo_loss": ppo_stats.get("loss", 0.0),
                "vf_loss": ppo_stats.get("value_loss", 0.0),
                "entropy": ppo_stats.get("entropy", 0.0),
                "loss_wm": wm_stats.get("loss_wm", 0.0),
                "loss_wm_ensemble": wm_ens_loss_val,
                "aux_loss": aux_loss_val,
                "dreaming_active": 1,
                "dreaming_steps": ep_planning_steps,
                "vru_collisions": ep_collisions,
                "route_completion": ep_completion,
                "stage": curriculum.current_stage(),
                "difficulty": last_meta.get("difficulty", 0.0),
                "state_dim": obs_dim,
                "defensive_mode": int(defensive_on),
                "defensive_mode_active": int(defensive_on),
                "planning_latency_ms": last_meta.get("latency_ms", 0.0),
                "avg_planning_latency_ms": avg_latency,
                # Tier-3 reward components + separated safety metrics.
                "r_progress": ep_reward_components.get("progress", 0.0),
                "r_vru": ep_reward_components.get("vru_risk", 0.0),
                "r_collision": ep_reward_components.get("collision", 0.0),
                "r_vehicle_collision": ep_reward_components.get("vehicle_collision", 0.0),
                "r_vehicle_proximity": ep_reward_components.get("vehicle_proximity", 0.0),
                "r_rear_risk": ep_reward_components.get("rear_risk", 0.0),
                "r_comfort": ep_reward_components.get("comfort", 0.0),
                "r_rules": ep_reward_components.get("rules", 0.0),
            }
            record.update(safety_summary)
            record.update(safe_acc.summarize())
            record["vru_collisions"] = max(ep_collisions,
                                           safety_summary["vru_collisions"])
            history.append(record)
            if logger is not None:
                logger.log(episode, record)

            if ckpt_dir and ep_return > best_eval_return:
                best_eval_return = ep_return
                _save_checkpoint(
                    os.path.join(ckpt_dir, "best_model.pt"),
                    episode, policy, world_model, optimizer_pi, ep_return,
                )
                if verbose:
                    print(f"  💾 New best S-DBS model saved "
                          f"(return={ep_return:.2f})")

            # ---- Tier-3: periodic interpretable summaries ---- #
            if verbose and episode % 50 == 0 and logger is not None:
                logger.create_summary_table()
                sdbs_planner.explainer.print_summary()

            if ckpt_dir and episode % 100 == 0:
                _save_checkpoint(
                    os.path.join(ckpt_dir, f"sdbs_episode_{episode:04d}.pt"),
                    episode, policy, world_model, optimizer_pi, ep_return,
                )
    finally:
        env.close()
        if logger is not None:
            logger.close()

    return history


def _print_variant_header(variant, cfg):
    """Announce the variant and the state dimension it will train/export at."""
    print(f"Variant: {variant} | state_dim={cfg.state_dim} "
          f"(tier1={cfg.enable_tier1_state}, tier2={cfg.enable_tier2_state}) | "
          f"default is the 28-dim v1 layout")


def main():
    # Ensure UTF-8 stdout so the 💾 marker renders on Windows consoles (cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Dreamer-PPO for CARLA")
    parser.add_argument("--mock", action="store_true",
                        help="run without CARLA installed")
    parser.add_argument("--sdbs", action="store_true",
                        help="use the S-DBS planner + risk-aware curriculum")
    parser.add_argument("--baseline", action="store_true",
                        help="run the PPO baseline (no world model / no dreaming)")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--town", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--rollout-size", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--log-name", default=None,
                        help="CSV filename under logs/ (default: per-variant name "
                             "so variants don't overwrite each other)")
    parser.add_argument("--init-world-model", default=None)
    args = parser.parse_args()

    if args.sdbs and args.baseline:
        parser.error("--sdbs and --baseline are mutually exclusive")

    # Each variant gets its own log file by default, so a full sweep
    # (baseline -> dreamer -> sdbs) leaves three CSVs
    # (logs/baseline.csv, logs/dreamer.csv, logs/sdbs.csv) for inspection.
    variant = "sdbs" if args.sdbs else "baseline" if args.baseline else "dreamer"
    log_name = args.log_name or f"{variant}.csv"

    if args.baseline:
        from training.ppo_baseline import train_baseline
        cfg = Config()
        if args.host is not None:
            cfg.host = args.host
        if args.port is not None:
            cfg.port = args.port
        if args.town is not None:
            cfg.town = args.town
        if args.max_episode_steps is not None:
            cfg.max_episode_steps = args.max_episode_steps
        if args.rollout_size is not None:
            cfg.rollout_size = args.rollout_size
        _print_variant_header(variant, cfg)
        train_baseline(cfg, mock=args.mock, num_episodes=args.episodes,
                       log_dir=args.log_dir, log_name=log_name)
    elif args.sdbs:
        from configs.sdbs_config import SDBSConfig
        cfg = SDBSConfig()
        if args.host is not None:
            cfg.host = args.host
        if args.port is not None:
            cfg.port = args.port
        if args.town is not None:
            cfg.town = args.town
        if args.max_episode_steps is not None:
            cfg.max_episode_steps = args.max_episode_steps
        if args.rollout_size is not None:
            cfg.rollout_size = args.rollout_size
        _print_variant_header(variant, cfg)
        train_sdbs(cfg, mock=args.mock, num_episodes=args.episodes,
                   device=args.device, log_dir=args.log_dir,
                   ckpt_dir=args.ckpt_dir, log_name=log_name,
                   init_world_model_path=args.init_world_model)
    else:
        cfg = Config()
        if args.host is not None:
            cfg.host = args.host
        if args.port is not None:
            cfg.port = args.port
        if args.town is not None:
            cfg.town = args.town
        if args.max_episode_steps is not None:
            cfg.max_episode_steps = args.max_episode_steps
        if args.rollout_size is not None:
            cfg.rollout_size = args.rollout_size
        _print_variant_header(variant, cfg)
        train(cfg, mock=args.mock, num_episodes=args.episodes,
              device=args.device, log_dir=args.log_dir,
              ckpt_dir=args.ckpt_dir, log_name=log_name,
              init_world_model_path=args.init_world_model,
              eval_interval=args.eval_interval)
    print(f"\nLog written to {args.log_dir}/{log_name}")


if __name__ == "__main__":
    main()
