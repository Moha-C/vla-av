"""Per-episode safety-event tracker.

Accumulates VRU safety (primary), vehicle safety (secondary), and lane-change
outcomes during an episode, then ``summarize()`` returns a flat dict for the
Logger. One instance is reset at the start of every episode.

Thresholds: a VRU near-miss is TTC < 2.5 s; a vehicle near-miss is TTC < 3.0 s;
a rear incident is a rear vehicle with TTC < 2.0 s.
"""

VRU_NEAR_MISS_TTC = 2.5
VEHICLE_NEAR_MISS_TTC = 3.0
REAR_INCIDENT_TTC = 2.0


class SafetyTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        # VRU safety
        self.vru_collisions = 0
        self.vru_near_misses = []          # (time, distance, ttc)
        self.min_ttc_vru = float("inf")
        self.sum_distance_to_vru = 0.0
        self.vru_observation_count = 0

        # Vehicle safety
        self.vehicle_collisions = 0
        self.vehicle_near_misses = []      # (time, distance, ttc, direction)
        self.min_ttc_vehicle = float("inf")
        self.sum_distance_to_vehicle = 0.0
        self.vehicle_observation_count = 0
        self.rear_incidents = 0

        # Lane changes
        self.lane_changes_attempted = 0
        self.lane_changes_safe = 0
        self.lane_changes_blocked_by_mandate = 0

        # Episode metadata
        self.timesteps = 0
        self.route_completion = 0.0

    # ------------------------------------------------------------------ #
    def step(self):
        self.timesteps += 1

    def record_vru_observation(self, distance, speed, ttc):
        if distance == float("inf"):
            return
        self.sum_distance_to_vru += float(distance)
        self.vru_observation_count += 1
        if ttc > 0:
            self.min_ttc_vru = min(self.min_ttc_vru, float(ttc))
            if ttc < VRU_NEAR_MISS_TTC:
                self.vru_near_misses.append((self.timesteps, distance, ttc))

    def record_vru_collision(self):
        self.vru_collisions += 1

    def record_vehicle_observation(self, distance, speed, ttc, direction="front"):
        if distance == float("inf"):
            return
        self.sum_distance_to_vehicle += float(distance)
        self.vehicle_observation_count += 1
        if ttc > 0:
            self.min_ttc_vehicle = min(self.min_ttc_vehicle, float(ttc))
            if ttc < VEHICLE_NEAR_MISS_TTC:
                self.vehicle_near_misses.append(
                    (self.timesteps, distance, ttc, direction))
            if direction == "rear" and ttc < REAR_INCIDENT_TTC:
                self.rear_incidents += 1

    def record_vehicle_collision(self):
        self.vehicle_collisions += 1

    def record_lane_change(self, is_safe, blocked_by_mandate=False):
        self.lane_changes_attempted += 1
        if is_safe:
            self.lane_changes_safe += 1
        if blocked_by_mandate:
            self.lane_changes_blocked_by_mandate += 1

    def record_episode_completion(self, route_completion):
        self.route_completion = float(route_completion)

    # ------------------------------------------------------------------ #
    def summarize(self):
        vru_obs = self.vru_observation_count
        veh_obs = self.vehicle_observation_count
        return {
            "vru_collisions": self.vru_collisions,
            "vru_near_misses": len(self.vru_near_misses),
            "min_ttc_vru": self.min_ttc_vru if self.min_ttc_vru < float("inf") else 0,
            "avg_distance_to_vru": (self.sum_distance_to_vru / vru_obs
                                    if vru_obs > 0 else 0),

            "vehicle_collisions": self.vehicle_collisions,
            "vehicle_near_misses": len(self.vehicle_near_misses),
            "min_ttc_vehicle": (self.min_ttc_vehicle
                                if self.min_ttc_vehicle < float("inf") else 0),
            "avg_distance_to_vehicle": (self.sum_distance_to_vehicle / veh_obs
                                        if veh_obs > 0 else 0),
            "rear_incidents": self.rear_incidents,

            "lane_changes_attempted": self.lane_changes_attempted,
            "lane_changes_safe": self.lane_changes_safe,
            "lane_changes_unsafe_prevented": self.lane_changes_blocked_by_mandate,
            "lane_change_success_rate": (
                self.lane_changes_safe / max(1, self.lane_changes_attempted)),

            "route_completion": self.route_completion,
            "episode_duration": self.timesteps,
        }
