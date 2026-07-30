"""Map-independent state features for generalization to unknown towns.

These features are *computed*, not learned/memorized: how centered the car is
in its lane, the road type, an estimated visibility, and whether there is
oncoming traffic. Because they describe geometry/relations rather than a
specific map, a policy that relies on them transfers to towns it never trained
on.

All indices match the 48-dim state in env/carla_env.py (lane offset/width at
6/7, curvature/junction at 8/9, ego speed/heading at 2/3, side-vehicle blocks
at 23-27 (left) and 28-32 (right)).
"""
import numpy as np

# Heading difference (radians) that counts as "oncoming" (~150-210 degrees).
_ONCOMING_LO = np.radians(150.0)
_ONCOMING_HI = np.radians(210.0)

_ROAD_TYPE_ONEHOT = {
    "straight":     [1.0, 0.0, 0.0, 0.0],
    "curve":        [0.0, 1.0, 0.0, 0.0],
    "intersection": [0.0, 0.0, 1.0, 0.0],
    "unknown":      [0.0, 0.0, 0.0, 1.0],
}
N_EXTRA_FEATURES = 7          # in_lane_center + road-type one-hot[4] + vis + oncoming

_WEATHER_VIS_REDUCTION = {
    "clear": 0.0,
    "rain": 0.2,
    "fog": 0.4,
    "night": 0.3,
}


def compute_in_lane_center(state):
    """1.0 = perfectly centered in the lane, 0.0 = at the lane edge."""
    lane_offset = float(state[6])
    lane_width = float(state[7])
    if lane_width == 0:
        return 0.5                      # unknown lane width
    edge_threshold = lane_width / 2.0
    in_lane = 1.0 - min(1.0, abs(lane_offset) / edge_threshold)
    return max(0.0, in_lane)


def compute_road_type(state):
    """Classify the road as straight / curve / sharp_curve / intersection."""
    curvature = float(state[8])
    if bool(state[9]):
        return "intersection"
    if abs(curvature) < 0.01:
        return "straight"
    if abs(curvature) < 0.05:
        return "curve"
    return "sharp_curve"


def estimate_visibility(info, config=None):
    """Estimate visibility in [0, 1] from VRU/vehicle density and weather."""
    info = info or {}
    vru_count = len(info.get("vru_list", []))
    vehicle_count = sum(
        1 for v in info.get("vehicle_list", [])
        if float(v.get("distance", v.get("dist", 0.0))) < 30
    )
    weather = info.get("weather", "clear")

    visibility = 1.0
    visibility -= 0.1 * min(1.0, vru_count / 3.0)
    visibility -= 0.05 * min(1.0, vehicle_count / 5.0)
    visibility -= _WEATHER_VIS_REDUCTION.get(weather, 0.1)
    return max(0.0, min(1.0, visibility))


def detect_oncoming_traffic(state):
    """True if a side vehicle within 50 m heads roughly opposite to the ego."""
    # Side-vehicle blocks (left 23-27, right 28-32) only exist with Tier-1
    # state; without them there is nothing to compare, so report no oncoming.
    if len(state) < 33:
        return False
    ego_heading = float(state[3])
    for block in (state[23:28], state[28:33]):     # left, right vehicle blocks
        if float(block[0]) < 50.0:                 # within 50 m
            diff = abs(float(block[2]) - ego_heading) % (2.0 * np.pi)
            if _ONCOMING_LO < diff < _ONCOMING_HI:
                return True
    return False


def compute_map_agnostic_features(state, info, config=None):
    """Bundle the map-agnostic features into a dict."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    info = info or {}
    return {
        "in_lane_center": compute_in_lane_center(state),
        "lane_width": float(state[7]),
        "road_curvature": float(state[8]),
        "is_intersection": bool(state[9]),
        "visibility": estimate_visibility(info, config),
        "oncoming_traffic": detect_oncoming_traffic(state),
        "road_type": compute_road_type(state),
        "time_of_day": info.get("time_of_day", "day"),
        "weather": info.get("weather", "clear"),
    }


def map_agnostic_vector(state, info, config=None):
    """Return the 7-dim map-agnostic feature vector for ``state`` (np.float32).

    in_lane_center (1) + road-type one-hot (4) + visibility (1) + oncoming (1).
    """
    feats = compute_map_agnostic_features(state, info, config)
    extra = [feats["in_lane_center"]]
    # 'sharp_curve' (and anything unmapped) falls back to the 'unknown' slot.
    extra.extend(_ROAD_TYPE_ONEHOT.get(feats["road_type"],
                                       _ROAD_TYPE_ONEHOT["unknown"]))
    extra.append(feats["visibility"])
    extra.append(float(feats["oncoming_traffic"]))
    return np.asarray(extra, dtype=np.float32)
