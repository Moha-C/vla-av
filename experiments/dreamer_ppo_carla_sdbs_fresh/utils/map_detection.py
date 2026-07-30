"""Detect whether the ego is on an unknown (untrained) map.

A simple heuristic: if the current location is outside the radius of every
trained map's center, treat the map as unknown and switch to defensive driving.
"""
import numpy as np


def _distance(a, b):
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def is_unknown_map(current_location, trained_map_list):
    """True if ``current_location`` is outside every trained map's radius."""
    for trained_map in trained_map_list:
        if _distance(current_location, trained_map["center"]) < trained_map["radius"]:
            return False        # within a known map
    return True


def activate_defensive_mode_if_unknown(location, trained_maps, defensive_controller):
    """Switch the controller to defensive mode when on an unknown map."""
    if is_unknown_map(location, trained_maps):
        print(f"Unknown map detected at {location}")
        defensive_controller.activate_defensive_mode()
        return True
    return False
