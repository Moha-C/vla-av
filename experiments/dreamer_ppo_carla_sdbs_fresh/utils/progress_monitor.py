"""Route-progress stall detector.

Catches a stuck / non-advancing ``route_progress`` signal early, so a broken
env/reward wiring is diagnosed during the run instead of being mistaken for a
policy-quality problem after a full sweep (the classic "0% route completion
across every variant" failure).
"""


class ProgressMonitor:
    """Detects a stuck/non-advancing route_progress signal early."""

    def __init__(self, stall_threshold_steps=200):
        self.history = []
        self.stall_threshold_steps = stall_threshold_steps

    def record(self, timestep, route_progress):
        self.history.append((timestep, float(route_progress)))

    def check_stalled(self) -> dict:
        """Return diagnostic info about the route_progress signal.

        If route_progress hasn't changed at all over ``stall_threshold_steps``
        steps, this flags a likely env/reward wiring bug rather than a policy
        quality issue.
        """
        if len(self.history) < self.stall_threshold_steps:
            return {'stalled': False, 'reason': 'insufficient_data'}

        recent = self.history[-self.stall_threshold_steps:]
        values = [v for _, v in recent]

        if max(values) - min(values) < 1e-6:
            return {
                'stalled': True,
                'reason': 'route_progress has not changed in '
                          f'{self.stall_threshold_steps} steps — '
                          'likely an env/reward wiring bug, not a '
                          'policy quality issue. Check that route_progress '
                          'is being read/updated correctly before drawing '
                          'any conclusions about policy performance.',
                'min': min(values),
                'max': max(values),
            }
        return {'stalled': False}
