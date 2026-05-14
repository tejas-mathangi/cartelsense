"""
colluding_agent.py — Type B tacit colluding driver agents.

All Type B agents share the SAME SharedPricingAlgorithm instance.
They never communicate directly, but because they feed identical inputs
into the same function, they produce correlated pricing behavior.

This is tacit collusion — the coordination is implicit in the shared
algorithm, not in any direct communication between agents.

Four collusion modes are implemented as behaviors the shared algorithm
can be configured to exhibit:

  MODE_PRICE_PARALLELISM   : all agents mirror surge multipliers tightly
  MODE_ZONE_AVOIDANCE      : agents collectively avoid low-margin zones
  MODE_SURGE_SYNCHRONIZE   : agents go offline together to manufacture surge
  MODE_PHANTOM_SCARCITY    : agents disappear briefly, return at higher price
"""

import numpy as np
from typing import Tuple, Optional, List
from agents.base_agent import BaseAgent, DriverObservation, DriverAction, DriverFeedback


# ---------------------------------------------------------------------------
# Collusion mode constants
# ---------------------------------------------------------------------------

MODE_PRICE_PARALLELISM  = "price_parallelism"
MODE_ZONE_AVOIDANCE     = "zone_avoidance"
MODE_SURGE_SYNCHRONIZE  = "surge_synchronization"
MODE_PHANTOM_SCARCITY   = "phantom_scarcity"
MODE_NONE               = "none"      # honest baseline for Type B (no collusion)


# ---------------------------------------------------------------------------
# Shared pricing algorithm — the "third-party app" all colluding agents use
# ---------------------------------------------------------------------------

class SharedPricingAlgorithm:
    """
    The pricing algorithm shared by all Type B colluding drivers.

    Think of this as the code inside a third-party pricing app like
    "SurgeMaster Pro" — every driver who installs it runs the same logic.

    Parameters
    ----------
    base_price        : starting price before any multipliers (₹)
    surge_sensitivity : how aggressively to raise price with demand/supply ratio
    collusion_mode    : which collusion behavior to exhibit
    avoidance_zones   : zones to collectively avoid (for MODE_ZONE_AVOIDANCE)
    offline_duration  : how long agents go dark (for surge/phantom modes)
    """

    def __init__(
        self,
        base_price:        float = 120.0,
        surge_sensitivity: float = 1.8,
        collusion_mode:    str   = MODE_PRICE_PARALLELISM,
        avoidance_zones:   Optional[List[Tuple[int, int]]] = None,
        offline_duration:  int   = 6,    # timesteps (= 30 minutes)
    ):
        self.base_price        = base_price
        self.surge_sensitivity = surge_sensitivity
        self.collusion_mode    = collusion_mode
        self.avoidance_zones   = avoidance_zones or []
        self.offline_duration  = offline_duration

        # Shared internal state — all agents reading from the same object
        # see the same internal clock, making their behavior synchronized
        self._global_step       = 0
        self._offline_countdown = 0   # counts down when group is "dark"
        self._in_phantom_phase  = False

    def tick(self) -> None:
        """Advance the shared global clock. Called once per timestep by simulator."""
        self._global_step += 1

        if self._offline_countdown > 0:
            self._offline_countdown -= 1
            if self._offline_countdown == 0:
                self._in_phantom_phase = False

    def compute_action(self, obs: DriverObservation) -> DriverAction:
        """
        Core logic: given an observation, return what action to take.
        All Type B agents call this same method → synchronized behavior.
        """

        if self.collusion_mode == MODE_PRICE_PARALLELISM:
            return self._price_parallelism(obs)

        elif self.collusion_mode == MODE_ZONE_AVOIDANCE:
            return self._zone_avoidance(obs)

        elif self.collusion_mode == MODE_SURGE_SYNCHRONIZE:
            return self._surge_synchronize(obs)

        elif self.collusion_mode == MODE_PHANTOM_SCARCITY:
            return self._phantom_scarcity(obs)

        else:  # MODE_NONE — competitive honest behavior for comparison
            return self._competitive(obs)

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _surge_multiplier(self, demand: float, n_competitors: int) -> float:
        """
        Core surge formula: price rises with demand, falls with competition.
        multiplier = base × (demand / (competitors + 1)) ^ sensitivity
        Clipped to [1.0, 4.0] range.
        """
        supply_demand_ratio = demand / (n_competitors + 1.0)
        multiplier = supply_demand_ratio ** self.surge_sensitivity
        return float(np.clip(multiplier, 1.0, 4.0))

    def _price_parallelism(self, obs: DriverObservation) -> DriverAction:
        """
        MODE 1: All agents apply the SAME surge multiplier with minimal noise.
        Honest independent agents add personal noise; these agents don't.
        The result: unnaturally tight price clustering across the group.
        """
        mult  = self._surge_multiplier(obs.local_demand, obs.n_competitors)
        price = self.base_price * mult
        # Tiny noise (±2%) — just enough to look plausible, not enough to break correlation
        noise = 1.0 + np.random.normal(0, 0.02)
        return DriverAction(price=price * noise, online=True)

    def _zone_avoidance(self, obs: DriverObservation) -> DriverAction:
        """
        MODE 2: Collectively avoid designated zones.
        If in an avoidance zone, go offline or move away.
        In allowed zones, use slightly elevated prices due to reduced competition.
        """
        if obs.current_zone in self.avoidance_zones:
            # Move to nearest non-avoided zone (simplified: just go offline)
            return DriverAction(price=self.base_price, online=False)

        # In non-avoided zones: fewer competitors (they're all avoiding too),
        # so the surge multiplier yields higher prices
        mult  = self._surge_multiplier(obs.local_demand, max(0, obs.n_competitors - 2))
        price = self.base_price * mult * 1.1   # 10% premium from reduced competition
        return DriverAction(price=price, online=True)

    def _surge_synchronize(self, obs: DriverObservation) -> DriverAction:
        """
        MODE 3: All agents go offline together every K timesteps for D duration.
        This manufactures artificial scarcity, triggering surge pricing.
        They return together at an elevated price.

        The synchronized offline window is controlled by the shared global clock,
        so all agents go dark and return at the exact same timestep.
        """
        # Every 48 timesteps (4 hours), trigger a synchronized offline window
        trigger_interval = 48
        in_offline_window = (self._global_step % trigger_interval) < self.offline_duration

        if in_offline_window:
            return DriverAction(price=self.base_price, online=False)

        # Just returned from offline — charge a premium (demand pent up)
        just_returned = (self._global_step % trigger_interval) == self.offline_duration
        price_boost   = 1.4 if just_returned else 1.0

        mult  = self._surge_multiplier(obs.local_demand, obs.n_competitors)
        price = self.base_price * mult * price_boost
        return DriverAction(price=price, online=True)

    def _phantom_scarcity(self, obs: DriverObservation) -> DriverAction:
        """
        MODE 4: When demand is rising, the group goes offline briefly to
        create phantom scarcity, then returns at a higher price.

        Triggered when: demand is above a threshold AND group is not already dark.
        """
        demand_threshold = 4.0   # λ above which the group triggers phantom scarcity

        # Trigger phantom if demand is high and we're not already in a phantom phase
        if obs.local_demand > demand_threshold and not self._in_phantom_phase:
            # Every agent reads this same flag → they all go dark simultaneously
            if self._offline_countdown == 0:
                self._offline_countdown = self.offline_duration
                self._in_phantom_phase  = True

        if self._in_phantom_phase:
            return DriverAction(price=self.base_price, online=False)

        # Post-phantom: charge elevated price
        mult  = self._surge_multiplier(obs.local_demand, obs.n_competitors)
        price = self.base_price * mult * 1.3   # 30% post-scarcity premium
        return DriverAction(price=price, online=True)

    def _competitive(self, obs: DriverObservation) -> DriverAction:
        """Honest competitive pricing — used when mode is MODE_NONE."""
        mult  = self._surge_multiplier(obs.local_demand, obs.n_competitors)
        price = self.base_price * mult
        noise = 1.0 + np.random.normal(0, 0.08)   # more noise = less correlated
        return DriverAction(price=price * noise, online=True)


# ---------------------------------------------------------------------------
# Type B Driver Agent
# ---------------------------------------------------------------------------

class ColludingAgent(BaseAgent):
    """
    Type B driver: uses a SharedPricingAlgorithm.

    All instances that share the same `algorithm` object are colluding —
    they behave in a coordinated way without direct communication.

    Parameters
    ----------
    driver_id    : unique integer id
    initial_zone : starting zone
    algorithm    : shared SharedPricingAlgorithm instance (same object for whole group)
    seed         : per-agent RNG seed
    """

    def __init__(
        self,
        driver_id:    int,
        initial_zone: Tuple[int, int],
        algorithm:    SharedPricingAlgorithm,
        seed:         int = 0,
    ):
        super().__init__(driver_id=driver_id, initial_zone=initial_zone, seed=seed)
        self.algorithm = algorithm   # shared — same object across the whole group

    def set_price(self, obs: DriverObservation) -> DriverAction:
        """Delegate entirely to the shared algorithm."""
        return self.algorithm.compute_action(obs)

    def observe(self, feedback: DriverFeedback) -> None:
        """
        Type B agents don't learn from feedback — they just follow the algorithm.
        In a real scenario this would be a SaaS product with fixed pricing logic.
        """
        pass   # no learning