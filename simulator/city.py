"""
city.py — Zone grid and demand engine for CartelSense simulator.

The city is an N×N grid of zones. Each zone has a base demand rate λ that
varies with time-of-day. At each timestep, zones generate ride requests
drawn from a Poisson process.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RideRequest:
    request_id:   int
    origin_zone:  Tuple[int, int]
    dest_zone:    Tuple[int, int]
    wtp:          float           # willingness-to-pay #
    timestep:     int
    accepted:     bool = False


@dataclass
class ZoneState:
    zone:         Tuple[int, int]
    timestep:     int
    demand_rate:  float
    requests:     List[RideRequest] = field(default_factory=list)
    driver_ids:   List[int]         = field(default_factory=list)


# ---------------------------------------------------------------------------
# Demand profile
# ---------------------------------------------------------------------------

class DemandProfile:
    """
    Encodes time-of-day demand variation over 288 timesteps (5-min steps).

    Three demand peaks:
      - Morning rush  : ~timestep 78  (6:30am)
      - Midday        : ~timestep 144 (12:00pm)
      - Evening rush  : ~timestep 210 (5:30pm)
    """

    STEPS_PER_DAY = 288

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._tod_curve = self._build_tod_curve()

    def _build_tod_curve(self) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, self.STEPS_PER_DAY)

        def gaussian_bump(center_step, width, height):
            return height * np.exp(
                -0.5 * ((np.arange(self.STEPS_PER_DAY) - center_step) / width) ** 2
            )

        morning  = gaussian_bump(78,  18, 2.5)
        lunch    = gaussian_bump(144, 12, 1.4)
        evening  = gaussian_bump(210, 20, 2.2)
        baseline = 0.6 + 0.4 * np.sin(t - np.pi / 2)

        curve = baseline + morning + lunch + evening
        noise = self.rng.normal(0, 0.05, size=self.STEPS_PER_DAY)
        return np.clip(curve + noise, 0.1, None)

    def get_multiplier(self, timestep: int) -> float:
        return float(self._tod_curve[timestep % self.STEPS_PER_DAY])


# ---------------------------------------------------------------------------
# City grid
# ---------------------------------------------------------------------------

class City:
    """
    N×N grid of zones. Manages demand generation and zone state.

    Parameters
    ----------
    grid_size   : side length of the grid (default 8 → 64 zones)
    base_demand : average requests/timestep for an average zone
    seed        : RNG seed for reproducibility
    """

    def __init__(self, grid_size: int = 8, base_demand: float = 3.0, seed: int = 42):
        self.grid_size   = grid_size
        self.base_demand = base_demand
        self.rng         = np.random.default_rng(seed)

        self.demand_profile    = DemandProfile(rng=self.rng)
        self.zone_intensity    = self._build_zone_intensity()
        self.transition_matrix = self._build_transition_matrix()

        self._request_counter = 0

        # Log-normal WTP params: median ~120, reasonable spread
        self.wtp_mu    = np.log(120)
        self.wtp_sigma = 0.5

    def _build_zone_intensity(self) -> np.ndarray:
        intensity = np.ones((self.grid_size, self.grid_size))
        center = self.grid_size / 2
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                dist     = np.sqrt((r - center) ** 2 + (c - center) ** 2)
                max_dist = np.sqrt(2) * center
                intensity[r, c] = 0.4 + 1.6 * (1 - dist / max_dist)
        return intensity

    def _build_transition_matrix(self) -> np.ndarray:
        n      = self.grid_size * self.grid_size
        matrix = np.zeros((n, n))
        center_idx = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)

        for i in range(n):
            r1, c1  = divmod(i, self.grid_size)
            weights = np.zeros(n)
            for j in range(n):
                if i == j:
                    continue
                r2, c2       = divmod(j, self.grid_size)
                dist         = np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)
                gravity      = 1.0 / (dist ** 2 + 1e-6)
                center_bonus = 2.0 if j == center_idx else 1.0
                weights[j]   = gravity * center_bonus
            matrix[i] = weights / weights.sum()

        return matrix

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def zone_to_idx(self, zone: Tuple[int, int]) -> int:
        return zone[0] * self.grid_size + zone[1]

    def idx_to_zone(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.grid_size)

    def all_zones(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]

    def get_zone_demand_rate(self, zone: Tuple[int, int], timestep: int) -> float:
        r, c = zone
        return self.base_demand * self.zone_intensity[r, c] * self.demand_profile.get_multiplier(timestep)

    # ------------------------------------------------------------------
    # Core: generate requests for all zones at a timestep
    # ------------------------------------------------------------------

    def generate_requests(
        self,
        timestep: int,
        driver_positions: Dict[int, Tuple[int, int]],
    ) -> Dict[Tuple[int, int], ZoneState]:
        """
        Generate ride requests for every zone at the given timestep.

        Parameters
        ----------
        timestep         : current step (0–287)
        driver_positions : {driver_id: zone}

        Returns
        -------
        {zone: ZoneState} for all zones
        """
        tod_mult = self.demand_profile.get_multiplier(timestep)

        # Build zone → driver list mapping
        zone_to_drivers: Dict[Tuple[int, int], List[int]] = {z: [] for z in self.all_zones()}
        for driver_id, zone in driver_positions.items():
            zone_to_drivers[zone].append(driver_id)

        zone_states = {}

        for zone in self.all_zones():
            r, c = zone
            lam  = self.base_demand * self.zone_intensity[r, c] * tod_mult
            n_req = self.rng.poisson(lam)

            requests = []
            for _ in range(n_req):
                origin_idx = self.zone_to_idx(zone)
                dest_idx   = self.rng.choice(
                    self.grid_size * self.grid_size,
                    p=self.transition_matrix[origin_idx],
                )
                dest_zone = self.idx_to_zone(dest_idx)
                wtp       = float(self.rng.lognormal(self.wtp_mu, self.wtp_sigma))

                requests.append(RideRequest(
                    request_id  = self._request_counter,
                    origin_zone = zone,
                    dest_zone   = dest_zone,
                    wtp         = wtp,
                    timestep    = timestep,
                ))
                self._request_counter += 1

            zone_states[zone] = ZoneState(
                zone        = zone,
                timestep    = timestep,
                demand_rate = lam,
                requests    = requests,
                driver_ids  = zone_to_drivers[zone],
            )

        return zone_states
