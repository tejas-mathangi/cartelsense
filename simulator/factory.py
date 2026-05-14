"""
factory.py — Convenience functions for building simulation scenarios.

Instead of manually wiring City + Agents + SharedAlgorithm + EpisodeRunner
every time, these factories create complete ready-to-run scenarios.
"""

import numpy as np
from typing import Tuple, Optional
from simulator.city import City
from simulator.episode import EpisodeRunner
from agents.colluding_agent import (
    ColludingAgent, SharedPricingAlgorithm,
    MODE_PRICE_PARALLELISM, MODE_ZONE_AVOIDANCE,
    MODE_SURGE_SYNCHRONIZE, MODE_PHANTOM_SCARCITY, MODE_NONE,
)


def make_colluding_scenario(
    collusion_mode:     str   = MODE_PRICE_PARALLELISM,
    n_honest:           int   = 6,
    n_colluding:        int   = 4,
    grid_size:          int   = 8,
    base_demand:        float = 3.0,
    seed:               int   = 42,
) -> EpisodeRunner:
    """
    Build a scenario with a mix of honest (competitive) and colluding drivers.

    The honest drivers also use SharedPricingAlgorithm with MODE_NONE —
    meaning they behave competitively (just like colluding agents structurally,
    but with no coordination). This is intentional: it isolates the collusion
    behavior rather than the agent architecture.

    Returns a ready-to-run EpisodeRunner.
    """
    rng  = np.random.default_rng(seed)
    city = City(grid_size=grid_size, base_demand=base_demand, seed=seed)

    all_zones = city.all_zones()

    # Shared algorithm for the colluding group
    avoidance_zones = all_zones[:4] if collusion_mode == MODE_ZONE_AVOIDANCE else []
    shared_algo = SharedPricingAlgorithm(
        collusion_mode  = collusion_mode,
        avoidance_zones = avoidance_zones,
    )

    # Honest algorithm (competitive, no coordination)
    honest_algo = SharedPricingAlgorithm(collusion_mode=MODE_NONE)

    agents = {}
    colluding_ids = []

    total_drivers = n_honest + n_colluding
    start_zones   = [all_zones[i % len(all_zones)] for i in range(total_drivers)]

    # Honest drivers (IDs 0 to n_honest-1)
    for i in range(n_honest):
        agents[i] = ColludingAgent(
            driver_id    = i,
            initial_zone = start_zones[i],
            algorithm    = honest_algo,
            seed         = seed + i,
        )

    # Colluding drivers (IDs n_honest to total-1)
    for i in range(n_colluding):
        did = n_honest + i
        agents[did] = ColludingAgent(
            driver_id    = did,
            initial_zone = start_zones[did],
            algorithm    = shared_algo,
            seed         = seed + did,
        )
        colluding_ids.append(did)

    return EpisodeRunner(
        city             = city,
        agents           = agents,
        colluding_ids    = colluding_ids,
        collusion_mode   = collusion_mode,
        shared_algorithm = shared_algo,
    )