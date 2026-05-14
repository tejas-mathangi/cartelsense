"""
episode.py — Episode runner and data labeler.

An episode = one simulated day (288 timesteps).
Runs the full simulation loop and produces a labeled dataset:
  - Price time series per driver
  - Zone state per timestep
  - Ground-truth collusion labels per timestep window

This is the training data generator for the detector.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from simulator.city import City
from simulator.market import Market
from agents.base_agent import BaseAgent, DriverObservation
from agents.colluding_agent import SharedPricingAlgorithm


# ---------------------------------------------------------------------------
# Episode data containers
# ---------------------------------------------------------------------------

@dataclass
class TimestepRecord:
    """Everything that happened at one timestep."""
    timestep:        int
    zone_demands:    Dict[Tuple[int, int], float]   # zone → λ
    driver_prices:   Dict[int, float]               # driver_id → price
    driver_online:   Dict[int, bool]                # driver_id → online status
    driver_zones:    Dict[int, Tuple[int, int]]     # driver_id → zone
    driver_earnings: Dict[int, float]               # driver_id → earnings
    n_requests:      Dict[Tuple[int, int], int]     # zone → request count


@dataclass
class EpisodeData:
    """
    Complete data from one simulated episode.

    Attributes
    ----------
    episode_id       : unique identifier
    collusion_mode   : which mode was active (or 'none')
    colluding_ids    : set of driver IDs that were colluding
    timestep_records : list of TimestepRecord, one per timestep
    labels           : per-timestep binary label (1 = active collusion)
    """
    episode_id:       int
    collusion_mode:   str
    colluding_ids:    List[int]
    timestep_records: List[TimestepRecord] = field(default_factory=list)
    labels:           List[int]            = field(default_factory=list)   # 0 or 1 per timestep


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """
    Runs a single episode and returns labeled EpisodeData.

    Parameters
    ----------
    city             : City instance
    agents           : {driver_id: BaseAgent}
    colluding_ids    : which driver IDs are colluding (for labeling)
    collusion_mode   : string name of the active collusion mode
    shared_algorithm : SharedPricingAlgorithm instance (for ticking its clock)
    n_steps          : number of timesteps per episode (default 288 = 1 day)
    """

    def __init__(
        self,
        city:              City,
        agents:            Dict[int, BaseAgent],
        colluding_ids:     List[int],
        collusion_mode:    str,
        shared_algorithm:  Optional[SharedPricingAlgorithm] = None,
        n_steps:           int = 288,
    ):
        self.city             = city
        self.agents           = agents
        self.colluding_ids    = set(colluding_ids)
        self.collusion_mode   = collusion_mode
        self.shared_algorithm = shared_algorithm
        self.n_steps          = n_steps
        self.market           = Market()

    def run(self, episode_id: int = 0) -> EpisodeData:
        """
        Execute one full episode and return labeled data.

        The simulation loop per timestep:
          1. City generates requests (demand)
          2. Each driver observes local state → sets price
          3. Market clears → drivers earn money
          4. Drivers receive feedback
          5. Record everything
        """
        episode = EpisodeData(
            episode_id     = episode_id,
            collusion_mode = self.collusion_mode,
            colluding_ids  = list(self.colluding_ids),
        )

        for t in range(self.n_steps):
            # Step shared algorithm clock (synchronizes colluding agents)
            if self.shared_algorithm is not None:
                self.shared_algorithm.tick()

            # Current driver positions
            driver_positions = {did: agent.zone for did, agent in self.agents.items()}

            # 1. Generate demand
            zone_states = self.city.generate_requests(t, driver_positions)

            # 2. Each driver sets price
            driver_actions = {}
            for did, agent in self.agents.items():
                zone  = agent.zone
                state = zone_states[zone]
                n_competitors = len(state.driver_ids) - 1   # exclude self

                obs = DriverObservation(
                    driver_id      = did,
                    current_zone   = zone,
                    timestep       = t,
                    local_demand   = state.demand_rate,
                    n_competitors  = max(0, n_competitors),
                    last_price     = agent.current_price,
                    last_earnings  = agent.episode_log[-1]["earnings"] if agent.episode_log and "earnings" in agent.episode_log[-1] else 0.0,
                    zone_intensity = self.city.zone_intensity[zone[0], zone[1]],
                )

                driver_actions[did] = agent.step(obs)

            # 3. Market clearing
            feedback_map = self.market.clear(zone_states, driver_actions, self.agents)

            # 4. Feedback to agents
            for did, agent in self.agents.items():
                agent.receive_feedback(feedback_map[did])

            # 5. Record timestep
            record = TimestepRecord(
                timestep        = t,
                zone_demands    = {z: s.demand_rate for z, s in zone_states.items()},
                driver_prices   = {did: driver_actions[did].price for did in self.agents},
                driver_online   = {did: driver_actions[did].online for did in self.agents},
                driver_zones    = {did: self.agents[did].zone for did in self.agents},
                driver_earnings = {did: feedback_map[did].earnings for did in self.agents},
                n_requests      = {z: len(s.requests) for z, s in zone_states.items()},
            )
            episode.timestep_records.append(record)

            # 6. Label: collusion is "active" when colluding agents exist and mode isn't none
            label = 1 if (self.colluding_ids and self.collusion_mode != "none") else 0
            episode.labels.append(label)

        return episode