"""
market.py — Market clearing engine.

At each timestep, after all drivers set their prices, the market clears:
  1. Each zone's ride requests are matched to available drivers
  2. Drivers earn money for accepted rides
  3. Feedback is sent to each driver

Matching rule: cheapest available driver in zone gets the ride,
IF their price <= passenger's WTP. Simple and realistic.
"""

import numpy as np
from typing import Dict, List, Tuple
from simulator.city import ZoneState, RideRequest
from agents.base_agent import BaseAgent, DriverFeedback, DriverAction


class Market:
    """
    Clears the market at each timestep.

    The market sees: zone states (demand) + driver actions (prices) →
    produces: matches (who drove whom) + feedback per driver.
    """

    def clear(
        self,
        zone_states:    Dict[Tuple[int, int], ZoneState],
        driver_actions: Dict[int, DriverAction],
        agents:         Dict[int, BaseAgent],
    ) -> Dict[int, DriverFeedback]:
        """
        Run market clearing for one timestep.

        Parameters
        ----------
        zone_states    : output of City.generate_requests()
        driver_actions : {driver_id: DriverAction} for this timestep
        agents         : {driver_id: BaseAgent} for position info

        Returns
        -------
        {driver_id: DriverFeedback}
        """
        # Initialize feedback for all drivers
        feedback: Dict[int, DriverFeedback] = {
            did: DriverFeedback(rides_completed=0, earnings=0.0, rides_missed=0)
            for did in driver_actions
        }

        for zone, state in zone_states.items():
            if not state.requests:
                continue

            # Get online drivers in this zone, sorted by price ascending
            # (cheapest driver gets matched first — realistic competitive behavior)
            zone_drivers = [
                (did, driver_actions[did])
                for did in state.driver_ids
                if did in driver_actions and driver_actions[did].online
            ]
            zone_drivers.sort(key=lambda x: x[1].price)

            # Match requests to drivers
            # Each driver can complete at most 1 ride per timestep (simplified)
            available_drivers = list(zone_drivers)   # copy so we can pop

            for request in state.requests:
                if not available_drivers:
                    break   # no more drivers, remaining requests go unserved

                driver_id, action = available_drivers[0]

                if action.price <= request.wtp:
                    # Match! Driver completes the ride
                    request.accepted = True
                    feedback[driver_id].rides_completed += 1
                    feedback[driver_id].earnings        += action.price
                    available_drivers.pop(0)   # driver is now busy
                else:
                    # Cheapest driver is still too expensive for this passenger
                    # Passenger leaves (no ride)
                    feedback[driver_id].rides_missed += 1

        return feedback