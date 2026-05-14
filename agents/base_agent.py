"""
base_agent.py — Abstract base class for all driver agents.

Every agent type (honest RL, tacit colluding, adaptive colluding) inherits
from BaseAgent and must implement:
  - set_price()  : decide what price to post this timestep
  - observe()    : receive feedback after the market clears
  - move()       : decide which zone to move to next timestep
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class DriverObservation:
    """
    Everything a driver can see at the start of a timestep.

    In a real platform drivers see: demand heat maps, surge indicators,
    competitor counts. We give agents exactly this local information —
    no more (no peeking at other agents' policies or prices directly).

    Attributes
    ----------
    driver_id       : this driver's id
    current_zone    : current (row, col)
    timestep        : current timestep
    local_demand    : λ for current zone this timestep
    n_competitors   : number of other drivers in same zone
    last_price      : price this driver posted last timestep
    last_earnings   : earnings last timestep
    zone_intensity  : static busyness of current zone (0.4–2.0)
    """
    driver_id:      int
    current_zone:   Tuple[int, int]
    timestep:       int
    local_demand:   float
    n_competitors:  int
    last_price:     float
    last_earnings:  float
    zone_intensity: float


@dataclass
class DriverAction:
    """
    What a driver decides to do this timestep.

    Attributes
    ----------
    price     : price to post (in rupees, e.g. 80–500)
    online    : whether to be available for rides at all
    move_to   : zone to move to AFTER this timestep (None = stay)
    """
    price:   float
    online:  bool  = True
    move_to: Optional[Tuple[int, int]] = None


@dataclass
class DriverFeedback:
    """
    Feedback received after the market clears at a timestep.

    Attributes
    ----------
    rides_completed : number of rides accepted this timestep
    earnings        : total earnings this timestep
    rides_missed    : requests that came in but price was above WTP
    """
    rides_completed: int
    earnings:        float
    rides_missed:    int


class BaseAgent(ABC):
    """
    Abstract base class for all driver agents.

    Parameters
    ----------
    driver_id      : unique integer id
    initial_zone   : starting (row, col)
    price_min      : floor on posted price
    price_max      : ceiling on posted price
    seed           : per-agent RNG seed
    """

    def __init__(
        self,
        driver_id:    int,
        initial_zone: Tuple[int, int],
        price_min:    float = 60.0,
        price_max:    float = 500.0,
        seed:         int   = 0,
    ):
        self.driver_id    = driver_id
        self.zone         = initial_zone
        self.price_min    = price_min
        self.price_max    = price_max
        self.rng          = np.random.default_rng(seed)

        # State tracked across timesteps
        self.current_price: float = 120.0   # start at base price
        self.online:        bool  = True
        self.total_earnings: float = 0.0
        self.episode_log:   List[dict] = []  # full history for analysis

    # ------------------------------------------------------------------
    # Abstract methods — every subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def set_price(self, obs: DriverObservation) -> DriverAction:
        """
        Given current observation, return a DriverAction.
        This is the agent's core decision function.
        """
        ...

    @abstractmethod
    def observe(self, feedback: DriverFeedback) -> None:
        """
        Receive feedback after market clearing.
        RL agents use this for learning; rule-based agents can ignore it.
        """
        ...

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def clip_price(self, price: float) -> float:
        """Ensure price stays within allowed bounds."""
        return float(np.clip(price, self.price_min, self.price_max))

    def step(self, obs: DriverObservation) -> DriverAction:
        """
        Public-facing method called by the simulator each timestep.
        Calls set_price(), records the action, and updates internal state.
        """
        action = self.set_price(obs)
        action.price = self.clip_price(action.price)

        self.current_price = action.price
        self.online        = action.online
        if action.move_to is not None:
            self.zone = action.move_to

        # Log for episode analysis
        self.episode_log.append({
            "timestep":     obs.timestep,
            "zone":         obs.current_zone,
            "price":        action.price,
            "online":       action.online,
            "local_demand": obs.local_demand,
            "n_competitors": obs.n_competitors,
        })

        return action

    def receive_feedback(self, feedback: DriverFeedback) -> None:
        """Called by simulator after market clearing."""
        self.total_earnings += feedback.earnings
        if self.episode_log:
            self.episode_log[-1]["earnings"]        = feedback.earnings
            self.episode_log[-1]["rides_completed"] = feedback.rides_completed
            self.episode_log[-1]["rides_missed"]    = feedback.rides_missed
        self.observe(feedback)

    def reset(self, initial_zone: Tuple[int, int]) -> None:
        """Reset agent state for a new episode."""
        self.zone            = initial_zone
        self.current_price   = 120.0
        self.online          = True
        self.total_earnings  = 0.0
        self.episode_log     = []