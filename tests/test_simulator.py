"""
test_simulator.py — Smoke tests for the simulator stack.
Run from /home/claude/cartelsense with: python -m tests.test_simulator
"""

import sys
sys.path.insert(0, "/home/claude/cartelsense")

import numpy as np
from simulator.city import City
from simulator.factory import make_colluding_scenario
from agents.colluding_agent import (
    MODE_PRICE_PARALLELISM, MODE_ZONE_AVOIDANCE,
    MODE_SURGE_SYNCHRONIZE, MODE_PHANTOM_SCARCITY,
)


def test_city_demand():
    print("TEST 1: City demand generation")
    city = City(grid_size=8, base_demand=3.0, seed=42)

    # Generate at peak (timestep 78 = morning rush) and off-peak (timestep 10)
    peak_states = city.generate_requests(78,  {})
    off_states  = city.generate_requests(10,  {})

    peak_total = sum(len(s.requests) for s in peak_states.values())
    off_total  = sum(len(s.requests) for s in off_states.values())

    print(f"  Peak requests (t=78):    {peak_total}")
    print(f"  Off-peak requests (t=10): {off_total}")
    assert peak_total > off_total, "Peak demand should exceed off-peak"

    # Check WTP distribution
    all_wtp = [r.wtp for s in peak_states.values() for r in s.requests]
    print(f"  WTP median: ₹{np.median(all_wtp):.1f}, mean: ₹{np.mean(all_wtp):.1f}")
    assert 80 < np.median(all_wtp) < 200, "WTP median out of expected range"
    print("  PASSED\n")


def test_episode_runs():
    print("TEST 2: Full episode run — price parallelism mode")
    runner  = make_colluding_scenario(collusion_mode=MODE_PRICE_PARALLELISM, seed=42)
    episode = runner.run(episode_id=0)

    assert len(episode.timestep_records) == 288, "Episode should have 288 timesteps"
    assert sum(episode.labels) == 288, "All timesteps should be labeled collusion=1"

    # Check price range is sensible
    all_prices = []
    for rec in episode.timestep_records:
        all_prices.extend(rec.driver_prices.values())

    print(f"  Price range: ₹{min(all_prices):.1f} – ₹{max(all_prices):.1f}")
    print(f"  Mean price: ₹{np.mean(all_prices):.1f}")
    assert 60 <= min(all_prices) <= 600, "Prices out of expected range"
    print("  PASSED\n")


def test_collusion_signal():
    print("TEST 3: Collusion correlation signal")

    # Run one colluding episode and one honest episode
    colluding_runner = make_colluding_scenario(collusion_mode=MODE_PRICE_PARALLELISM, seed=42)
    honest_runner    = make_colluding_scenario(collusion_mode="none", seed=42)

    colluding_ep = colluding_runner.run(episode_id=0)
    honest_ep    = honest_runner.run(episode_id=1)

    def avg_price_correlation(episode, agent_ids):
        """Average pairwise price correlation among a group of agents."""
        price_series = {}
        for did in agent_ids:
            price_series[did] = [rec.driver_prices[did] for rec in episode.timestep_records]

        ids   = list(agent_ids)
        corrs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                c = np.corrcoef(price_series[ids[i]], price_series[ids[j]])[0, 1]
                corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    # Compare correlation of colluding group vs honest group
    colluding_group = colluding_runner.colluding_ids
    honest_group    = list(range(colluding_runner.agents.__len__() - len(colluding_group)))

    colluding_corr = avg_price_correlation(colluding_ep, colluding_group)
    honest_corr    = avg_price_correlation(honest_ep, list(range(6)))

    print(f"  Colluding group avg pairwise price correlation: {colluding_corr:.3f}")
    print(f"  Honest group avg pairwise price correlation:    {honest_corr:.3f}")
    assert colluding_corr > honest_corr, \
        "Colluding agents should have higher price correlation than honest agents"
    print("  PASSED\n")


def test_all_modes():
    print("TEST 4: All four collusion modes run without error")
    modes = [MODE_PRICE_PARALLELISM, MODE_ZONE_AVOIDANCE,
             MODE_SURGE_SYNCHRONIZE, MODE_PHANTOM_SCARCITY]

    for mode in modes:
        runner  = make_colluding_scenario(collusion_mode=mode, seed=99)
        episode = runner.run(episode_id=0)
        total_earnings = sum(
            sum(rec.driver_earnings.values())
            for rec in episode.timestep_records
        )
        print(f"  {mode:30s} | total earnings: ₹{total_earnings:,.0f}")
        assert len(episode.timestep_records) == 288
    print("  PASSED\n")


if __name__ == "__main__":
    test_city_demand()
    test_episode_runs()
    test_collusion_signal()
    test_all_modes()
    print("=" * 50)
    print("All tests passed.")