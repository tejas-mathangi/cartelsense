# CartelSense — Project Journey

> A personal log of how this project was built, what decisions were made, why they were made, and what problems came up along the way. Updated at every major milestone.

**Repo:** https://github.com/tejas-mathangi/cartelsense  
**Target venues:** ACM EC or AAAI 2027, arXiv preprint first  
**Constraint:** CPU-only (no GPU available)

---

## Phase 0 — Finding the Right Project

### Where it started

Looked at two friends' projects for inspiration:

- **Harshith's Algorithmic Collusion Detector** — a MARL + CNN system detecting tacit collusion between trading bots in a simulated limit order book. Four collusion schemes (wash trading, tape painting, spoofing, mirror trading), a C++14 simulator, and IPPO honest-agent validation.
- **TOPOS v2** — a cognitive architecture experiment giving a frozen 7B LLM an external workspace (EMA state vector, ChromaDB episodic memory, NetworkX concept graph, ensemble curiosity module) to produce emergent personality without fine-tuning.

Both were strong student projects. The goal was to find something at the same level or above — novel, publishable, buildable on a laptop, and built from scratch.

### Project ideas considered

Four directions were evaluated:

| Idea | Research novelty | Publishable | CPU feasible |
|---|---|---|---|
| Mechanistic Diff — AST-level semantic version control | High | MSR / ICSE | Yes |
| MemoryOS — external memory for small LLMs | Medium | NeurIPS workshops | Yes |
| Curriculum Cartographer — RL agent that learns to teach | Medium | EDM / AIED | Yes |
| Sparse Interpreter — mechanistic interpretability toolkit | Very high | ICLR / NeurIPS | Yes |

All four were viable. The final decision was to go with a project structurally similar to Harshith's — same detect-via-simulation methodology — but in a completely different and arguably more legally relevant domain.

### Why gig economy pricing collusion

Harshith's project focused on financial markets (limit order books). The analog problem in ride-share / gig economy markets is less explored academically, more directly tied to ongoing antitrust litigation (real lawsuits against Uber and delivery platforms have cited algorithmic pricing coordination), and the simulator is a contribution in itself — no public MARL ride-share pricing sandbox exists.

The research question crystallised as:

> *Can a temporal anomaly detector identify tacit collusion between independently-operating pricing algorithms in a simulated ride-share market, using only observable price and demand signals — without access to the agents' internal policies?*

**Project name: CartelSense.**

---

## Phase 1 — Understanding the Problem (Before Writing Code)

Before touching a keyboard, the full problem was mapped out conceptually.

### Key concepts learned

**Tacit collusion** — coordination that emerges from shared behavior without explicit communication. If multiple drivers all use the same third-party pricing app, they produce correlated prices without ever talking to each other. This is legally ambiguous (no explicit agreement exists) but economically identical to a cartel.

**Why existing detection fails** — current antitrust tools look at individual driver behavior. A single driver charging surge prices during rush hour is legitimate. The collusion signal only appears at the group level, in the *correlation structure* across drivers. No existing system detects this.

**The detection gap** — this is the core research contribution: cross-driver coordination signals as a detection primitive.

### Architecture designed upfront

```
Market Simulator
  → city grid (N×N zones)
  → Poisson demand engine
  → three driver agent types
  → four collusion modes
  → episode generator + labeler

Feature Extractor
  → per-driver time series features
  → cross-driver correlation features (the novel signal)
  → sliding window tensors

TCN Detector
  → Temporal Convolutional Network
  → binary classification per window
  → trained on labeled episode data

Validation Loop
  → honest RL agents
  → false positive rate measurement
  → ablation studies
```

### Four collusion modes defined

| Mode | What happens | Detection signal |
|---|---|---|
| Price Parallelism | All colluders post same price simultaneously | Low price dispersion, high pairwise correlation |
| Zone Avoidance | Group collectively avoids certain zones | Correlated offline status in specific zones |
| Surge Synchronization | Group goes offline together to manufacture surge | Synchronized offline windows, then correlated price spike |
| Phantom Scarcity | Group disappears when demand rises, returns at higher price | Correlated availability toggling |

---

## Phase 2 — Layer 1: The Simulator

### What was built

**6 Python files, ~600 lines total.**

```
cartelsense/
├── simulator/
│   ├── city.py        ← zone grid + demand engine
│   ├── market.py      ← market clearing
│   ├── episode.py     ← simulation loop + labeler
│   └── factory.py     ← scenario builder
├── agents/
│   ├── base_agent.py      ← abstract base class
│   └── colluding_agent.py ← Type B agents + shared algorithm
└── tests/
    └── test_simulator.py  ← 4 smoke tests
```

### city.py — The demand engine

**Design decisions made:**

*Grid size: 8×8 = 64 zones.* Small enough to run fast on CPU, large enough to have meaningful spatial structure. Zones represent neighborhoods.

*Timestep: 5 minutes. One day = 288 timesteps.* Matches real ride-share platform granularity.

*Demand: Poisson process.* Ride requests are random but arrive at a known average rate λ. Poisson is the standard model for arrival processes — it's what queuing theory is built on. λ varies by zone (downtown busier than suburbs) and time (rush hours busier than 3am).

*Three demand peaks:* Morning rush (t=78, ~6:30am), lunch (t=144, ~12pm), evening rush (t=210, ~5:30pm). Each modeled as a Gaussian bump added to a sinusoidal baseline. Realistic.

*Zone intensity map:* Center zones get 2.0× base demand, corner zones get 0.4× base demand. Implemented as a static multiplier grid computed at initialization.

*Transition matrix:* Where do rides go? Gravity model — rides are more likely to go to nearby zones and to the city center. Computed once and stored as a (64×64) matrix.

*Willingness to Pay: log-normal distribution.* Parameters: μ=log(120), σ=0.5. This gives median WTP ≈ ₹116, mean ≈ ₹129. Right-skewed — most passengers have moderate WTP, a small tail of high-WTP passengers (late for flights, don't care about cost). Log-normal chosen because: (a) prices can't be negative, (b) matches empirical consumer behavior data.

**What the numbers looked like after running:**
- Peak requests (t=78): 713 across the whole city
- Off-peak requests (t=10): 49
- WTP median: ₹116, mean: ₹129.5

The 14× difference between peak and off-peak is realistic and expected.

### agents/ — The agent architecture

**BaseAgent (abstract class):** Defines the interface every agent must implement — `set_price()` and `observe()`. Also handles bookkeeping: clipping prices to [₹60, ₹500], logging every timestep to `episode_log`, updating earnings.

The abstract class pattern is important here — it means the simulator doesn't care what kind of agent it's talking to. Honest agents, colluding agents, and future RL agents all look the same to the simulation loop. This is clean software design and makes adding new agent types easy.

**ColludingAgent (Type B):** The key design insight — all colluding agents hold a reference to the *same* `SharedPricingAlgorithm` object. They never communicate directly, but because they call `compute_action()` on the same object, they're reading the same internal state (including the shared global clock `_global_step`). This is exactly how real tacit collusion works: independent agents, shared algorithm.

**SharedPricingAlgorithm:** Implements all four modes. The surge multiplier formula:

```
multiplier = (demand / (competitors + 1)) ^ surge_sensitivity
price = base_price × multiplier
```

Clipped to [1.0×, 4.0×] range. Surge sensitivity = 1.8 (aggressive but not absurd).

Each mode adds behavioral logic on top of this base formula:
- Parallelism: adds ±2% noise (just enough to look organic)
- Zone avoidance: goes offline in designated zones
- Surge sync: uses `_global_step % 48 < offline_duration` to synchronize offline windows
- Phantom scarcity: triggers when `local_demand > 4.0` and sets shared `_offline_countdown`

### market.py — Market clearing

Simple and realistic: cheapest online driver in a zone gets matched to a ride request if their price ≤ passenger's WTP. One ride per driver per timestep. Unmatched requests are lost (passenger cancels).

This is a simplified first-price sealed-bid auction run once per timestep. It's not the only valid market model, but it's defensible and computationally trivial.

### episode.py — The simulation loop

Per timestep:
1. Tick shared algorithm clock
2. City generates requests
3. Each agent observes local state → sets price
4. Market clears
5. Agents receive feedback
6. Everything recorded to `TimestepRecord`
7. Label assigned (1 if collusion active, 0 otherwise)

**Label simplification:** Current labeling is binary at the episode level — either all timesteps are collusion or none are. A future refinement will make labels per-window (some windows might have more active collusion than others depending on mode).

### Test results

```
TEST 1: City demand generation         PASSED
  Peak requests (t=78): 713
  Off-peak requests (t=10): 49
  WTP median: ₹116.0, mean: ₹129.5

TEST 2: Full episode run               PASSED
  Price range: ₹84.9 – ₹500.0
  Mean price: ₹367.1

TEST 3: Collusion correlation signal   PASSED
  Colluding group avg pairwise correlation: 0.980
  Honest group avg pairwise correlation:   0.908

TEST 4: All four modes run             PASSED
  price_parallelism   | earnings: ₹42,128
  zone_avoidance      | earnings: ₹42,379
  surge_synchronization | earnings: ₹42,500
  phantom_scarcity    | earnings: ₹38,269
```

**Test 3 is the most important result so far.** Colluding drivers show 0.980 pairwise correlation vs 0.908 for honest drivers. The gap (0.072) is the signal the detector will learn to exploit. This validates the simulation design before any ML is written.

**Test 4 earnings breakdown** tells a story: phantom scarcity earns least (agents are offline a lot, missing rides), surge synchronization earns most (short offline windows, then elevated prices with high acceptance). This matches intuition.

### Problems faced

*Import errors on first run:* Python couldn't find modules. Fix: run with `python -m tests.test_simulator` from the `cartelsense/` root, and set `PYTHONPATH=.` in `.env`. The `-m` flag treats the invocation as a module, which handles relative imports correctly.

*Price ceiling hits:* Mean price of ₹367 seemed high. Investigation showed the surge multiplier can hit 4.0× during peak demand in downtown zones (high λ, few competitors). ₹120 × 4.0 = ₹480, which is near the ₹500 ceiling. This is expected behavior — surge pricing during peak in a low-supply zone is legitimate. No fix needed, but worth noting as a potential paper discussion point.

---

## What's Next

**Layer 2 — Feature Extractor:**
- Sliding window generator over episode data
- Per-driver features: price, online status, earnings rate
- Cross-driver features: pairwise price correlations, price dispersion (std dev), synchrony score, availability correlation
- Output: labeled tensor dataset ready for the TCN

**Layer 3 — TCN Detector:**
- 3-block dilated causal TCN
- Binary cross-entropy training
- Evaluation: precision, recall, F1 by collusion mode

**Layer 4 — Honest RL Agents (Type A):**
- Q-learning with local observation only
- Trained independently, no shared state
- Used for false positive rate validation

**Layer 5 — Ablation Studies:**
- Cross-driver vs per-driver features only
- Performance by collusion mode
- Group size sensitivity (what if only 2 of 10 drivers collude?)
- Window size sensitivity

---

*Last updated: Layer 1 complete — simulator passing all tests.*