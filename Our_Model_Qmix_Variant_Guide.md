# 🚦 Our Model (QMIX Variant) - Complete Architectural Guide

This document provides a comprehensive breakdown of **Our Model (QMIX Variant)** tailored for the Intelligent Traffic Management system governing the Connaught Place ecosystem in SUMO. 

It explains the mathematical formulation, the recurrent neural architectures, the integration with Eclipse SUMO, and how the Deep Multi-Agent Reinforcement Learning (MARL) paradigm drastically reduces waiting times across the grid compared to Native Webster controllers.

---

## 🚀 1. What is QMIX?

**QMIX** is a state-of-the-art Deep Multi-Agent Reinforcement Learning (MARL) algorithm designed for cooperative environments where multiple discrete agents (in our case, traffic lights) must work together to optimize a **global reward** (total grid efficiency and reduced waiting time).

### Why a "Variant"?
Standard tabular Q-Learning fails in continuous, multi-agent systems because the state space grows exponentially. QMIX solves this using **Centralized Training with Decentralized Execution (CTDE)**. 
In *Our Variant*, each traffic junction possesses a **Recurrent Neural Network (RNN)** acting as its "brain." While the agents learn cooperatively using an underlying mixing network to factorize the joint Q-Value during the training phase, during inference (execution on the dashboard), they operate completely autonomously using their localized observations and hidden memory states.

---

## 🧠 2. The Recurrent Neural Architecture (RNNAgent)

Unlike classical feed-forward networks that only see a single "snapshot" of the road, traffic is fundamentally a *temporal* sequence. Knowing that a cluster of cars is approaching requires memory.

Our model uses PyTorch to build a bespoke **RNNAgent** (`src/algos/v2/networks.py`) equipped with hidden recurrent memory loops (GRU/LSTM cells):

1. **Input Layer (Observation Space):** 
   - Dynamically generated from the `MultiAgentEnv` wrapper (`env.obs_size`). 
   - Variables include **queue length**, **vehicle density**, and **active phase trackers** mapped across up to 8 incoming lanes.
2. **Hidden Layer 1 & 2 (Recurrent):** 
   - These interconnected parameters pass temporal information back to themselves through time (the dotted blue loops seen in the frontend visualization). 
   - They allow the agent to "remember" traffic waves that entered the grid minutes ago, predicting exactly when they will reach the central junction.
3. **Output Layer (Action Logits):** 
   - Yields individual Expected Q-Values mapping to 4 discrete phase actions (e.g., *Extend Green Phase A*, *Transition Phase A $\rightarrow$ B*). 
   - The system utilizes an `argmax` function to physically trigger the highest-value action via the TraCI API.

---

## 📡 3. Eclipse SUMO & TraCI Integration

The bridge between the PyTorch Neural calculations and the deterministic physics of the vehicles occurs inside the `SUMOEnv` class (`src/core/envs/sumo_env.py`).

### The Control Loop Sequence:
1. **Perception**: At Step $T$, Python natively maps the SUMO XML geometries, counting cars waiting at intersections to populate the input vector.
2. **Forward Pass**: The normalized PyTorch Tensors are evaluated inside the recurrent GPU matrices (`preview_sumo.py` / `dual_runner.py`).
3. **Actuation**: If an agent requests a phase transition, `SUMOEnv` enacts extreme precision:
   - It intercepts the raw binary action.
   - It forces a **3-second Yellow Transition Phase** mathematically into the engine.
   - It physically executes the `sumo.simulationStep()` to advance the clock precisely according to the temporal config.
4. **Reward Assignment**: The environment yields a reward functionally mapped to $- \sum(\text{Accumulated Waiting Time})$, forcing the model to intrinsically prioritize clearing heavily congested arteries.

---

## ⚡ 4. Real-time Telemetry & Execution 

When deployed inside the **Dual-Simulation Frontend Dashboard**, the model operates simultaneously alongside the legacy SUMO native logic (Webster's timing method). 

The `dual_runner` spawns highly independent, asymmetric threads targeting separate RAM instances:
* `v2_env.step(actions)` commands the PyTorch RNN engine.
* `native_env.step(None)` permits the raw C++ physics to dictate flow.

The metrics are intercepted precisely at identical timestamps, pushed completely asynchronously through the FastAPI `WebSocket`, and graphed organically across the Recharts module in raw temporal steps. 

### Why Our Model Wins
By anticipating incoming flows rather than reacting rigidly to fixed countdown timers, the RNN safely clusters green-light priorities, reducing "Stop-and-Go" waves drastically. This translates logically to the massive efficiency divergence demonstrated natively on the Cumulative Wait Time dashboard graphs. 

---

<div align="center">
  <em>Developed for the SIH 2025 Deep Traffic Management Benchmark</em>
</div>
