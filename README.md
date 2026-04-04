<div align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python Version"/>
  <img src="https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react" alt="React"/>
  <img src="https://img.shields.io/badge/AI-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Simulation-SUMO-3B82F6?style=for-the-badge" alt="SUMO"/>
  
  <br/>
  
  <h1 align="center">🚦 QuMiks: AI-Driven Multi-Agent Traffic Control</h1>
  
  <p align="center">
    <strong>A high-performance deep reinforcement learning benchmark suite and live telemetry dashboard utilizing Recurrent QMIX variants mapped against real-world urban topologies.</strong>
  </p>
</div>

---

## 📖 Overview

This repository hosts a production-grade Intelligent Transportation System developed to dynamically alleviate metropolitan network congestion. Using Eclipse SUMO (Simulation of Urban MObility) as the deterministic physics engine, we bridge multi-agent vehicular traffic environments with advanced Deep Recurrent Neural Networks (PyTorch). 

The system provides a **Concurrent Dual-Engine execution**. It runs our fine-tuned QMIX memory-variant AI side-by-side with Native standard traffic controllers (Webster's Method) and streams real-time latency optimization graphs directly to a high-fidelity React dashboard via Fast-API WebSockets.

---

## 🛠 Features

- **Multi-Agent Deep RL (MARL):** Custom implementations of Recurrent Neural Agents parsing tensor boundaries across 8-lane macroscopic intersections.
- **Dual-Simulation Synchronization:** A highly parallelized `dual_runner` architecture that instantiates two separate TraCI threads on different RAM ports to emulate identical traffic volumes simultaneously without bottlenecking.
- **Real-World Topologies:** Complete OSM maps embedded directly into SUMO XML matrices (e.g., Connaught Place, India).
- **Streaming Telemetry Dashboard:** A beautifully dynamic React interface using WebSockets, `recharts`, and raw Canvas HTML5 alpha-blending for native map projections on boot.

---

## 📂 Project Architecture

```graphql
SIH/
├── config/                  # PyMARL standard configuration YAML definitions
│   └── improved_qmix.yaml   # Hyperparameters targeting RNNAgent optimization
├── maps/                    # SUMO native xml definitions
│   ├── connaught_place.net.xml  # Physical topology (Lanes, Junctions)
│   ├── connaught_place.rou.xml  # Dynamic vehicle route flows
│   └── *.sumocfg            # Environmental engine configurations
├── models/                  # Checkpoint storage
│   └── v2/                  # Compiled PyTorch weight states (.pt)
├── requirements.txt         # Comprehensive Python library dependencies
├── run_server.py            # Primary Backend Bootstrapper (Uvicorn / FastAPI)
└── src/                     # Core Logic Matrix
    ├── algos/               # AI Blueprints (Webster controllers & QMIX RNNAgents)
    ├── backend/             # FastAPI App, Dual-Runner engines, Socket integrations
    ├── core/                # TraCI PyMARL Environment definitions (sumo_env.py)
    └── frontend/            # Vite + React UI Dashboard
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisite: Eclipse SUMO Installation
This AI framework relies entirely on the Eclipse SUMO core physics engine. 
* **Windows/Linux/Mac**: Download and install Eclipse SUMO from [https://eclipse.dev/sumo/](https://eclipse.dev/sumo/)
* **CRITICAL**: You must add `SUMO_HOME` as an Environment Variable pointing to your installation directory (e.g., `C:\Program Files (x86)\Eclipse\Sumo`), and add the `/bin` folder to your System `PATH`.

### 2. Python Environment Setup
We strictly recommend isolating dependencies. 
```bash
# 1. Create and bind the virtual environment
python -m venv venv
venv\Scripts\activate  # (On Mac/Linux: source venv/bin/activate)

# 2. Inject libraries (Includes PyTorch, FastAPI, TraCI)
pip install -r requirements.txt
```

### 3. Frontend Environment Setup
The interface relies on TailwindCSS and Node modules.
```bash
# Enter the frontend matrix
cd src/frontend

# Install dependencies and compile
npm install
npm run build     # Use this for production compilation
npm run dev       # Or run this to launch the hot-reload environment
```

### 4. Launching the System
Once installed, the platform spins up natively from the root directory. This launches the Uvicorn webserver, parses the exact Connaught Place geometry to the API, and boots the socket listeners.

```bash
# Ensure you are in the root directory (SIH/)
python run_server.py

# Expected Output:
# INFO: Uvicorn running on http://0.0.0.0:8000
```
Navigate to `http://localhost:8000` (or your frontend Vite port) to view the telemetry UI. Clicking "Launch Engine" initializes the PyTorch CUDA/CPU buffers and launches both PyMARL TraCI endpoints seamlessly in the background.

---

## 📊 Benchmarking Modes
To bypass the GUI and run raw numerical analytical evaluations in the terminal:
```bash
# Run the Native traffic evaluator
python src/benchmark.py --mode native

# Run the Custom RNN (QMIX Variant) evaluation
python src/benchmark.py --mode v2
```

---

## 🏛 References & Citations

1. **Eclipse SUMO**: Lopez, P. A. et al. (2018). *Microscopic Traffic Simulation using SUMO*. IEEE Intelligent Transportation Systems Conference (ITSC).
2. **QMIX MARL Framework**: Rashid, T. et al. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*. ICML.
3. **TraCI Protocol**: Python TraCI implementation bindings.
4. UI Technologies: **React**, **Vite**, **FastAPI**, **Tailwind CSS**, **Recharts**.

***Produced by the SIH 2025 Intelligent Automation Team.***
