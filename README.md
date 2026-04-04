# SIH 2025: Advanced Traffic Management with QMIX & Indian Smart Standards

## 🚦 Project Overview
This project is a high-performance framework for benchmarking and training multi-agent reinforcement learning (MARL) controllers for urban traffic. It specifically targets the complex radial geometry of **Connaught Place, New Delhi**, using a modular approach to compare AI against traditional Indian engineering standards (IRC).

---

## 📂 Project Structure

```text
SIH/
├── src/
│   ├── core/           # SUMOEnv and shared utilities
│   ├── algos/          # Controller implementations
│   │   ├── indian/     # Webster's Method (IRC:93-1985)
│   │   ├── legacy/     # Standard QMIX (Rashid et al.)
│   │   └── v2/         # Recurrent QMIX (GRU + MaxPressure)
│   ├── config/         # YAML configurations for all modes
│   └── benchmark.py    # Main Entry Point (Train/Eval/Compare)
├── maps/               # Connaught Place network, route, and config
├── models/             # Saved weights (.pt files)
└── requirements.txt    # Base Python dependencies
```

---

## 🛠️ Installation & GPU Setup (NVIDIA 3050 Laptop)

To leverage your **NVIDIA RTX 3050** for training, follow these steps exactly:

1. **Activate your environment**:
   ```bash
   venv\Scripts\activate
   ```

2. **Uninstall CPU-only PyTorch**:
   ```bash
   pip uninstall torch torchvision
   ```

3. **Install CUDA-enabled PyTorch (v12.1)**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Remaining Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage: The Benchmark Suite

The `src/benchmark.py` script is the central command for all operations.

### **1. Run Comparison (Scorecard)**
Run all models (Native, Indian, Legacy, V2) back-to-back to generate a performance scorecard.
```bash
python src/benchmark.py --mode compare
```

### **2. Train a Model**
Training the **V2 (Recurrent)** model (calibrated for approx 2 hours on RTX 3050):
```bash
python src/benchmark.py --mode v2 --task train
```

### **3. Individual Evaluation**
Test a specific controller (e.g., the Indian Standard Webster Controller):
```bash
python src/benchmark.py --mode indian --task eval
```

---

## 📈 Controller Descriptions

| Mode | Type | Description |
|---|---|---|
| **NATIVE** | Baseline | The default fixed-time signal program from the original map file. |
| **INDIAN** | Engineering | Dynamically calculates optimal cycles using **Webster's Method** (IRC:93-1985 & IRC:106-1990). |
| **LEGACY** | RL | Standard QMIX with 3-layer MLP agents and a monotonic mixer. |
| **V2 (CURRENT)** | RL | Upgraded QMIX using **GRU cells** for memory and **MaxPressure** for network-wide throughput. |

---

## 📝 License
Licensed under the MIT License. Developed for the Smart India Hackathon (SIH) 2025.
