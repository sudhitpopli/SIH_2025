# QMIX Complete Guide: Advanced Mechanics & Implementation Status

This expanded guide covers the heavy-lifting mathematics and the neural logic that allows QMIX to learn complex coordination in urban traffic networks like **Connaught Place**.

---

## 🚀 CURRENT STATUS: v2 (Recurrent Architecture)

The project has been upgraded from simple MLPs to a **v2 Recurrent Architecture** (GRU-based). 

### key Features of v2:
1. **Persistent Memory**: Traffic flows are non-Markovian. A GRU layer allows the agent to remember history even when the current "snapshot" (observation) is partially blocked.
2. **MaxPressure Reward**: Instead of simple wait time, we optimize for **Throughput** using the pressure metric ($Incoming - Outgoing$). This maximizes vehicle flow across the entire grid.
3. **Hypernetwork Conditioning**: The central mixer is now conditioned on the global state $s$, allowing it to dynamically adjust the contribution of each agent based on the overall traffic load.

---

## 1. How does it predict "Future Q"?

### The Math: The Target Value
To update our current prediction ($Q_{curr}$), we calculate a **Target Value** ($y$):
$$y = R + \gamma \cdot \max_{a'} Q_{target}(s', a')$$

- **$s'$**: The state we just landed in.
- **$\max_{a'} Q_{target}(s', a')$**: We ask our network: *"In this new state $s'$, what is the absolute best success score I could get in the next step?"*
- **Recursive Logic**: By doing this millions of times, the "True Success" found in terminal states (end of traffic jam) "flows backward" through time to the very first step.

---

## 2. Why pull 32 Random Transitions? (I.I.D Sampling)

### The "Stale Memory" Problem
If we used the $Q_i$ values from when the transition actually happened, we would be learning from an "Old, Stupid" version of the agent.

1. **Re-calculating**: When we pull an old memory from the buffer, we **re-run** our **Current Net** on that old state. This tells us: *"Based on how smart I am NOW, what should I have done THEN?"*
2. **Breaking Correlations**: Consecutive steps in a simulation are too similar. Picking 32 random moments ensures a balanced, stable education for the AI, preventing it from "overfitting" onto a single traffic cycle.

---

## 3. Adam Optimizer: The Smart Navigator
Adam is the "Smart Driver" of the learning process. Unlike a basic gradient descent that always moves at a fixed rate, Adam:
- **Uses Momentum**: If the weights have been moving in a certain direction for a while, it "speeds up" that change.
- **Is Adaptive**: It keeps a separate "Learning Rate" for every weight in the network. If a weight is naturally sensitive, it nudges it gently. If a weight is stuck, it gives it a bigger push.

---

## 4. Backpropagation: Command & Control
Mathematically, $Q_{tot}$ is a function of the Mixer weights ($W$), which are themselves a function of the State ($s$) through a **Hypernetwork**.
$$Q_{tot} = \text{Mixer}\left( Q_i, \text{HyperNet}(s) \right)$$

### The Chain Rule
During backpropagation, the "Error" flows backward:
$$\frac{\partial \text{Error}}{\partial \text{HyperNet Weights}} = \frac{\partial \text{Error}}{\partial Q_{tot}} \cdot \frac{\partial Q_{tot}}{\partial \text{Mixer}} \cdot \frac{\partial \text{Mixer}}{\partial \text{HyperNet}}$$

- **Chain of Command**: The Error at $Q_{tot}$ is a feedback signal. It passes down through the Mixer (the managers) to the Hypernetwork (the staff) who decided how the Mixer should behave for that specific state.

---

## 5. Benchmarking Against the Indian Standard (Webster's)
For the SIH presention, we compare our AI against **IRC:93-1985 (Webster's Formula)**.

### The Algorithm:
1. **Critical Ratio (Y)**: $Y = \sum (q_i / S_i)$, where $q$ is flow and $S$ is saturation flow.
2. **Optimum Cycle ($C_o$)**: $C_o = (1.5L + 5) / (1 - Y)$.
3. **Splits**: Green time is apportioned proportionally to the critical lane flow in each phase.

**Our AI is considered "Successful" if its average reward significantly exceeds the Webster-calculated baseline.**
