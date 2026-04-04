# QMIX Complete Guide: Advanced Mechanics & Optimization

This expanded guide covers the heavy-lifting mathematics and the neural logic that allows QMIX to learn complex coordination.

---

## 1. How does it predict "Future Q"?
Neural networks don't have a crystal ball. They use **Bootstrapping**—a method of using current estimates to improve future ones.

### The Math: The Target Value
To update our current prediction ($Q_{curr}$), we calculate a **Target Value** ($y$):
$$y = R + \gamma \cdot \max_{a'} Q_{target}(s', a')$$

- **$s'$**: The state we just landed in.
- **$\max_{a'} Q_{target}(s', a')$**: We ask our network: *"In this new state $s'$, what is the absolute best success score I could get in the next step?"*
- **Recursive Logic**: By doing this millions of times, the "True Success" found in terminal states (end of traffic jam) "flows backward" through time to the very first step.

---

## 2. Why pull 32 Random Transitions?
**User Question**: *"Can it not use the $Q_i$ it calculated before?"*

### The "Stale Memory" Problem
If we used the $Q_i$ values from when the transition actually happened, we would be learning from an "Old, Stupid" version of the agent.
1. **Re-calculating**: When we pull an old memory from the buffer, we **re-run** our **Current Net** on that old state. This tells us: *"Based on how smart I am NOW, what should I have done THEN?"*
2. **Breaking Correlations**: Consecutive steps in a simulation are too similar. If we only learned from the last 1 second, the network would "hallucinate" that the whole world looks like that 1 second. Picking 32 random moments (i.i.d. sampling) ensures a balanced, stable education for the AI.

---

## 3. What is Adam? (Adaptive Moment Estimation)
Adam is the "Smart Driver" of the learning process. Unlike a basic car that always moves at 10km/h, Adam:
- **Uses Momentum**: If the weights have been moving in a certain direction for a while, it "speeds up" that change.
- **Is Adaptive**: It keeps a separate "Learning Rate" for every single weight in the network. If a weight is very sensitive, it nudges it gently. If a weight is stuck, it gives it a bigger push.

---

## 4. Backpropagation & The Chain Rule
**User Question**: *"How does error change weights determined by another network?"*

Mathematically, $Q_{tot}$ is a function of the Mixer weights ($W$), which are themselves a function of the State ($s$) through a **Hypernetwork**.
$$Q_{tot} = \text{Mixer}\left( Q_i, \text{HyperNet}(s) \right)$$

### The Chain Rule
During backpropagation, the "Error" flows backward using the **Chain Rule** of calculus:
$$\frac{\partial \text{Error}}{\partial \text{HyperNet Weights}} = \frac{\partial \text{Error}}{\partial Q_{tot}} \cdot \frac{\partial Q_{tot}}{\partial \text{Mixer Weights}} \cdot \frac{\partial \text{Mixer Weights}}{\partial \text{HyperNet Weights}}$$

- Imagine a chain of command. The Error at $Q_{tot}$ is like a general's feedback. It passes down through the Mixer (the managers) to the Hypernetwork (the staff) who actually "decided" how to dress the Mixer for that specific state.

---

## 5. How to Beat the Default SUMO Model
The research paper you added (**Rashid et al., 2018**) highlights exactly how to improve:

### A. Add GRUs (Recurrent Memory)
Traffic is a **time-series**. An MLP only sees a "snapshot."
- **Benefit**: A GRU layer allows the light to remember: *"Wait, that car has been sitting there for 3 cycles already,"* even if the current snapshot only shows one car.
- **Action**: Modify `AgentQNetwork` to include `nn.GRU`.

### B. Reward Shaping (MaxPressure)
Instead of just negative wait time, use the **Pressure** metric from the paper:
$$\text{Pressure} = \text{Cars coming In} - \text{Cars going Out}$$
By minimizing pressure, the AI focuses on **Throughput** (clearing as many cars as possible) rather than just being "sad" that cars are waiting.

### C. Prioritized Replay
Instead of picking 32 transitions at **Random**, pick the ones where the Prediction Error was the **Largest**.
- **Logic**: Don't spend time re-learning things you already know perfectly. Focus on the "surprising" mistakes (like sudden jams).
