# The Absolute Beginner's Guide to QMIX

If you have never studied Reinforcement Learning, Neural Networks, or Traffic Simulation... fear not. This guide will explain **QMIX** from the ground up using plain English.

---

## 1. What is Reinforcement Learning (RL)?

Imagine trying to teach a dog to fetch a ball. 
* You don't write a mathematical formula for how the dog's legs should move.
* Instead, you throw the ball. If the dog brings it back, you give it a **treat (positive reward)**. If it runs away, you do nothing (or give a **negative reward**).
* Over time, the dog *learns* the optimal behavior just by maximizing its treats.

This is exactly how AI works in our traffic simulation.
* **The Agent:** The AI controlling the traffic light.
* **The Action:** Choosing whether the light should be Green or Red.
* **The Environment:** The SUMO Traffic physics simulator.
* **The Reward:** If cars move fast, the AI gets +5 points. If traffic jams, it gets -10 points. 

Instead of programming an algorithm like "If 10 cars are waiting, turn green," the AI just rapidly tries millions of random combinations until it figures out the math that earns the absolute highest score. 

## 2. The Problem with Multiple Agents

Teaching one dog is easy. But what if you have 8 dogs (8 traffic lights) in Connaught Place, and they all have to chase the same ball without crashing into each other?

This is called **Multi-Agent Reinforcement Learning (MARL)**.

If every traffic light acts selfishly to clear its own immediate cars, it might accidentally dump 50 cars into the very next intersection, causing a massive gridlock. We need the traffic lights to successfully cooperate. 

But how do you make 8 different AI brains work together?

---

## 3. Enter QMIX (The Master Conductor)

**QMIX** is the algorithm our project uses to solve this exact problem. Think of it like a musical orchestra. 

### Part A: The Local Brains (The Musicians)
Every single traffic light has its own small AI brain (A Neural Network). 
Because a traffic light doesn't have cameras matching the entire city, it can only see the cars immediately waiting at its own stopping lane. This is called **Partial Observability**. 
* Light #1 calculates: "Based on my 10 waiting cars, I think Phase 2 gets a score of **Q(local) = 5**."
* Light #2 calculates: "Based on my empty road, I think Phase 1 gets a score of **Q(local) = 1**."

### Part B: The Mixing Network (The Conductor)
If the lights just blindly acted on those scores, it would be chaos.
QMIX introduces a central "God-View" brain called the **Mixing Network**. 

Instead of letting the dogs run loose, the Mixing Network takes all the local scores from all 8 traffic lights and feeds them into a massive blender. It looks at the **Global State** (a map of the entire city's traffic) and mixes them together to output one final, unified score: **$Q_{total}$**. 

### The Secret Magic: "Monotonicity"
QMIX forces a very strict mathematical rule: **The Conductor can only use Positive weights (Absolute values).**

Why does this matter?
Because it ensures that if Traffic Light #1 makes an action that improves its local score, it *guarantees* that the total global score goes up. A local light is mathematically incapable of taking an action that benefits itself while hurting the city. They are literally forced into perfect cooperation.

---

## 4. Why V2? (Adding Memory)

In the codebase, you'll see a `legacy` folder and a `v2` folder.

In **Legacy QMIX**, the AI has severe amnesia. It wakes up every 5 seconds, looks at the cars, makes a decision, and immediately forgets everything that just happened. If a wave of 100 cars passed by one minute ago, it has no idea.

In **V2 (Recurrent QMIX)**, we upgraded the local brains with a **GRU (Gated Recurrent Unit)**. 
A GRU is literally an artificial memory cell. When the AI looks at the road, it combines what it sees right *now* with a compressed memory of everything it saw over the last 10 minutes. 

Because traffic flows in "waves", an AI with a memory can realize: *"Ah, intersection A just dumped 50 cars. I can't see them yet, but I remember this pattern from 2 minutes ago. I need to turn my light Green 10 seconds from now to catch the wave."*

## 5. How it Learns (BPTT)
To train an AI with memory, you can't just teach it random 5-second fragments. 

**BPTT** stands for *Backpropagation Through Time*. When our SUMO simulation finishes a 15-minute episode, PyTorch scoops up that entire chronological, continuous 15-minute video of traffic. It plays the tape backward (Backpropagating), mathematically calculating exactly at what second the AI made a mistake, and adjusting the weights of the brain so it never makes that mistake again.
