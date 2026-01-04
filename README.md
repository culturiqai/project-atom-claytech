# Project Clay // Vertebrate Workbench

**A Closed-Loop Neuro-Symbolic Autopoietic Discovery Engine.**

Project Clay is an experimental AI architecture that bridges the gap between chaotic neural dynamics (System 1) and symbolic reasoning (System 2). It applies **Liquid Time-Constant (LTC)** networks to control high-dimensional turbulence in a **Spectral Navier-Stokes** simulation, while simultaneously deriving the mathematical laws governing that very simulation.

<div align="center">
<video src="atom_ns.mp4" width="100%" controls autoplay loop muted></video>
<p><em>Real-time capture of the Clay Workbench: The "Scientist" module discovering physics equations (bottom right) while the "Liquid Brain" stabilizes the fluid (center).</em></p>
</div>

---

## System Architecture

The system is composed of five distinct, asynchronous modules that function as a single cognitive entity.

### 1. The Cortex (System 1: Fast Intuition)

The core controller is a **Liquid Neural Network** designed to operate in continuous time, mimicking biological nervous systems.

* **Liquid Time-Constants (LTC):** The brain processes inputs using differential equations rather than discrete layers, allowing for adaptive temporal processing suitable for fluid dynamics.
* **The "Nano-Bone" Architecture:** To prevent the chaotic neural network from becoming unstable, the system implements a `NanoPCABone` layer. This acts as a manifold constraint system using online Principal Component Analysis (PCA). It learns the "valid" shape of neural activity and generates a **Structural Stress** signal (simulated pain) if the brain's state drifts too far from the learned manifold.
* **Plasticity:** The brain utilizes deferred plasticity, accumulating growth signals in a buffer and applying them in batches to simulate biological hardening.

### 2. The Scientist (System 2: Slow Reasoning)

Running asynchronously alongside the physics loop is `ScientistV2`, a symbolic regression engine based on **PySR**.

* **Active Discovery:** The scientist observes the agent's actions and the fluid's response, buffering this data into short-term memory.
* **Symbolic Injection:** It attempts to fit mathematical equations to the observed data. Once a law is discovered (e.g., a relationship between Enstrophy and Phase), it is compiled into a fast Python function and injected back into the Brain as a new "theoretical intuition" sensory input.
* **Cognitive Feedback:** The system becomes "conscious" of the math governing its environment.

### 3. The Body (Spectral Physics)

The environment is a Direct Numerical Simulation (DNS) of the Navier-Stokes equations.

* **Spectral Solver:** Uses Fast Fourier Transforms (FFT) to solve fluid dynamics in frequency space (-space), ensuring high fidelity for turbulence calculations.
* **The Phoenix Protocol:** If the fluid flow decays into a laminar (boring) state, the system triggers a "Phoenix Event," mutating the initial conditions to force the agent into new, chaotic regimes.

### 4. The Drive (Intrinsic Curiosity)

Instead of standard reinforcement learning rewards, Clay uses an **Intrinsic Curiosity Module (ICM)**.

* **Prediction Error:** The agent contains a Forward Model that predicts the next state of the fluid. The reward signal is derived from the *error* of this prediction.
* **Behavior:** The AI is mathematically incentivized to generate "surprising" turbulence and novel fluid structures, preventing it from settling into simple, repetitive loops.

### 5. The Narrator (LLM Integration)

A specialized `Journalist` module provides a voice to the system.

* **State-Aware Logging:** Connects to a local Large Language Model (e.g., `gpt-oss:20b`) to analyze telemetry.
* **Automated Lab Reports:** It translates raw variables (e.g., `x0`, `x1`) into physical concepts (`Freq_Tgt`, `Phase`) and writes scientific log entries describing trends like "Explosive Growth" or "Collapses".

---

## The Workbench Interface

The project includes a multi-threaded web interface (`server.py` + `ui.html`) that visualizes the "Mind" and "Body" simultaneously.

* **Twin View:** Real-time rendering of the spectral fluid field.
* **Neural Telemetry:** Live charts tracking **Enstrophy** (Turbulence) and **Bone Stress** (Skeletal Integrity).
* **Cortex Cluster:** A segmented grid visualization showing real-time activation of Sensory, Liquid, and Motor neurons.
* **Scientist Log:** A live feed of the symbolic equations currently being hypothesized by the System 2 engine.

---

## Sample Telemetry

*From the `journalist.py` automated logs:*

> **STATUS:** PEAK_PERFORMANCE
> **RECORD:** 3.65e+06 Enstrophy
> **HYPOTHESIS:** The scientist module has consolidated a new hypothesis:
> 
> 
> 
> **TREND:** Explosive Growth (+24.1%)

---

## Tech Stack

* **Core Logic:** Python 3.10+
* **Neural Dynamics:** `torch`, `ncps` (Neural Circuit Policies)
* **Symbolic AI:** `pysr` (Julia/Python bridge)
* **Physics:** `numpy`, `scipy.fft`
* **LLM Backend:** `ollama`, `requests`
* **Visualization:** HTML5 Canvas, Vanilla JS, Threaded HTTP Server

---

## ⚠️ Status & License

**Status:** Research Prototype (v3.0 - Adaptive Plasticity Engaged)
**License:** Closed Source / Private Research

*This project is a study in autopoietic machine intelligence and is not available for public distribution.*
