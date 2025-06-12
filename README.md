# 🦇 Bat Algorithm

This repository provides a Python implementation of the **Bat Algorithm**, a nature-inspired metaheuristic for global optimization problems. Originally introduced by Xin-She Yang in 2010, the algorithm mimics the echolocation behavior of microbats to navigate the search space and find optimal or near-optimal solutions.

This implementation also includes a variant proposed by **Tawhid and Dsouza (2018)**, which introduces specific parameter settings and enhancements aimed at improving convergence behavior and performance on benchmark problems.

---

## 📖 Overview

The **Bat Algorithm (BA)** is a population-based optimization technique inspired by the echolocation behavior of bats. Bats use sound pulses to detect prey and obstacles, adjusting their flight and frequency based on feedback. This behavior is translated into a stochastic optimization process where artificial "bats" explore and exploit the search space to minimize (or maximize) an objective function.

**Key features:**
- Balances exploration and exploitation using loudness and pulse rate
- Incorporates frequency tuning for diverse search behavior
- Includes both the original **Yang (2010)** version and the enhanced **Tawhid & Dsouza (2018)** variant
- Can be applied to both continuous and combinatorial problems

---

## 📚 References

> **Yang, X.-S. (2010)**. A New Metaheuristic Bat-Inspired Algorithm. In: *Nature Inspired Cooperative Strategies for Optimization (NICSO 2010)*. Studies in Computational Intelligence, vol 284. Springer.  
> [📄 Springer Link](https://doi.org/10.1007/978-3-642-12538-6_6)

> **Tawhid, M. A., & Dsouza, K. (2018)**. An Improved Bat Algorithm Based on Directional Echolocation. *International Journal of Applied Metaheuristic Computing (IJAMC), 9*(3), 47–64.  
> [📄 Tawhid Version Link](https://www.emerald.com/insight/content/doi/10.1016/j.aci.2018.04.001/full/html)

---

## ⚙️ How It Works

Each bat in the population represents a candidate solution. The core update rules include:

1. **Frequency Update:** Each bat adjusts its frequency randomly between a min and max value.
2. **Velocity Update:** Bats move based on their current velocity and the position of the best bat found so far.
3. **Position Update:** New solutions are generated based on velocity and optionally via a local random walk.
4. **Loudness and Pulse Rate:** Control the probability of accepting new solutions and local search intensity.



---

## 🔧 Implementation Highlights

- Written in **Python** with **NumPy**
- Live progress display using the `rich` library
- Includes both **Yang (2010)** and **Tawhid & Dsouza (2018)** variants
- Allows for custom objective functions and parameter tuning
- Modular and easily extensible for benchmarking and experimentation

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/leodarkseid/bat-algorithm.git
cd bat-algorithm
```

### 2. Create Env

Create a virtual env and install requirements.txt with pip

### 3. RUN

#### TEST
To carry out comparative tests, just run 

```bash
python -m test
```

### 
To Run each version of the algorithm run either

```bash
python -m runTawhid
```

OR 

```bash
python -m runYang
```