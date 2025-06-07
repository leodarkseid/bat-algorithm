# 🦇 Bat Algorithm (Yang, 2010)

This repository provides a Python implementation of the **Bat Algorithm**, a nature-inspired metaheuristic for global optimization problems. Originally introduced by Xin-She Yang in 2010, the algorithm mimics the echolocation behavior of microbats to navigate the search space and find optimal or near-optimal solutions.

---

## 📖 Overview

The **Bat Algorithm (BA)** is a population-based optimization technique inspired by the echolocation behavior of bats. Bats use sound pulses to detect prey and obstacles, adjusting their flight and frequency based on feedback. This behavior is translated into a stochastic optimization process where artificial "bats" explore and exploit the search space to minimize (or maximize) an objective function.

**Key features:**
- Balances exploration and exploitation using loudness and pulse rate
- Incorporates frequency tuning for diverse search behavior
- Can be applied to both continuous and combinatorial problems

---

## 📚 Reference

> **Yang, X.-S. (2010)**. A New Metaheuristic Bat-Inspired Algorithm. In: Nature Inspired Cooperative Strategies for Optimization (NICSO 2010). Studies in Computational Intelligence, vol 284. Springer.

[📄 DOI Link (Springer)](https://doi.org/10.1007/978-3-642-12538-6_6)

---

## ⚙️ How It Works

Each bat in the population represents a candidate solution. The core update rules include:

1. **Frequency Update:** Each bat adjusts its frequency randomly between a min and max value.
2. **Velocity Update:** Bats move based on their current velocity and the position of the best bat found so far.
3. **Position Update:** New solutions are generated based on velocity and optionally via a local random walk.
4. **Loudness and Pulse Rate:** Control the probability of accepting new solutions and local search intensity.

These mechanisms help the algorithm converge toward the global optimum while avoiding premature convergence.

---

## 🔧 Implementation Highlights

- Written in **Python** with **NumPy**
- Live progress display using the `rich` library
- Allows for custom objective functions and parameter tuning
- Modular and easily extensible for benchmarking and experimentation

---
## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/leodarkseid/bat-algorithm.git
cd bat-algorithm