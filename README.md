# 🧬 BitLife Evolution

An evolutionary algorithm for evolving small, bit-based creatures. The fitness function can be dynamically changed during the evolutionary process itself, simulating how natural evolution adapts to shifting environments.

## About

Creatures are 2D patterns on a grid, made up of 1s (alive) and 0s (dead), evolving through Conway's Game of Life rules. The key idea is that evolutionary pressure is not fixed, it can change mid-run, forcing populations to re-adapt in real time. This allows us to explore how natural evolution works through algorithms.

The simulation environment is powered by [Seagull](https://github.com/ljvmiranda921/seagull), a Python library for Conway's Game of Life.
