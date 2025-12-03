# EE 289Q HW3 – Traveling Salesman Problem Solver

This repository contains my implementation of the TSP heuristic solver for EE 289Q HW3.  
The solver evaluates many candidate cycles using a combination of **Nearest Neighbor** initialization and **2-opt local search**, under a strict time limit of ~55 seconds per graph.

---

## 📂 Files Included

- `tsp_solver.py` — main solver (Python)
- `solution_<916006596>.txt` — Best cycle found for Graph A and Graph B
- `report.pdf` — Written report (overview, flowchart, results)

---

## 🧠 Summary of Results

### Graph A
- Best cost: 2529.79
- Runtime: ~55s
- Cycles evaluated: 1.090e+08

### Graph B
- Best cost: 393.34
- Runtime: ~55s
- Cycles evaluated: 1.030e+08

---

## 📄 Algorithm Overview
- Load graph from edge-list format  
- Construct multiple **Nearest Neighbor tours** starting from random nodes  
- Improve each tour with **2-opt local search**  
- Track:
  - best tour 
  - number of cycles evaluated
  - cost improvements
- Stop improving once **time limit** is reached

---



