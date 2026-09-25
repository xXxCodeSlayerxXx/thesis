# The Three-Dimensional Open Dimension Problem

**Synthesizing Strategies for Efficient Bin Packing**

Bachelor thesis in Artificial Intelligence — Vrije Universiteit Amsterdam, June 2026

**Author:** William Mooijer  
**Supervisor:** Daan van den Berg  
**Second reader:** Bob Borsboom

The full thesis is in [`thesis.pdf`](thesis.pdf). This repository contains the simulation environment, packing algorithms, experiment harness, datasets, and results that the thesis is built on.

---

## Overview

This project studies the **three-dimensional rectangular open dimension problem (3D-RODP)**: packing a complete order of rectangular boxes onto a single pallet so that **stack height (max-z) is minimized**. Unlike classical 3D bin packing (fit items into as few bins as possible), the pallet footprint is fixed and height is the open dimension.

The work uses **real box types and orders from Centraal Boekhuis**, the main logistical service provider of Dutch-language books. Every millimeter of pallet height saved has direct cost and environmental impact in that setting.

A custom pallet simulator is used to compare three classical methods against a **branch-and-bound (BnB)** searcher that combines:

- Extreme-point candidate generation (Crainic et al.)
- BFD warm-start as an initial upper bound
- Bounding rules (trivial max-z bound; tall-low remaining-box bound)
- Symmetry breaking
- A **deduplication** filter
- A **top-X candidate limiter** — a per-node beam-search variant introduced in this work

On the generated test set, BnB improved stack height over best-fit decreasing in every difficulty class (**31.6 mm on average**, **45.6 mm** on the largest orders). Relinquishing the exact optimality guarantee cut the search tree by a factor of ~44 while sacrificing only ~2% packing height. Top-X limiting (default `X = 5`) reduced the remaining tree to about **4.5%** of its pre-filter size, making all 40 industrial orders tractable.

---

## Repository map

```
.
├── thesis.pdf                      Full bachelor thesis
├── boxoptimizer.ipynb              Interactive packing environment (primary)
├── boxoptimizer_export.py          Same code as a runnable Python script
├── data_processing.ipynb           Order generation, CSV collation, result analysis
├── environment.yml                 Conda environment
├── bnb_performance_metrics.csv     Ablation log of BnB development
├── bnb_structure_archive.txt       Deprecated BnB rules/filters and historical structure
├── data/
│   ├── boxtypes.csv                10 industrial box types (mm)
│   ├── orders.csv                  40 Centraal Boekhuis orders (30–40 boxes)
│   └── orders_test.csv             Generated test orders (ID encodes box count)
└── results/
    ├── figures/                    Thesis figures
    ├── pallet_images/              Per-order heightmap + 3D + metrics composites
    ├── algo_comparisons/           Random / FFD / BFD / BnB
    ├── optg_comparisons/           BnB with vs without optimality guarantee
    ├── topx_comparisons/           BnB across top-X limits (1–15 and unlimited)
    └── old_results/                Earlier experimental runs kept for provenance
```

Each comparison directory follows the same layout:

| Subfolder | Contents |
|---|---|
| `given_orders/` | Per-order CSVs for the 40 industrial orders (`O1`–`O40`) |
| `test_orders/` | Per-order CSVs for generated orders (`T1000`–…) |
| `collated/` | Concatenated files grouped by box-count band (`1000–1099`, …) |

Incomplete collated files are marked `_incomplete.csv`.

---

## Problem setup

| Parameter | Value |
|---|---|
| Pallet size (X × Y × Z) | 1000 × 1400 × 1400 mm |
| Objective | Minimize max stack height |
| Items | Orthogonal cuboids; 10 named types, 9 unique dimension triples |
| Rotations | Horizontal and vertical, on by default |
| Stability | ≥ 70% of a box’s footprint must rest on support |
| Industrial orders | 40 orders, 30–40 boxes each |
| Test orders | Generated partitions of *n* boxes over the 10 types; order ID `n × 100 + k` |

Wäscher classification: **3D rectangular open dimension problem**. Dyckhoff code of the industrial instance: **3/V/D/R**.

---

## The packing environment

All simulation is done in [`boxoptimizer.ipynb`](boxoptimizer.ipynb), along with its respective python script export [`boxoptimizer_export.py`](boxoptimizer_export.py).

### `Pallet`

A millimeter-resolution heightmap plus:

- Incremental **place / undo** (`place_box` returns a delta; `remove_box` restores it) so BnB can backtrack without copying the pallet
- **Extreme points** as the discrete candidate set (origin projections of box edges)
- Placement validity: in-bounds, support percentage, no overlap
- Metrics and visualisation (2D heightmap, 3D bar plot, composite results figure)

### Metrics

| Metric | Direction | Meaning |
|---|---|---|
| **Max-z** | minimize | Primary objective: height of the packed stack |
| Volume utilization | maximize | Occupied volume / (footprint × max-z) |
| Centre of gravity Z | minimize | Volume-weighted height (mass proxy) |
| Packing score | maximize | Height-weighted layer fill (floor counts more than the top) |
| Order fulfilment | maximize | Fraction of the order actually placed |
| Area at z = 0 | — | Pallet-floor coverage |

### Algorithms

| Key | Method | Role |
|---|---|---|
| `RANDOM` | Random (x, y) attempts per box | Lower-bound baseline; used to study attempt-budget vs fulfilment |
| `FFD` | First-fit decreasing | Greedy: largest-first, first valid candidate. Poor on 3D-RODP (tends to stack in a corner) |
| `BFD` | Best-fit decreasing | Greedy: largest-first, choose the placement that best improves the chosen metric. Strong heuristic and BnB warm-start |
| `BNB` | Branch and bound | Recursive search over extreme points and orientations |

`process_order(order_id, algo, …)` is the single entry point for packing one order.

### Branch and bound (in brief)

1. Sort boxes (default: decreasing volume).
2. Warm-start with BFD to get a tight upper bound on max-z.
3. Recurse over remaining boxes. At each node:
   - **Rule 1 (trivial):** prune if current height already ≥ best complete height.
   - **Rule 4 (tall-low):** prune if the tallest remaining box, in its flattest orientation at its lowest feasible extreme point, would already exceed the bound.
   - **Filter 5 (top-X):** keep only the X best (orientation, extreme-point) pairs, ranked by resulting top-z then x+y. Off when the optimality guarantee is on.
   - **Filter 1 (deduplication):** keep one representative of equivalent (orientation, landing-z) profiles. Off under the guarantee.
   - **Filter 2 (symmetry breaking):** for consecutive identical dimension triples, skip lexicographically smaller placements than the previous box.
4. Reconstruct the pallet from the best placement sequence.

Deprecated experiments (volume bound, look-ahead bound, heightmap-hash filter, two-surface filter) are archived in [`bnb_structure_archive.txt`](bnb_structure_archive.txt). The measured effect of each addition and removal is in [`bnb_performance_metrics.csv`](bnb_performance_metrics.csv).

---

## Experiments

Four experiment drivers live in the optimizer notebook/script. Analysis and figure generation live in [`data_processing.ipynb`](data_processing.ipynb).

| Function | What it measures | Output |
|---|---|---|
| `run_random_fulfillment_test` | Fulfilment vs random attempt budget | `results/figures/random_fulfillment_test-*.png` |
| `run_optimality_guarantee_test` | Exact vs heuristic BnB (height and tree size) | `results/optg_comparisons/` |
| `run_topx_limiting_test` | Top-X = 1…15 vs unlimited | `results/topx_comparisons/` |
| `run_algorithm_comparison_test` | Random / FFD / BFD / BnB head-to-head | `results/algo_comparisons/` |

`data_processing.ipynb` also:

- Generates the test-order CSV (100 random 10-way partitions per box count)
- Concatenates per-order CSVs into box-count bands
- Finds missing order IDs after long batch runs
- Computes extreme metric gaps between algorithm pairs
- Builds the thesis figures from collated CSVs

Headline results (see Chapter 5 of the thesis, and `results/figures/`):

- BnB beat BFD in **every** box-count class; 735 test orders saved **≥ 50 mm**.
- Keeping the optimality guarantee grew the tree **~44×** for a **~2%** reduction in height.
- Top-X = 5 was chosen as the default: large tree reduction, ~50% of the unlimited-run match rate that X = 10 achieves.

---

## Setup

```bash
conda env create -f environment.yml
conda activate scriptie
```

Dependencies: Python, NumPy, pandas, matplotlib, tqdm, ipykernel, ipywidgets.

Open `boxoptimizer.ipynb` or run the export:

```bash
python boxoptimizer_export.py
```

The script detects notebook vs terminal (progress bars, display) automatically.

---

## Usage

### Pack a single order

In the notebook, the last cell is the testing area. Typical settings:

```python
current_order_dict = orders_dict       # or test_orders_dict
current_orderID    = 15
current_algo       = Algorithm.BNB     # RANDOM | FFD | BFD | BNB
current_criterion  = Criterion.VOLUME  # LENGTH | WIDTH | HEIGHT | AREA | VOLUME
current_metric     = Metric.MAX_Z      # PACKING_SCORE | VOLUME_UTILIZATION | COG_Z | MAX_Z
```

Then:

```python
pallet, stats = process_order(
    current_orderID,
    algo=current_algo,
    criterion=current_criterion,
    order_dict=current_order_dict,
    metric=current_metric,
)
pallet.get_pallet_results(
    current_algo, current_orderID, current_order_dict,
    print_mode=True,   # show heightmap + 3D + metrics
    save_mode=True,    # write a composite PNG under results/
    bnb_stats=stats if current_algo == Algorithm.BNB else None,
)
```

### Global knobs

Defined near the top of the optimizer:

| Setting | Default | Effect |
|---|---|---|
| `PALLET_DIMS` | `(1000, 1400, 1400)` | Pallet millimeters |
| `BNB_OPTIMALITY_GUARANTEE` | `False` | Exact search (disables top-X and dedup) |
| `BNB_TOPX_DEFAULT_LIMIT` | `5` | Candidates kept per BnB node |
| `SUPPORTED_AREA_PERCENTAGE` | `70` | Minimum supported footprint |
| `HOR_ROTATION_ALLOWED_DEFAULT` / `VER_ROTATION_ALLOWED_DEFAULT` | `True` | Allowed orientations |
| `DEFAULT_MAX_ATTEMPTS` | `2000` | Random placer attempt cap |

### Batch experiments

Flags in the testing cell (`testing_random_fulfillment`, `testing_optg_comparisons`, `testing_topx_comparisons`, `testing_algo_comparisons`) run the corresponding driver. The comparison tests can be dispatched with `ProcessPoolExecutor` over missing order IDs.

After a batch, run the concatenation / missing-order / plotting cells in `data_processing.ipynb`.

---

## Data notes

`data/boxtypes.csv` columns: `ID`, `NAME`, `LENGTH`, `WIDTH`, `HEIGHT` (mm). Types 8 (`IM_DOOS_GROOT`) and 10 (`IM_EDU_GROOT`) are dimensionally identical; symmetry breaking treats them as interchangeable.

Order CSVs are wide: `order_id, amt_1, …, amt_10`. Test-order IDs are `box_count * 100 + index` (so `1523` is a 15-box instance). Industrial orders are IDs 1–40.

---

## Citation

If you use this repository, please cite the thesis:

> Mooijer, W. (2026). *The Three-Dimensional Open Dimension Problem: Synthesizing Strategies for Efficient Bin Packing*. Bachelor thesis, Vrije Universiteit Amsterdam.

```bibtex
@thesis{mooijer2026odp,
  author  = {Mooijer, William},
  title   = {The Three-Dimensional Open Dimension Problem: Synthesizing Strategies for Efficient Bin Packing},
  school  = {Vrije Universiteit Amsterdam},
  year    = {2026},
  type    = {Bachelor thesis},
}
```
