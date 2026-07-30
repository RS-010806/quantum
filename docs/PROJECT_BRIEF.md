# QUBOLens

## Feature selection, made visual

> **Find the strongest features. Drop the noise. See what changes.**

### What it is

QUBOLens is an open-source feature selection lab. A user uploads a CSV or opens
a sample dataset, chooses how many inputs to keep, and gets a clear result that
balances:

1. **predictive relevance** — keep signals related to the target;
2. **diversity** — avoid paying twice for nearly identical inputs;
3. **deployment budget** — keep exactly the requested number of features.

Behind the friendly interface, the choice is encoded as a **Quadratic
Unconstrained Binary Optimization (QUBO)** problem. A built-in classical search
solves it immediately, and the same technical matrix can be exported for other
compatible solvers.

### Why it is useful

Real ML systems often pay for inputs before the model makes a prediction:
sensors, database joins, API calls, preprocessing, memory, and latency. Feature
selection is therefore an efficiency decision, not just a statistics exercise.

A simple top-\(k\) ranking judges every feature alone. It can keep two strong
signals that repeat the same information. QUBOLens considers **pairs** as well
as individual features, making that trade-off explicit and visible.

| User gets | Why it matters |
|---|---|
| Exact feature budget | Maps to an operational constraint |
| Result in seconds | Fast enough to explore different limits |
| Fair comparisons | Shows whether the chosen set actually helps |
| Score-by-size chart | Shows where extra inputs stop helping |
| Search trace and interaction map | Makes the optimization inspectable |
| Downloadable technical matrix | Preserves a path to other solvers |
| CLI, Python API, and web UI | Works in a demo, notebook, or pipeline |

### The idea behind the project

My takeaway is that quantum ideas can be useful before quantum hardware is.
The practical first step is **a clearer way to frame a difficult choice**.

QUBO turns a fuzzy question - "which signals are worth keeping together?" -
into a precise optimization problem. It can be tested with the included search
today and moved to quantum or hybrid solvers later. The useful artifact is a
reproducible problem, comparison, and trade-off.

The live experiment asks whether considering both usefulness and overlap finds
a stronger fixed-size feature set than ranking each feature alone. It shows
both outcomes—even when the answer is “they are essentially tied.”

### How it works

```text
data → relevance/redundancy graph → technical matrix → repeatable search
  └────────────────────────── model checks + comparisons ────────────┘
                                  ↓
             comparison · selected signals · exportable matrix
```

\[
E(x) = -\sum_i r_i x_i
+ \lambda\sum_{i<j} R_{ij}x_ix_j
+ P(\sum_i x_i-k)^2
\]

Low energy means high relevance, low duplication, and the correct budget.
QUBOLens uses a repeatable search and fixed data splits for comparison. Uploads
remain in memory. The whole app runs as one Python service and deploys directly
from the repository.

### Why the result is easy to inspect

- The algebra is unit-tested against the readable objective for every state of
  a small QUBO.
- Runs are seeded end to end.
- QUBOLens, simple ranking, and all-feature methods use the same data splits.
- The interface clearly says the scores are for exploration, not final
  production validation.
- The repository includes tests, a health check, CI, deployment configuration,
  a CLI, and an MIT license.
- The method follows published QUBO feature-selection formulations,
  including [Mücke et al.](https://doi.org/10.1007/s42484-023-00099-z), the
  constraint constructions summarized by
  [Glover et al.](https://arxiv.org/abs/1811.11538), and recent MIQUBO work by
  [Pranjic et al.](https://arxiv.org/abs/2411.19609).

### Important limits

Correlation is not the whole story; these comparison scores are not a held-out
production estimate; and the included search is classical, not quantum
computation. Those are explicit boundaries, not footnotes.

**Repository:** [github.com/RS-010806/quantum](https://github.com/RS-010806/quantum)

**License:** MIT · **Local requirement:** Python 3.11+ · **Deployment:** Render
