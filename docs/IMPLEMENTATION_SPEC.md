# QUBOLens — Complete Implementation Specification

> Status: implemented in this repository. This document is intentionally exact
> enough for another engineering agent to reproduce or extend the project
> without guessing.

## 1. Product decision

**Problem:** ML teams often need a fixed input budget because features have
acquisition, preprocessing, storage, or latency cost. Ranking features
independently ignores duplicated signal.

**Product:** a web workbench and Python library that help people choose a
smaller, less repetitive feature set, compare it with simple alternatives,
visualize the search, and export the technical optimization problem.

**Positioning:** lead with the practical feature-selection benefit. Explain the
QUBO and quantum connection only after the user understands the job the product
does. Do not imply quantum speedup.

**Audience:** ML engineers, students learning quantum optimization, and
researchers who want an inspectable classical baseline.

## 2. Required user journey

1. The first viewport explains feature selection in plain language and offers a
   sample run immediately.
2. A seeded edge-telemetry classification run starts automatically.
3. The user selects a built-in scenario or uploads a CSV.
4. The user chooses exact budget \(k\), redundancy weight \(\lambda\), and
   search depth.
5. The API returns selection, baselines, Pareto data, annealing trace, matrix,
   export, runtime, insight, and caveat in one response.
6. The UI renders:
   - four headline metrics;
   - a plain-language observation;
   - budget frontier and energy trace;
   - selected feature chips and coefficient heat map;
   - like-for-like baseline table;
   - downloadable result and QUBO JSON.
7. The result always labels evaluation as diagnostic.

## 3. Stack and rationale

| Layer | Choice | Reason |
|---|---|---|
| Runtime | Python 3.11+ standard library | One-command start; tiny Render build; auditability |
| Server | `ThreadingHTTPServer` | Sufficient for a bounded stateless demonstrator |
| Solver | Custom seeded Metropolis annealer | Exposes algorithm; zero vendor lock-in |
| ML check | Small logistic/linear gradient models | Same evaluator for every subset; no binary wheels |
| Frontend | Semantic HTML + CSS + Canvas | No build tool; fast, responsive visual artifact |
| API | JSON over same origin | Easy CLI/web integration; no CORS/configuration burden |
| Deployment | Render Blueprint | Repository-to-service deployment with health check |
| CI | `unittest` + `compileall` | Works without package installation |

Do not add a database: uploads and results are intentionally ephemeral.

## 4. Mathematical contract

Given binary feature decisions \(x_i\), normalized relevance \(r_i\), absolute
pairwise correlation \(R_{ij}\), required size \(k\), redundancy weight
\(\lambda\), and cardinality penalty \(P\):

\[
E(x)=
-\sum_i r_i x_i
+\frac{\lambda}{\max(1,k-1)}\sum_{i<j}R_{ij}x_ix_j
+P(\sum_i x_i-k)^2.
\]

Use \(x_i^2=x_i\) to expand:

- diagonal \(Q_{ii}=-r_i+P(1-2k)\);
- upper-triangular pair
  \(Q_{ij}=\lambda R_{ij}/\max(1,k-1)+2P\);
- constant offset \(Pk^2\).

The solver may omit the constant while comparing states; exports retain it.
The unit test must compare matrix energy plus offset with the readable objective
for every binary state of a small problem.

The pair scale prevents redundancy magnitude from growing simply because a
larger budget has more pairs. Use \(P=2.5\) by default.

## 5. Solver contract

- Initialize each read with exactly \(k\) random active bits.
- Derive a geometric temperature schedule from maximum QUBO bias.
- Visit variables in a seeded random order each sweep.
- Accept \(\Delta E\leq0\), otherwise with probability
  \(\exp(-\Delta E/T)\).
- Track the best feasible state and approximately 48 trace checkpoints.
- Repair cardinality deterministically if required.
- Expose `fast`, `balanced`, and `deep` read/sweep budgets.
- The same seed and inputs must return the same mask and energy.

This is a reference solver, not a benchmark against optimized annealing
libraries or QPUs.

## 6. Data and evaluation contract

- CSV: UTF-8 header, 30–2,500 rows, 2–40 features, ≤2.5 MB.
- Missing numeric values: median. Missing categories: mode.
- Categories: deterministic ordinal encoding, reported in result notes.
- Constant features: dropped and reported.
- Classification: exactly two target labels, mapped deterministically to 0/1.
- Regression: numeric target.
- Relevance: absolute target correlation normalized by the maximum.
- Redundancy: absolute pairwise feature correlation.
- Baselines: top-\(k\) relevance and all features.
- Evaluation: same seeded folds and same linear learner for each fixed subset.
- Classification primary score: ROC AUC. Regression primary score: \(R^2\).
- Every response warns that post-selection CV is not an unbiased held-out
  estimate.

## 7. API contract

### `GET /api/health`

Returns service name, semantic version, and `status: ok`.

### `GET /api/datasets`

Returns built-in slugs, task, shape, and default budget.

### `POST /api/optimize`

Demo body:

```json
{
  "source": "demo",
  "dataset": "edge-failure",
  "k": 6,
  "redundancy_weight": 0.65,
  "quality": "balanced",
  "seed": 42
}
```

CSV body additionally provides `csv`, `target`, `task`, and `name`.

Response top-level keys are stable: `dataset`, `selection`, `benchmark`,
`insight`, `features`, `frontier`, `annealing`, `qubo`, `runtime`, `caveat`.

## 8. Repository acceptance checklist

- [x] End-to-end web application.
- [x] Library and CLI interfaces.
- [x] Two seeded built-in scenarios.
- [x] In-memory CSV workflow.
- [x] QUBO JSON export.
- [x] Three comparative visualizations.
- [x] Responsive, keyboard-accessible interface.
- [x] Deterministic unit and integration tests.
- [x] MIT license, contributing and security guidance.
- [x] GitHub Actions check.
- [x] Render Blueprint and health endpoint.
- [x] Concise shareable project brief.

Before a release, run:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q qubolens
python3 -m qubolens --demo edge-failure -k 6 --quality fast
```

Then start the server and check `/api/health` plus one `/api/optimize` response.

## 9. Research basis and honest interpretation

- Mücke et al. introduce a generalized QUBO for selecting a specified feature
  count from importance and redundancy, evaluated on classical, gate-model,
  and annealing backends: [paper](https://doi.org/10.1007/s42484-023-00099-z).
- Glover et al. derive general QUBO constraint-penalty constructions:
  [tutorial](https://arxiv.org/abs/1811.11538).
- Pranjic et al. formulate mutual-information feature selection as MIQUBO and
  demonstrate an equipment-price workflow:
  [preprint](https://arxiv.org/abs/2411.19609).
- Hellstern et al. emphasize dataset dependence and report classical stochastic
  methods outperforming noisy quantum runs in their experiments:
  [preprint](https://arxiv.org/abs/2306.10591).
- D-Wave’s documented sampler abstraction accepts QUBO mappings, supporting the
  replaceable-backend design:
  [sampler API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/api_ref.html).

The engineering conclusion is not “quantum wins.” It is:

> Formulating ML resource trade-offs as a portable binary energy model is
> useful now; future hardware can be evaluated against a reproducible
> baseline without changing the product contract.
