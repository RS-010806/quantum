# QUBOLens

### Find the strongest features. Drop the noise. See what changes.

QUBOLens is an open-source feature selection lab. Choose how many model inputs
you want to keep, and it finds a useful set with less repeated information.
Every result is visual, downloadable, and compared with simple alternatives in
the same polished interactive workbench.

Under the hood, the feature choice is written as a Quadratic Unconstrained
Binary Optimization (QUBO) problem and solved with a built-in classical search.
That makes the project useful now while preserving a clear connection to
quantum optimization.

**No account, external service, or runtime dependency is required.**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/RS-010806/quantum)

[Read the polished two-page project brief](output/pdf/QUBOLens-Project-Brief.pdf)
or the [Markdown edition](docs/PROJECT_BRIEF.md).

## The idea

My takeaway is simple: **quantum ideas can be useful before quantum hardware
is**. The practical first step is learning to turn a messy choice - which
features are worth keeping together? - into a clear optimization problem that
can be tested and compared.

The project makes that idea tangible without making a speedup claim.

## What people can do

- Run two deterministic, realistic synthetic ML scenarios immediately.
- Upload a CSV (binary classification or regression, up to 40 features).
- Set an exact feature budget and tune the redundancy pressure.
- Watch the search improve as it tries different feature mixes.
- Compare QUBOLens with a simple ranking and the full feature set.
- Inspect the quality/size frontier, search trace, chosen inputs, and feature
  interaction map.
- Download the complete result or the technical optimization matrix.
- Run the same workflow from Python or the command line.

Uploaded data is parsed in memory and never written to disk.

## Why feature selection?

Feature budgeting is a small but real ML systems problem. Fewer inputs can mean
less acquisition, preprocessing, storage, and inference work. Selecting them is
combinatorial: with 18 signals and a budget of 6, there are 18,564 possible
subsets. A relevance-only ranking can keep two sensors that say nearly the same
thing. QUBOLens gives that interaction a quadratic cost.

The implemented objective is:

\[
E(x) =
-\sum_i r_i x_i
+ \frac{\lambda}{\max(1,k-1)}\sum_{i<j} R_{ij}x_ix_j
+ P\left(\sum_i x_i-k\right)^2
\]

where \(x_i\) is a binary “keep” decision, \(r_i\) is normalized target
correlation, \(R_{ij}\) is absolute pairwise feature correlation, \(\lambda\)
controls redundancy pressure, and the last term enforces exactly \(k\)
features.

The cardinality penalty is expanded into linear and quadratic coefficients.
The exported mapping uses the unambiguous upper-triangular convention
\(E(x)=\sum_{i\leq j}Q_{ij}x_ix_j+\mathrm{offset}\).

## Quick start

Python 3.11+ is the only requirement.

```bash
python3 -m qubolens.server
```

Open <http://localhost:8000>. The first seeded experiment runs automatically.

Run it headlessly:

```bash
python3 -m qubolens --demo edge-failure -k 6 --quality fast
python3 -m qubolens --csv measurements.csv --target outcome -k 8 \
  --quality balanced --output result.json
```

Use the library:

```python
from qubolens import load_csv_dataset, optimize_dataset

dataset = load_csv_dataset(
    open("measurements.csv", encoding="utf-8").read(),
    target_name="outcome",
)
result = optimize_dataset(dataset, k=8, redundancy_weight=0.65, seed=42)

print(result["selection"]["names"])
print(result["qubo"]["export"])
```

Call the API:

```bash
curl -s http://localhost:8000/api/optimize \
  -H 'content-type: application/json' \
  -d '{"source":"demo","dataset":"edge-failure","k":6,
       "redundancy_weight":0.65,"quality":"fast","seed":42}'
```

## Architecture

```text
CSV or built-in scenario
        │
        ▼
in-memory preprocessing ──► relevance + redundancy profile
        │                                  │
        │                                  ▼
        │                          cardinality QUBO
        │                                  │
        │                                  ▼
        │                       seeded Metropolis search
        │                                  │
        └────────► fixed-subset CV ◄────────┘
                         │
                         ▼
        baselines + frontier + portable QUBO JSON
                         │
                         ▼
                responsive browser workbench
```

The backend is Python standard library only:

- `qubolens/core.py` — statistics, QUBO encoder, Metropolis annealer.
- `qubolens/data.py` — seeded scenarios and bounded in-memory CSV handling.
- `qubolens/evaluate.py` — dependency-free logistic/ridge-style diagnostics.
- `qubolens/pipeline.py` — selection, baselines, frontier, result contract.
- `qubolens/server.py` — threaded JSON API and static asset server.
- `qubolens/web/` — semantic HTML, responsive CSS, and Canvas visualizations.

## Reproducibility and checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q qubolens
```

Seeds control scenario generation, solver initialization, annealing order, and
fold assignment. The test suite verifies the QUBO algebra against the readable
objective for every state of a small problem, solver cardinality and
determinism, CSV behavior, and the end-to-end export contract.

## Deploy on Render

The repository includes a `render.yaml` Blueprint. The button above creates one
free Python web service, runs compilation and tests during the build, starts
`python -m qubolens.server`, and checks `/api/health`. Render can deploy directly
from a linked Git repository and recognizes Blueprint configuration from the
repository root ([Render Blueprint reference](https://render.com/docs/blueprint-spec)).

## Scientific grounding

QUBOLens is an implementation project informed by prior work:

1. Mücke et al. formulate fixed-size feature selection as a generalized QUBO
   balancing importance and redundancy and evaluate it on classical and quantum
   backends: [*Feature Selection on Quantum Computers*](https://doi.org/10.1007/s42484-023-00099-z).
2. Glover, Kochenberger, and Du explain how constraints such as cardinality can
   be represented with QUBO penalty functions:
   [*A Tutorial on Formulating and Using QUBO Models*](https://arxiv.org/abs/1811.11538).
3. Pranjic, Mummaneni, and Tutschku apply an MIQUBO formulation to an ML
   pipeline and a used-equipment forecasting case:
   [*Quantum Annealing based Feature Selection in Machine Learning*](https://arxiv.org/abs/2411.19609).
4. Hellstern, Dehn, and Zaefferer report that outcomes are dataset-dependent
   and that classical stochastic optimization remained superior in their
   studied noisy settings—an important counterweight to hype:
   [*Quantum computer based Feature Selection in Machine Learning*](https://arxiv.org/abs/2306.10591).
5. The export shape mirrors the `sample_qubo` abstraction used by D-Wave’s
   sampler interface:
   [D-Wave sampler API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/api_ref.html).

## What this project does **not** claim

- The built-in search is classical; the quantum connection is the problem
  formulation and its compatibility with other solvers.
- Post-selection cross-validation is a diagnostic comparison, not an unbiased
  generalization estimate. Confirm a subset on held-out data.
- Correlation-based relevance does not capture every nonlinear or conditional
  interaction.
- A lower QUBO energy does not guarantee a better downstream model.
- The interactive cap is intentional. Larger problems need screening,
  decomposition, or a more optimized sampler.

Those boundaries are part of the product.

## Next useful extensions

- A sampler adapter protocol for D-Wave, QAOA, simulated bifurcation, and MILP.
- Nested-CV mode for stricter model-selection estimates.
- Per-feature acquisition or preprocessing costs with slack-variable budgets.
- scikit-learn-compatible `TransformerMixin` wrapper as an optional extra.
- Multiclass relevance and nonlinear dependency measures.

See [the implementation specification](docs/IMPLEMENTATION_SPEC.md) for exact
engineering decisions and [the project brief](docs/PROJECT_BRIEF.md) for the
concise shareable story.

## License

MIT. Contributions and careful negative results are welcome.
