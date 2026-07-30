# QUBOLens technical guide

This guide explains how the project is organized, how feature selection works,
and where to make changes.

## Project overview

QUBOLens chooses exactly \(k\) features by balancing:

- connection with the target;
- overlap between selected features;
- the feature limit chosen by the user.

Every method is checked with the same data splits and model. The web app shows
the result, the comparison, and the search behavior in one place.

## User flow

1. Open an included dataset or upload a supported data file.
2. Choose how many features to keep.
3. Run the prepared default search.
4. Review the answer, chosen inputs, and comparison.
5. Open optional settings or technical charts only when needed.

The edge-failure example runs automatically when the page opens.

## Technology

| Area | Implementation |
|---|---|
| Runtime | Python 3.11+ |
| Web server | Python `ThreadingHTTPServer` |
| Search | Seeded Metropolis simulated annealing |
| Model checks | Small logistic and linear models |
| Interface | HTML, CSS, JavaScript, and Canvas |
| API | JSON endpoints served from the same application |
| Tests | Python `unittest` |
| Deployment | Render Blueprint |

The application has no runtime package dependencies. Uploaded files and results
stay in memory.

## Repository map

```text
quantum/
├── qubolens/
│   ├── core.py          feature statistics, QUBO construction, search
│   ├── data.py          included datasets and upload preparation
│   ├── evaluate.py      model training and comparison scores
│   ├── pipeline.py      complete feature-selection workflow
│   ├── server.py        web server, static files, and JSON API
│   ├── cli.py           command-line interface
│   └── web/
│       ├── index.html   page structure and accessible controls
│       ├── styles.css   layout, responsive design, and animation
│       ├── app.js       interactions, API calls, and charts
│       └── og.png       repository and social preview
├── tests/               unit and end-to-end tests
├── docs/                project brief and this guide
├── render.yaml          Render service definition
└── pyproject.toml       package metadata and CLI entry point
```

## Selection method

For feature decisions \(x_i\), target relevance \(r_i\), feature overlap
\(R_{ij}\), requested size \(k\), overlap weight \(\lambda\), and size penalty
\(P\), QUBOLens uses:

\[
E(x)=
-\sum_i r_i x_i
+\frac{\lambda}{\max(1,k-1)}\sum_{i<j}R_{ij}x_ix_j
+P(\sum_i x_i-k)^2.
\]

The three parts reward useful features, discourage repeated information, and
keep exactly the requested number of features.

The exported upper-triangular matrix follows:

\[
E(x)=\sum_{i\leq j}Q_{ij}x_ix_j+\text{offset}.
\]

Its coefficients are:

- \(Q_{ii}=-r_i+P(1-2k)\)
- \(Q_{ij}=\lambda R_{ij}/\max(1,k-1)+2P\)
- offset \(=Pk^2\)

The default size penalty is \(P=2.5\). Dividing the overlap term by
\(\max(1,k-1)\) keeps its scale more consistent across feature limits.

## Search behavior

The search starts with exactly \(k\) selected features. During each pass, it
tries changes in a seeded random order and accepts improvements immediately.
It can also accept temporary setbacks early in the run, which helps it leave
weak local solutions.

The `fast`, `balanced`, and `deep` modes change the number of attempts and
steps. The same dataset, settings, and seed return the same result.

The search is intentionally small and inspectable. The exported matrix can be
used later with another compatible optimizer.

## Data preparation

Uploads support:

- CSV, TSV, delimited TXT, JSON, JSONL, NDJSON, and XLSX;
- UTF-8, UTF-16, or Windows-1252 text;
- files up to 20 MB;
- up to 100 source columns and 40 prepared inputs;
- a repeatable sample of up to 5,000 rows for large files;
- numeric, date, categorical, and free-text inputs;
- binary classification or numeric regression targets.

Missing numeric values use the column median. Dates become numeric time values.
Low-cardinality categories become indicator columns. Free text becomes
interpretable length, word-count, vocabulary-variety, digit-share, and
common-keyword measures. Likely identifiers, empty columns, and constant
columns are removed and reported in the result. If preparation creates more
than 40 inputs, the 40 strongest target links continue to the interactive
selection.

Target connection is measured with absolute correlation and scaled to
\([0,1]\). Feature overlap is the absolute pairwise correlation.

## Result comparison

QUBOLens compares three choices:

1. the feature set found by QUBOLens;
2. the top \(k\) features ranked individually;
3. every available feature.

All three use the same folds and model settings. Classification reports ROC
AUC, accuracy, and log loss. Regression reports \(R^2\), mean absolute error,
and root mean squared error.

These are exploration scores. A final feature set should be confirmed on
separate data.

## JSON API

### Health

```http
GET /api/health
```

### Included datasets

```http
GET /api/datasets
```

### Run feature selection

```http
POST /api/optimize
Content-Type: application/json
```

Example:

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

The browser first sends the encoded file and optional target to
`POST /api/inspect`. The response describes the detected format, row count,
columns, target type, and prepared input count.

Upload optimization requests use `source: "upload"` and add `file` (base64),
`filename`, `target`, `task`, and `name`. The earlier `source: "csv"` request
shape remains available for compatibility.

The response contains dataset details, selected features, comparisons, charts,
the reusable matrix, timing, a plain-language observation, and a validation
note.

## Run and test

Start the web application:

```bash
python3 -m qubolens.server
```

Run all checks:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q qubolens
```

Run from the command line:

```bash
python3 -m qubolens --demo edge-failure -k 6 --quality fast
```

## Safe extension points

- Add datasets in `qubolens/data.py`.
- Add result fields in `qubolens/pipeline.py` and document them here.
- Add another search method behind the same matrix input.
- Add nonlinear feature measures while keeping the comparison splits fixed.
- Add a scikit-learn wrapper as an optional package extra.

## Research basis

- Mücke et al. describe fixed-size feature selection using importance and
  overlap:
  [paper](https://doi.org/10.1007/s42484-023-00099-z).
- Glover, Kochenberger, and Du explain QUBO constraint construction:
  [tutorial](https://arxiv.org/abs/1811.11538).
- Pranjic, Mummaneni, and Tutschku apply a related formulation to an equipment
  price workflow:
  [preprint](https://arxiv.org/abs/2411.19609).
- Hellstern, Dehn, and Zaefferer show why comparison with established methods
  matters:
  [preprint](https://arxiv.org/abs/2306.10591).
- D-Wave documents the QUBO input format supported by its sampler interface:
  [documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/api_ref.html).
