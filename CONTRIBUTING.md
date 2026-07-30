# Contributing

Thank you for improving QUBOLens. Small, evidence-backed changes are preferred.

1. Open an issue for a behavior change or new solver adapter.
2. Create a focused branch.
3. Add or update tests.
4. Run `python3 -m unittest discover -s tests -v`.
5. Run `python3 -m compileall -q qubolens`.
6. Open a pull request describing the user-facing change and any scientific
   assumptions.

Please do not describe a classical heuristic as a quantum algorithm or claim
speedup without a controlled benchmark. New relevance measures should document
data leakage, scaling, and evaluation implications.
