# Contributing

Thank you for improving QUBOLens. Keep each change focused and easy to review.

1. Open an issue for a behavior change or new search method.
2. Create a focused branch.
3. Add or update tests.
4. Run `python3 -m unittest discover -s tests -v`.
5. Run `python3 -m compileall -q qubolens`.
6. Open a pull request describing what changes for users and how it was tested.

New feature measures should explain how they handle data leakage, scaling, and
evaluation. Performance claims should include a reproducible comparison.
