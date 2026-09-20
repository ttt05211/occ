# Local extraction validation — 2026-09-20

Environment available to this extraction run:

- Python 3.13.5
- Linux x86_64
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- SciPy 1.17.0
- PyYAML 6.0.3
- pytest 9.0.2
- CUDA/GPU: unavailable
- nuScenes-devkit + dataset: unavailable
- outbound package download: unavailable

Executed:

```text
python -m compileall -q src tools tests                         PASS
pytest -q                                                        PASS (18 passed)
python -m pip wheel . --no-deps --no-build-isolation ...         PASS
pip install -e . --no-deps --no-build-isolation                  PASS in the provided environment
python tools/{check_data,prepare_data,train,evaluate,benchmark_runtime}.py --help   PASS
personal-path / credential-pattern scan                          PASS (no matches)
```

A truly isolated virtual environment could not install PyTorch/SciPy/PyYAML because this runner has no outbound package access. This is recorded as an environment limitation, not an install pass. GitHub Actions performs the normal online clean installation path.

Real checkpoint, real-window, full-validation and CUDA runtime acceptance remain `NOT RUN`; see `docs/REPRODUCTION.md`.
