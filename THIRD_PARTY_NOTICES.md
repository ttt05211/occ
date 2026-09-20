# Third-party dependencies and notices

No dataset, model checkpoint, OccFM source tree, or third-party binary is vendored in this repository.

Core dependencies:

| Dependency | Declared minimum | License family | Purpose |
|---|---:|---|---|
| NumPy | 1.24 | BSD-3-Clause | array/geometry/metrics |
| SciPy | 1.10 | BSD-3-Clause | connected components and frozen majority-fill reference |
| PyTorch | 2.1 | BSD-3-Clause | learned model/training |
| PyYAML | 6.0 | MIT | configs |

Optional dependencies:

| Dependency | Declared minimum | License family | Purpose |
|---|---:|---|---|
| nuscenes-devkit | 1.1.11 | Apache-2.0 | nuScenes metadata/annotations/splits |
| Matplotlib | 3.7 | Matplotlib/PSF-style | visualization |
| pytest | 8 | MIT | tests |

The extraction environment used NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0+cpu, PyYAML 6.0.3, and pytest 9.0.2 for local CPU tests. `nuscenes-devkit` was not installed in that environment, so real-data tests were not run there.

Dataset licenses/terms are not granted by this repository. Users must obtain nuScenes/Occ3D legally and comply with their respective terms.
