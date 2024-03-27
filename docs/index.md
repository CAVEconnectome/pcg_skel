---
title: About
---

`pcg_skel` is a package used to rapidly build neuronal skeletons from electron microscopy data in the [CAVE ecosystem](https://github.com/CAVEconnectome).
It integrates structural data, connectivity data, and local features stored across many aspects of a CAVE system, and creates objects as [MeshParty](https://meshparty.readthedocs.io/en/latest/) meshes, skeletons, and MeshWork files for subsequent analysis.
By harnessing the way the structural data is stored, you can build skeletons for even very large neurons quickly and with little memory use.
Actual skeletonization is done with functionality within MeshParty, while `pcg_skel` focuses on data retrieval and integration across many CAVE services.

## Installation

To install `pcg_skel`, just use pip. The package is available for python 3.9 and above.

```bash
pip install pcg_skel
```

## Installing from Source (for Developers)

To install `pcg_skel` from source, you can clone the repository and install it using pip.

```bash
pip install -e .
```

`pcg_skel` uses [Hatch](https://hatch.pypa.io/latest/) for packaging, and there are tests that can be run within the environment using:

```bash
hatch run test:run_tests
```
