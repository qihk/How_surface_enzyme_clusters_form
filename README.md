## Files

### `Urease-CG-generate.py`

Generates the coarse-grained (CG) representation of urease.

The atomic structure of urease is taken from the Protein Data Bank using:

- **PDB ID:** `6ZJA`

The script converts the urease structure into the coarse-grained representation used in the simulations.

---

### `Urease-AA-generate.py`

Generates the initial simulation configuration containing multiple urease molecules.

This script is used to place a large number of urease molecules in the simulation box and construct the initial configuration for subsequent molecular dynamics simulations.

---

### `colloid-structure.data`

Contains the atomic information of the spherical colloidal particle used as the base structure.

This file provides the particle coordinates and structural information required by the colloid-generation and modification scripts.

---

### `generate-random.py`

Generates a spherical colloidal particle with randomly distributed catalytic or reactive surface sites.

This configuration is used to represent a surface without spatial correlations in the distribution of active sites.

---

### `generate-nbinomial.py`

Generates a spherical colloidal particle with a heterogeneous surface distribution constructed according to a negative binomial distribution.

This configuration is used to reproduce a patchy surface organization with spatially heterogeneous catalytic-site distributions.

---

### `generate-Janus.py`

Generates a Janus-type colloidal particle.

Catalytic or reactive sites are restricted to one hemisphere of the particle, while the opposite hemisphere remains passive.

---

### `modify-colloid.py`

Modifies the original spherical colloid structure to generate a **surface-roughened colloidal particle model**.

The script introduces surface topographical heterogeneity into the initially smooth colloidal particle.

---

### `modify-colloid-hollow.py`

Generates a modified colloidal structure containing local hollow or concave surface features.

This script can be used to construct geometrically heterogeneous surfaces for studying the effect of local surface confinement on enzyme immobilization and spatial organization.

---

## Typical Workflow

A typical structure-generation workflow is:

1. Generate or prepare the urease model using `Urease-CG-generate.py`.
2. Construct the initial multi-enzyme simulation system using `Urease-AA-generate.py`.
3. Use `colloid-structure.data` as the base colloidal particle structure.
4. Generate the desired surface organization:
   - random distribution: `generate-random.py`
   - heterogeneous/patchy distribution: `generate-nbinomial.py`
   - Janus distribution: `generate-Janus.py`
5. Introduce surface geometrical heterogeneity when needed using:
   - `modify-colloid.py`
   - `modify-colloid-hollow.py`

## Requirements

The scripts are written in Python. Depending on the specific script, common scientific Python packages such as the following may be required:

- NumPy
- SciPy
- MDAnalysis or related molecular-structure libraries

Please check the individual scripts for exact dependencies.

## Structural Input

The coarse-grained urease model is based on the experimentally resolved urease structure:

- **PDB ID: 6ZJA**

The corresponding PDB structure should be downloaded and placed in the appropriate working directory before running `Urease-CG-generate.py`.

## Notes

These scripts were developed for constructing urease-functionalized colloidal models with different surface organizations and geometries, including random, patchy, Janus, rough, and locally confined surface configurations.
