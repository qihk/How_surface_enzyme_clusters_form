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

### `modify-colloid.py` and `modify-colloid-hollow.py`

Modifies the original spherical colloid structure to generate a **surface-roughened colloidal particle model**.

The script introduces surface topographical heterogeneity into the initially smooth colloidal particle.

Analysis scripts are available upon reasonable request.
