import numpy as np
import random

Box_x = 30.0
Box_y = 30.0
Box_z = 30.0

AtomTypes = 5
Nsurface = 2700
Number_Type2 = int(Nsurface * 0.32)

Nsolvent = int(Box_x * Box_y * Box_z * 10)
MinDist_Surface = 0.8
R_colloid = 3.0

File_Colloid_Data = "colloid-structure.data"
Output_LAMMPS_Data = "CP-lammps.data"

def Read_Colloid_Data(filename):
    atoms = []
    center_x = Box_x / 2
    center_y = Box_y / 2
    center_z = Box_z / 2
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() == '':
                continue
            x, y, z, *_ = map(float, line.strip().split())
            atoms.append([x + center_x, y + center_y, z + center_z])
    return np.array(atoms, dtype=float)

atoms = Read_Colloid_Data(File_Colloid_Data)

def Choose_Type2(atoms,
                 Number_Type2,
                 min_cutoff=0.3,
                 max_cutoff=0.3,
                 cluster_fraction=1.0,
                 r_size=1.57,
                 p_size=0.27,
                 max_clusters=1000):
    n_total = atoms.shape[0]
    n_candidate = min(Nsurface, n_total)
    atom_indices = np.arange(n_candidate)
    selected_indices = set()

    center = np.array([Box_x / 2, Box_y / 2, Box_z / 2], dtype=float)
    rel = atoms[atom_indices, :3] - center
    r = np.linalg.norm(rel, axis=1)
    cos_theta = rel[:, 2] / (r + 1e-8)
    weights = 1.0
    weights = np.clip(weights, 0, None)
    weights /= np.sum(weights)

    cluster_target = int(Number_Type2 * cluster_fraction)
    if cluster_target <= 0:
        return list(np.random.choice(atom_indices, Number_Type2, replace=False))

    actual_clusters = []

    while cluster_target > 0 and len(actual_clusters) < max_clusters:
        raw_size = np.random.negative_binomial(int(r_size), float(p_size)) + 1
        target_size = min(raw_size, cluster_target)

        unselected = np.array([i for i in atom_indices if i not in selected_indices])
        if unselected.size == 0:
            break
        w = weights[unselected]
        w /= np.sum(w)
        start = np.random.choice(unselected, p=w)
        cluster_set = set([start])
        selected_indices.add(start)
        growth_centers = [start]

        local_cutoff = np.random.uniform(min_cutoff, max_cutoff)

        while len(cluster_set) < target_size and growth_centers:
            new_growth_centers = []
            for center_idx in growth_centers:
                if len(cluster_set) >= target_size:
                    break
                dists = np.linalg.norm(atoms[atom_indices, :3] - atoms[center_idx, :3], axis=1)
                mask_unselected = np.array([idx not in selected_indices for idx in atom_indices])
                neighbors = atom_indices[(dists <= local_cutoff) & mask_unselected]
                np.random.shuffle(neighbors)
                for n in neighbors:
                    if len(cluster_set) >= target_size:
                        break
                    cluster_set.add(n)
                    selected_indices.add(n)
                    new_growth_centers.append(n)
            growth_centers = new_growth_centers

        actual_clusters.append(len(cluster_set))
        cluster_target -= len(cluster_set)

    current_selected = len(selected_indices)
    if current_selected < Number_Type2:
        remaining = Number_Type2 - current_selected
        unselected = np.array([i for i in atom_indices if i not in selected_indices])
        additional = np.random.choice(unselected, remaining, replace=False)
        selected_indices.update(additional)

    return list(selected_indices)

selected_indices = Choose_Type2(atoms, Number_Type2)

def Assign_Atom_Properties(atoms, selected_indices):
    num_atoms = len(atoms)
    atom_types = np.ones(num_atoms, dtype=int)
    for index in selected_indices:
        atom_types[index] = 2
    atom_types[Nsurface:] = 3
    return atom_types

atom_types = Assign_Atom_Properties(atoms, selected_indices)

def Add_Solvent_Particles(
    atoms,
    atom_types,
    nsolvent,
    min_dist_surface,
    sphere_radius,
    max_trials_per_particle=100000,
    print_interval=1000
):
    surface_atoms = atoms[:Nsurface]
    solvent_atoms = []

    center = np.array([Box_x / 2, Box_y / 2, Box_z / 2], dtype=float)
    sphere_r2 = sphere_radius ** 2
    min_dist2 = min_dist_surface ** 2

    print("Start adding solvent particles...")

    for n in range(nsolvent):
        placed = False

        for trial in range(max_trials_per_particle):
            candidate = np.array([
                random.uniform(0.0, Box_x),
                random.uniform(0.0, Box_y),
                random.uniform(0.0, Box_z)
            ], dtype=float)

            dc = candidate - center
            if np.dot(dc, dc) < sphere_r2:
                continue

            d2_surface = np.sum((surface_atoms - candidate) ** 2, axis=1)
            if np.any(d2_surface < min_dist2):
                continue

            solvent_atoms.append(candidate)
            placed = True
            break

        if (n + 1) % print_interval == 0 or (n + 1) == nsolvent:
            progress = (n + 1) / nsolvent * 100
            print(f"Progress: {n+1}/{nsolvent} ({progress:.2f}%)")

    solvent_atoms = np.array(solvent_atoms, dtype=float)
    new_atoms = np.vstack([atoms, solvent_atoms])
    new_atom_types = np.concatenate([atom_types, np.full(nsolvent, 4, dtype=int)])
    print("Finished adding solvent particles.")
    return new_atoms, new_atom_types

atoms, atom_types = Add_Solvent_Particles(
    atoms=atoms,
    atom_types=atom_types,
    nsolvent=Nsolvent,
    min_dist_surface=MinDist_Surface,
    sphere_radius=R_colloid
)

def Write_LAMMPS_Data(filename, atoms, atom_types):
    with open(filename, 'w') as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"\t{len(atoms)} atoms\n")
        f.write(f"\t{AtomTypes} atom types\n")
        f.write(f"\t0.0 {Box_x} xlo xhi\n")
        f.write(f"\t0.0 {Box_y} ylo yhi\n")
        f.write(f"\t0.0 {Box_z} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("\t1 0.15\n")
        f.write("\t2 0.15\n")
        f.write("\t3 2.42\n")
        f.write("\t4 1.00\n")
        f.write("\t5 1.00\n\n")
        f.write("Atoms\n\n")
        for i, atom in enumerate(atoms):
            f.write(f"\t{i+1}\t{atom_types[i]}\t{atom[0]:.6f}\t{atom[1]:.6f}\t{atom[2]:.6f}\n")
    print(f"LAMMPS data written to {filename}, generated {Number_Type2} type2 atoms ===> NBinomial")
    print(f"Generate {Nsolvent} solvent particles of type 4")
    print(f"Excluded inner sphere centered at box center with R = {R_colloid}")


Write_LAMMPS_Data(Output_LAMMPS_Data, atoms, atom_types)
