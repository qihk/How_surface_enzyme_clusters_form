import numpy as np
import random

Box_x = 30.0
Box_y = 30.0
Box_z = 30.0

AtomTypes = 5
Nsurface = 2700
Number_Type2 = int(Nsurface * 0.50)

Nsolvent = int(Box_x * Box_y * Box_z * 10)
MinDist_Surface = 0.8
R_colloid = 3.0

File_Colloid_Data = "colloid-structure.data"
Output_LAMMPS_Data = "CP-lammps.data"

def Read_Colloid_Data(filename1):
    atoms = []
    center_x = Box_x / 2
    center_y = Box_y / 2
    center_z = Box_z / 2

    with open(filename1, 'r') as file:
        for line in file:
            if line.strip() == '':
                continue

            x, y, z, *_ = map(float, line.strip().split())
            atoms.append([x + center_x, y + center_y, z + center_z])

    return np.array(atoms, dtype=float)

atoms = Read_Colloid_Data(File_Colloid_Data)

def Choose_Type2(atoms):
    center_x = Box_x / 2

    atoms_subset = atoms[:Nsurface]
    right_half_indices = [i for i, coords in enumerate(atoms_subset) if coords[0] > center_x]

    if len(right_half_indices) < Number_Type2:
        print("len(right_half_indices) = ", len(right_half_indices), ", Number_Type2 = ", Number_Type2)
        raise ValueError("no enough atoms")

    selected_indices = random.sample(right_half_indices, Number_Type2)
    return selected_indices

selected_indices = Choose_Type2(atoms)

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
        f.write(f"\t{AtomTypes} atom types\n\n")

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
            atom_id = i + 1
            atom_type = atom_types[i]
            f.write(f"\t{atom_id}\t{atom_type}\t{atom[0]:.6f}\t{atom[1]:.6f}\t{atom[2]:.6f}\n")
        f.write("\n")

    print(f"LAMMPS data written to {filename}")
    print(f"Generate {Number_Type2} type2 atoms on surface ===> Janus")
    print(f"Generate {Nsolvent} solvent particles of type 4")
    print(f"Excluded inner sphere centered at box center with R = {R_colloid}")


Write_LAMMPS_Data(Output_LAMMPS_Data, atoms, atom_types)