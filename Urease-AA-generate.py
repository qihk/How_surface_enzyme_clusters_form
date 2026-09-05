import numpy as np
from scipy.spatial import cKDTree

Urease_CG_File = "urease-cg.xyz"
File_Colloid_Data_List = ["colloid-structure.data", "colloid-structure-SiO2.data", "colloid-structure-SiO2-hollow.data"]
File_Colloid_Data = File_Colloid_Data_List[2]
output_file = "urease-colloid.lammpsdata"
Box_x = 30.0
Box_y = 30.0
Box_z = 30.0
Nurease = 1000

num_bonds = 0
AtomTypes = 8
BondTypes = 1

def read_urease_structure(filename):
    urease_cg_data = {
        "id": [],
        "type": [],
        "x": [],
        "y": [],
        "z": [],
        "q": []
    }

    with open(filename, 'r') as file:
        natoms = int(file.readline().strip())
        empty_line = file.readline()

        for atom_idx in range(natoms):
            atom_data = file.readline().split()

            urease_cg_data["id"].append(int(atom_data[0]))

            if atom_data[1] == "C":
                urease_cg_data["type"].append(4)
            elif atom_data[1] == "N":
                urease_cg_data["type"].append(5)
            elif atom_data[1] == "Ni":
                urease_cg_data["type"].append(6)
            elif atom_data[1] == "Si":
                urease_cg_data["type"].append(7)
            
            urease_cg_data["x"].append(float(atom_data[2]))
            urease_cg_data["y"].append(float(atom_data[3]))
            urease_cg_data["z"].append(float(atom_data[4]))
            urease_cg_data["q"].append(float(atom_data[5]))
    
    return urease_cg_data

def read_colloid_structure(filename):
    colloid_data = {
        "id": [],
        "type": [],
        "x": [],
        "y": [],
        "z": []
    }

    with open(filename, 'r') as file:
        natoms = int(file.readline().strip())
        empty_line = file.readline()

        for atom_idx in range(natoms):
            atom_data = file.readline().split()

            colloid_data["id"].append(int(atom_data[0]))
            colloid_data["type"].append(int(atom_data[1]))
            colloid_data["x"].append(float(atom_data[2]))
            colloid_data["y"].append(float(atom_data[3]))
            colloid_data["z"].append(float(atom_data[4]))
    
    return colloid_data

urease_cg_data = read_urease_structure(Urease_CG_File)
colloid_data = read_colloid_structure(File_Colloid_Data)

def random_rotation_matrix():
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1

    r = np.sqrt(1 - z**2)
    qw = np.cos(theta/2)
    qx = r * np.cos(phi) * np.sin(theta/2)
    qy = r * np.sin(phi) * np.sin(theta/2)
    qz = z * np.sin(theta/2)

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])
    return R

def generate(urease_cg_data, colloid_data, Nurease,
                Box_x, Box_y, Box_z,
                R=8.0, min_dist_atom=1.5, max_trials=10000):

    colloid_coords = np.vstack((colloid_data["x"], colloid_data["y"], colloid_data["z"])).T
    colloid_center = colloid_coords.mean(axis=0)
    box_center = np.array([Box_x/2, Box_y/2, Box_z/2])
    shift_vec = box_center - colloid_center
    colloid_coords += shift_vec

    total_atoms = {"id": [], "mol": [], "type": [], "x": [], "y": [], "z": [], "q": []}
    current_id = 0

    for idx in range(len(colloid_coords)):
        current_id += 1
        total_atoms["id"].append(current_id)
        total_atoms["mol"].append(1)
        total_atoms["type"].append(colloid_data["type"][idx])
        total_atoms["x"].append(colloid_coords[idx][0])
        total_atoms["y"].append(colloid_coords[idx][1])
        total_atoms["z"].append(colloid_coords[idx][2])
        total_atoms["q"].append(-0.01)

    urease_coords = np.vstack((urease_cg_data["x"], urease_cg_data["y"], urease_cg_data["z"])).T
    urease_types = urease_cg_data["type"]
    urease_charges = urease_cg_data["q"]
    urease_coords -= urease_coords.mean(axis=0)

    existing_coords = np.vstack((total_atoms["x"], total_atoms["y"], total_atoms["z"])).T
    tree = cKDTree(existing_coords)   # 初始 KDTree

    placed_count = 0
    for i in range(Nurease):
        success = False
        mol_id = i + 2
        for trial in range(max_trials):
            cx, cy, cz = np.random.uniform(0, Box_x), np.random.uniform(0, Box_y), np.random.uniform(0, Box_z)
            candidate_center = np.array([cx, cy, cz])

            if np.linalg.norm(candidate_center - box_center) < R:
                continue

            Rmat = random_rotation_matrix()
            rotated_coords = urease_coords @ Rmat.T + candidate_center

            dists, _ = tree.query(rotated_coords, k=1)
            if np.min(dists) < min_dist_atom:
                continue

            for j, coord in enumerate(rotated_coords):
                current_id += 1
                total_atoms["id"].append(current_id)
                total_atoms["mol"].append(mol_id)
                total_atoms["type"].append(urease_types[j])
                total_atoms["x"].append(coord[0])
                total_atoms["y"].append(coord[1])
                total_atoms["z"].append(coord[2])
                total_atoms["q"].append(urease_charges[j])

            existing_coords = np.vstack((existing_coords, rotated_coords))
            tree = cKDTree(existing_coords)
            placed_count += 1
            print(f"\rPlaced {placed_count}/{Nurease} ureases", end="")
            success = True
            break

        if not success:
            print(f"\nWarning: Urease {i+1} could not be placed after {max_trials} trials.")

    print("\nGeneration complete.")
    return total_atoms

total_atoms = generate(urease_cg_data, colloid_data, Nurease, Box_x, Box_y, Box_z)

def write_lammps_data(total_atoms, filename):
    with open(filename, 'w') as f:
        natoms = len(total_atoms["id"])

        f.write(f"LAMMPS Urease with Colloid data file\n\n")
        f.write(f"\t{natoms} atoms\n")
        f.write(f"\t{num_bonds} bonds\n\n")

        f.write(f"\t{AtomTypes} atom types\n")
        f.write(f"\t{BondTypes} bond types\n\n")

        f.write(f"\t0.0 {Box_x} xlo xhi\n")
        f.write(f"\t0.0 {Box_y} ylo yhi\n")
        f.write(f"\t0.0 {Box_z} zlo zhi\n\n")

        f.write("Masses\n\n")
        f.write("\t1 0.15\n")
        f.write("\t2 0.15\n")
        f.write("\t3 2.42\n")
        f.write("\t4 0.05\n")
        f.write("\t5 0.05\n")
        f.write("\t6 0.05\n")
        f.write("\t7 0.05\n")
        f.write("\t8 0.05\n\n")

        f.write("Atoms # full\n\n")
        for i in range(natoms):
            atom_id = total_atoms["id"][i]
            mol_id = total_atoms["mol"][i]
            atom_type = total_atoms["type"][i]
            q = total_atoms["q"][i]
            x = total_atoms["x"][i]
            y = total_atoms["y"][i]
            z = total_atoms["z"][i]

            f.write(f"{atom_id} {mol_id} {atom_type} {q:.3f} {x:.3f} {y:.3f} {z:.3f}\n")

    print(f"LAMMPS data written to {filename}")

write_lammps_data(total_atoms, output_file)