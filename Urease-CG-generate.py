import numpy as np
from Bio.PDB import PDBParser
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from scipy.spatial import ConvexHull
import random
import os

pdb_file = "6ZJA.pdb"
n_total = 70
lj_diameter = 0.075
n_type2 = 10
Ni_resnames = ["NI", "Ni"]
lys_arg_resnames = ["LYS", "ARG"]

parser = PDBParser(QUIET=True)
structure = parser.get_structure("urease", pdb_file)

coords_all = []
coords_lys_arg = []
coords_Ni = []

for atom in structure.get_atoms():
    resname = atom.get_parent().get_resname()
    pos = atom.get_coord()
    coords_all.append(pos)

    if resname in lys_arg_resnames:
        coords_lys_arg.append(pos)
    if resname in Ni_resnames:
        coords_Ni.append(pos)

coords_all = np.array(coords_all)
coords_lys_arg = np.array(coords_lys_arg)
coords_Ni = np.array(coords_Ni)

print("read total atoms: ", len(coords_all))
print("read -NH atoms: ", len(coords_lys_arg))
print("read Ni atoms: ", len(coords_Ni))

kmeans = KMeans(n_clusters=n_total, random_state=42).fit(coords_all)
cg_coords = kmeans.cluster_centers_

hull = ConvexHull(cg_coords)
surface_indices = list(set(hull.vertices))

if len(coords_lys_arg) > 0:
    dist_mat = distance_matrix(coords_lys_arg, cg_coords)
    lys_arg_nearest = np.argmin(dist_mat, axis=1)

    lys_arg_surface = [idx for idx in lys_arg_nearest if idx in surface_indices]
    lys_arg_unique = list(set(lys_arg_surface))
else:
    lys_arg_unique = []

if len(lys_arg_unique) >= n_type2:
    type2_indices = random.sample(lys_arg_unique, n_type2)
else:
    type2_indices = lys_arg_unique

cg_coords = np.vstack([cg_coords, coords_Ni])
types = np.ones(len(cg_coords), dtype=int)
charges = np.zeros(len(cg_coords))

for idx in type2_indices:
    types[idx] = 2

for i in range(len(coords_Ni)):
    types[-(i+1)] = 3

center = np.mean(cg_coords, axis=0)
mean_radius_before = np.mean(np.linalg.norm(cg_coords - center, axis=1))
scale = lj_diameter / mean_radius_before
cg_coords = (cg_coords - center) * scale

with open("urease-cg.xyz", "w") as f:
    f.write(f"{len(cg_coords)}\n")
    f.write(f"\n")
    atom_idx = 1
    type_map = {1:"C", 2:"N", 3:"Ni"}
    for (x, y, z), t, q in zip(cg_coords, types, charges):
        atom_name = type_map[t]
        f.write(f"{atom_idx}\t{atom_name}\t{x:.8f}\t{y:.8f}\t{z:.8f}\t{q:.2f}\n")
        atom_idx += 1

print("CG-structure outputs to file urease-cg.xyz...")

center = np.mean(cg_coords, axis=0)
radii = np.linalg.norm(cg_coords - center, axis=1)
mean_radius = np.mean(radii)
cg_coords = np.vstack([cg_coords, center])
types = np.append(types, 4)
charges = np.append(charges, 0.01)

with open("urease-cg.xyz", "w") as f:
    f.write(f"{len(cg_coords)}\n")
    f.write(f"\n")
    atom_idx = 1
    type_map = {1:"C", 2:"N", 3:"Ni", 4:"Si"}
    for (x, y, z), t, q in zip(cg_coords, types, charges):
        atom_name = type_map[t]
        f.write(f"{atom_idx}\t{atom_name}\t{x:.8f}\t{y:.8f}\t{z:.8f}\t{q:.2f}\n")
        atom_idx += 1