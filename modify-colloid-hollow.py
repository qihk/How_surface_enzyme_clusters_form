import numpy as np
import random
from scipy.spatial.distance import pdist, squareform

input_file = "colloid-structure-modify.data"
output_file = "colloid-structure-SiO2-hollow.data"

R = 3.0
surface_count = 2700
dent_depth = 0.5
dent_span_radius = 1.0

lambda_dent = 25
min_dist_threshold = 0.15

if dent_span_radius >= R:
    raise ValueError("dent_span_radius can not larger than R")

target_angle_rad = np.arcsin(dent_span_radius / R)
target_angle_deg = np.degrees(target_angle_rad)

try:
    data = np.loadtxt(input_file)
    if data.shape[1] >= 5:
        atoms = data[:, 2:5]
    else:
        atoms = data[:, :3]
except Exception as e:
    print(f"error file: {e}")
    indices = np.arange(0, surface_count, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/surface_count)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)
    atoms = np.column_stack((x, y, z))
    interior_atoms = np.empty((0, 3))

if len(atoms) > surface_count:
    surface_atoms = atoms[:surface_count]
    interior_atoms = atoms[surface_count:]
else:
    surface_atoms = atoms
    interior_atoms = np.empty((0, 3))

positions = surface_atoms.copy()
initial_positions = surface_atoms.copy()
atom_types = np.ones(len(surface_atoms), dtype=int)
active_mask = np.ones(len(surface_atoms), dtype=bool)
occupied = np.zeros(len(surface_atoms), dtype=bool)

def apply_smooth_deformation(pos, center_dir, theta, theta_max, depth, R_base):
    ratio = theta / theta_max
    factor = 0.5 * (1 + np.cos(np.pi * ratio))
    current_R = np.linalg.norm(pos)
    new_R = R_base - depth * factor
    new_pos = (pos / current_R) * new_R
    return new_pos

def remove_overlaps(local_indices, positions, threshold):
    local_pos = positions[local_indices]
    n = len(local_pos)
    if n < 2: return []
    
    dists = squareform(pdist(local_pos))
    to_remove = set()
    
    for i in range(n):
        if i in to_remove: continue
        for j in range(i + 1, n):
            if j in to_remove: continue
            if dists[i, j] < threshold:
                to_remove.add(j)
                
    return [local_indices[k] for k in to_remove]

shuffled_indices = list(range(len(surface_atoms)))
random.shuffle(shuffled_indices)
target_dent_count = max(1, np.random.poisson(lambda_dent))

print(f"generate {target_dent_count}...")

count_success = 0

for _ in range(target_dent_count):
    for idx in shuffled_indices:
        if occupied[idx] or not active_mask[idx]: continue
        
        center_vec = initial_positions[idx]
        center_dir = center_vec / np.linalg.norm(center_vec)
        norm_pos = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        dots = np.dot(norm_pos, center_dir)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        local_mask = (angles < target_angle_rad) & active_mask & (~occupied)
        local_indices = np.where(local_mask)[0]
        
        if len(local_indices) == 0: continue

        for idx0 in local_indices:
            theta = angles[idx0]
            old_pos = positions[idx0]
            
            new_pos = apply_smooth_deformation(old_pos, center_dir, theta, target_angle_rad, dent_depth, R)
            
            positions[idx0] = new_pos
            atom_types[idx0] = 2
            occupied[idx0] = True
            
        remove_list = remove_overlaps(local_indices, positions, min_dist_threshold)
        
        if len(remove_list) > 0:
            active_mask[remove_list] = False

        count_success += 1
        break

final_positions = positions[active_mask]
final_types = atom_types[active_mask]

if len(interior_atoms) > 0:
    out_pos = np.vstack((final_positions, interior_atoms))
    out_types = np.concatenate((final_types, np.full(len(interior_atoms), 3)))
else:
    out_pos = final_positions
    out_types = final_types

with open(output_file, 'w') as f:
    f.write(f"{len(out_pos)}\n\n")
    for i, (atype, pos) in enumerate(zip(out_types, out_pos), start=1):
        f.write(f"{i} {atype} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")