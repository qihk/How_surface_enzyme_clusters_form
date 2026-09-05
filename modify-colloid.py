import numpy as np
import random
import math

input_file = "colloid-structure-modify.data"
output_file = "colloid-structure-SiO2.data"

R = 3.0
surface_count = 2700
bump_height = 1.0 
bump_span_radius = 0.9
lambda_bump = 2
density_factor = 1.0

if bump_span_radius >= R:
    raise ValueError(f"{bump_span_radius} can not be larger than {R}")

target_angle_rad = np.arcsin(bump_span_radius / R)
target_angle_deg = np.degrees(target_angle_rad)

try:
    data = np.loadtxt(input_file)
    if data.shape[1] >= 5:
        atoms = data[:, 2:5]
    else:
        atoms = data[:, :3]
except Exception as e:
    print(f"error {e}")
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

initial_positions = surface_atoms.copy()
positions = surface_atoms.copy()
atom_types = np.ones(len(surface_atoms), dtype=int)
occupied = np.zeros(len(surface_atoms), dtype=bool)

def get_bump_geometry(R_base, h, theta_rad):
    z_edge = R_base * np.cos(theta_rad)
    x_edge = R_base * np.sin(theta_rad)
    H_top = R_base + h
    
    Rb = (x_edge**2 + (z_edge - H_top)**2) / (2 * (H_top - z_edge))
    offset_dist = H_top - Rb
    return Rb, offset_dist

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0: return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def fibonacci_sphere_cap(n_points, radius, cap_angle_rad):
    if n_points <= 0: return np.empty((0, 3))
    points = []
    cos_alpha = np.cos(cap_angle_rad)
    golden_ratio = (1 + 5**0.5) / 2
    for i in range(n_points):
        z = 1 - (1 - cos_alpha) * (i + 0.5) / n_points
        r_at_z = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        points.append([r_at_z * np.cos(theta) * radius, r_at_z * np.sin(theta) * radius, z * radius])
    return np.array(points)

def ray_sphere_intersection(origin, direction, sphere_center, sphere_radius):
    oc = origin - sphere_center
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - sphere_radius**2
    discriminant = b*b - 4*c
    if discriminant < 0: return None
    t = (-b + np.sqrt(discriminant)) / 2.0
    return direction * t

shuffled_indices = list(range(len(surface_atoms)))
random.shuffle(shuffled_indices)
target_bump_count = max(1, np.random.poisson(lambda_bump)) 
new_atoms_list = []

print(f"generate {target_bump_count}...")

count_success = 0
for _ in range(target_bump_count):
    for idx in shuffled_indices:
        if occupied[idx]: continue
        
        center_vec = initial_positions[idx]
        center_dir = center_vec / np.linalg.norm(center_vec)
        
        R_bump, offset_dist = get_bump_geometry(R, bump_height, target_angle_rad)
        bump_center_pos = center_dir * offset_dist
        
        dists_rad = np.array([np.arccos(np.clip(np.dot(center_dir, v/np.linalg.norm(v)), -1, 1)) for v in initial_positions])
        local_mask = (dists_rad < target_angle_rad) & (~occupied)
        local_indices = np.where(local_mask)[0]
        
        if len(local_indices) == 0: continue
        
        for idx0 in local_indices:
            orig_dir = initial_positions[idx0] / np.linalg.norm(initial_positions[idx0])
            new_pos = ray_sphere_intersection(np.array([0,0,0]), orig_dir, bump_center_pos, R_bump)
            if new_pos is not None:
                positions[idx0] = new_pos
                atom_types[idx0] = 2
                occupied[idx0] = True
        
        area_orig = 2 * np.pi * R**2 * (1 - np.cos(target_angle_rad))
        
        L_chord_sq = (R * np.sin(target_angle_rad))**2 + (R * (1 - np.cos(target_angle_rad)) + bump_height)**2
        area_bump = np.pi * L_chord_sq
        
        n_expected = int(len(local_indices) * (area_bump / area_orig) * density_factor)
        n_to_add = max(0, n_expected - len(local_indices))
        
        if n_to_add > 0:
            val = np.clip(1 - area_bump / (2 * np.pi * R_bump**2), -1.0, 1.0)
            alpha_bump_rad = np.arccos(val)
            
            cap_points = fibonacci_sphere_cap(n_to_add, R_bump, alpha_bump_rad)
            rot_mat = rotation_matrix_from_vectors(np.array([0, 0, 1]), center_dir)
            final_points = (cap_points @ rot_mat.T) + bump_center_pos
            new_atoms_list.append(final_points)
            
        count_success += 1
        break

# ========= 合并数据 ==========
if len(new_atoms_list) > 0:
    inserted_points = np.vstack(new_atoms_list)
else:
    inserted_points = np.empty((0, 3))

all_positions = np.vstack((positions, inserted_points, interior_atoms))
all_types = np.concatenate((
    atom_types,
    np.full(len(inserted_points), 2),
    np.full(len(interior_atoms), 3)
))

with open(output_file, 'w') as f:
    f.write(f"{len(all_positions)}\n\n")
    for i, (atype, pos) in enumerate(zip(all_types, all_positions), start=1):
        f.write(f"{i} {atype} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")