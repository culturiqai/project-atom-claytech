import os
# FORCE APPLE METAL OPTIMIZATION
# This ensures JAX does not try to split the grid across non-existent devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
import jax.numpy as jnp
import xlb
import trimesh
import numpy as np
import time
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC, FullwayBounceBackBC, RegularizedBC, ExtrapolationOutflowBC
)
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk

# --- 1. STUDIO-GRADE CONFIGURATION ---
# We upscale by 1.5x (from 256 to 384). 
# Total Voxels: ~14.1 Million. 
NX, NY, NZ = 384, 192, 192 
grid_shape = (NX, NY, NZ)

# PHYSICS: THE "GOLDILOCKS" ZONE
# Re=15,000 is the sweet spot for KBC on this grid size.
# High enough to shed vortices, low enough to be stable without LES smearing.
Re = 1000.0        
wind_speed = 0.025   
num_steps = 4000     # Run long enough to develop a full wake
start_avg_step = 1000 # Start averaging after wake develops
print_interval = 50

# --- 2. BACKEND SETUP ---
compute_backend = ComputeBackend.JAX
precision_policy = PrecisionPolicy.FP32FP32 # Apple Metal optimized
velocity_set = xlb.velocity_set.D3Q27(precision_policy, compute_backend)

xlb.init(velocity_set=velocity_set, default_backend=compute_backend, default_precision_policy=precision_policy)
grid = grid_factory(grid_shape, compute_backend=compute_backend)

# --- 3. GEOMETRY ENGINEERING ---
print(">>> Loading & Prepping Geometry (Studio Mode)...")
mesh = trimesh.load('lambo.stl')

# SCALING: 22% Blockage
# We maintain the aerodynamic proportion relative to the new, larger tunnel.
target_length = NX * 0.22
scale_factor = target_length / mesh.extents[0]
mesh.apply_scale(scale_factor)
print(f" -> Car Length: {target_length:.1f} voxels")

# POSITIONING
min_vals, max_vals = mesh.bounds
mesh.apply_translation([-(min_vals[0] + max_vals[0])/2, -(min_vals[1] + max_vals[1])/2, -(min_vals[2] + max_vals[2])/2])
# Place on floor with tire clearance
z_offset = (mesh.extents[2] / 2) + 1.5 
mesh.apply_translation([NX/4.0, NY/2.0, z_offset])

# VOXELIZATION
print(" -> Voxelizing (High Res)...")
voxel_grid = mesh.voxelized(pitch=1.0)
car_points = voxel_grid.points 
global_indices = np.round(car_points).astype(int)

# SAFETY CLIPPING
valid_mask = (
    (global_indices[:,0] >= 1) & (global_indices[:,0] < NX-1) &
    (global_indices[:,1] >= 1) & (global_indices[:,1] < NY-1) &
    (global_indices[:,2] >= 1) & (global_indices[:,2] < NZ-1)
)
global_indices = global_indices[valid_mask]
car_indices_list = [global_indices[:,0].tolist(), global_indices[:,1].tolist(), global_indices[:,2].tolist()]

# --- 4. BOUNDARY CONDITIONS ---
box_no_edge = grid.bounding_box_indices(remove_edges=True)
box = grid.bounding_box_indices()

# INLET
bc_inlet = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=box_no_edge["left"])
# OUTLET
bc_outlet = ExtrapolationOutflowBC(indices=box_no_edge["right"])
# FLOOR (No Slip)
bc_floor = FullwayBounceBackBC(indices=box_no_edge["bottom"])
# CAR (No Slip)
bc_car = HalfwayBounceBackBC(indices=car_indices_list)

# SKY & WALLS (Open Air - Slip)
# Explicit concatenation fix applied
faces_to_slip = ["top", "front", "back"]
all_x = [box_no_edge[face][0] for face in faces_to_slip]
all_y = [box_no_edge[face][1] for face in faces_to_slip]
all_z = [box_no_edge[face][2] for face in faces_to_slip]
slip_indices = [np.concatenate(all_x).tolist(), np.concatenate(all_y).tolist(), np.concatenate(all_z).tolist()]
bc_slip = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=slip_indices)

# --- 5. COMPILATION ---
print(">>> Compiling KBC Solver (Entropic Stabilized)...")
# We return to "KBC" (Kinetic Boundary Condition).
# This is the "Pure Physics" mode. It is self-stabilizing via entropy.
stepper = IncompressibleNavierStokesStepper(
    grid=grid, 
    boundary_conditions=[bc_floor, bc_slip, bc_inlet, bc_outlet, bc_car], 
    collision_type="KBC" 
)

f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
macro = Macroscopic(compute_backend=compute_backend, precision_policy=precision_policy, velocity_set=velocity_set)

# VISCOSITY SETUP
nu = wind_speed * (mesh.extents[0]) / Re
omega = 1.0 / (3.0 * nu + 0.5)
print(f" -> Resolution: {NX}x{NY}x{NZ}")
print(f" -> Re: {Re}")
print(f" -> Viscosity (nu): {nu:.6f}")
print(f" -> Relaxation (omega): {omega:.4f}")

# --- 6. INSTRUMENTATION ---
@jax.jit
def compute_kinetic_energy(f):
    rho, u = macro(f)
    # Sum of v^2 * rho (approx)
    return jnp.sum(rho * (u[0]**2 + u[1]**2 + u[2]**2))

# --- 7. SIMULATION LOOP ---
print(">>> STARTING STUDIO RENDER SIMULATION...")
print(">>> (This will take time. Go grab a coffee.)")
start_time = time.time()

# Averaging Accumulators
u_sum = None
rho_sum = None
avg_count = 0

for step in range(num_steps):
    # Standard KBC Step (No manual LES injection needed)
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
    f_0, f_1 = f_1, f_0
    
    # Instrumentation & Averaging
    if step >= start_avg_step:
        rho, u = macro(f_0)
        if u_sum is None:
            u_sum, rho_sum = u, rho
        else:
            u_sum, rho_sum = u_sum + u, rho_sum + rho
        avg_count += 1

    if step % print_interval == 0:
        ke_total = compute_kinetic_energy(f_0)
        elapsed = time.time() - start_time
        
        # Stability Check
        if jnp.isnan(ke_total):
            print(f"!!! CRASH DETECTED AT STEP {step:04d} !!!")
            print(">>> Simulation Instability (NaN). Stopping early.")
            break

        # We track Global Kinetic Energy. 
        # If this plateaus, we have reached steady-state turbulence.
        # If it spikes to 1e10, we crashed.
        print(f"Step {step:04d} | Global KE: {ke_total:.4f} | Avg Samples: {avg_count} | {elapsed:.2f}s")
        start_time = time.time()

# --- 8. CLEAN OUTPUT ---
if avg_count > 0:
    print(f">>> Simulation Complete. Saving Averaged High-Fidelity Data ({avg_count} samples)...")
    u_avg = u_sum / avg_count
    rho_avg = rho_sum / avg_count
    
    rho_final = jnp.squeeze(rho_avg)
    u_final = u_avg
    filename_base = "lambo_studio_kbc_averaged"
else:
    print(">>> Warning: No averaging samples collected (crash or short run). Saving instantaneous snapshot.")
    rho_final = jnp.squeeze(macro(f_0)[0])
    u_final = macro(f_0)[1]
    filename_base = "lambo_studio_kbc_snapshot"

u_mag = jnp.sqrt(u_final[0]**2 + u_final[1]**2 + u_final[2]**2)

# Save VTI
# Save VTI
fields = {
    # "velocity": u_final, # Removed to avoid dimension mismatch with scalars
    "velocity_magnitude": u_mag,
    "velocity_x": u_final[0],
    "velocity_y": u_final[1],
    "velocity_z": u_final[2],
    "density": rho_final,
    "pressure_approx": (rho_final - 1.0) / 3.0
}
# Output filename is crucial for the Viz script
save_fields_vtk(fields, timestep=num_steps)
# Rename the generic output to our specific name
try:
    # XLB saves as .vtk by default even if we want .vti structure
    os.rename(f"fields_{num_steps:07d}.vtk", f"{filename_base}_{num_steps:07d}.vtk")
    print(f">>> SAVED: {filename_base}_{num_steps:07d}.vtk")
except OSError as e:
    print(f">>> SAVED: fields_{num_steps:07d}.vtk (Rename failed: {e})")
print(">>> READY FOR CINEMATIC VIZ.")