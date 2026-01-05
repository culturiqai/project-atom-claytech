import pyvista as pv
import numpy as np
import os

print(">>> INITIALIZING LAMBO-STANDARD DIAGNOSTICS...")

# --- 1. UNIVERSAL LOADER (The Bridge) ---
# Looks for data in current dir OR High-Fieldity-Dev
search_paths = [".", "High-Fieldity-Dev"]
target_file = None

for path in search_paths:
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if (f.endswith(".vtk") or f.endswith(".vti")) and ("fields" in f or "lambo" in f)]
        if files:
            target_file = os.path.join(path, sorted(files)[-1])
            print(f" -> Found Data: {target_file}")
            break

if not target_file:
    raise FileNotFoundError("No physics data found! Run a simulation first.")

grid_vol = pv.read(target_file)

# --- DEBUG & PRE-PROCESSING ---
print(f" -> Available Arrays: {grid_vol.array_names}")

# Convert Cell Data to Point Data if needed
# (XLB simulation often saves as cell-centered data)
if grid_vol.n_points > 0 and len(grid_vol.point_data) == 0 and len(grid_vol.cell_data) > 0:
    print(" -> Detecting Cell Data... Converting to Point Data for Visualization.")
    grid_vol = grid_vol.cell_data_to_point_data()

# Ensure 'velocity' vector exists (reconstruct from scalars if needed)
if "velocity" not in grid_vol.point_data:
    if "velocity_x" in grid_vol.point_data:
        print(" -> Reconstructing Velocity Vector from Scalars...")
        vx = grid_vol.point_data["velocity_x"]
        vy = grid_vol.point_data["velocity_y"]
        vz = grid_vol.point_data["velocity_z"]
        # Use numpy column_stack to create (N, 3) array
        grid_vol.point_data["velocity"] = np.column_stack((vx, vy, vz))

# Determine Mode based on resolution or bounds
if hasattr(grid_vol, "dimensions"):
    RES_X = grid_vol.dimensions[0]
else:
    # Fallback for PolyData: Use X-bounds length
    RES_X = int(grid_vol.bounds[1] - grid_vol.bounds[0])
    
if RES_X >= 380: # Allow some tolerance
    print(" -> MODE DETECTED: PRO (High-Fidelity)")
    NX, NY, NZ = 384, 192, 192
    SCALE = 0.22
    WIND_SPEED = 0.025
    Z_OFFSET_FIX = 1.5
else:
    print(f" -> MODE DETECTED: TOY (Legacy/Fallback) [Res: {RES_X}]")
    NX, NY, NZ = 256, 128, 128
    SCALE = 0.45
    WIND_SPEED = 0.04
    Z_OFFSET_FIX = 2.0

# --- 2. GEOMETRY ALIGNMENT ---
print(" -> Loading Car Geometry...")
# Try loading from root or subfolder
try:
    mesh = pv.read("lambo.stl")
except:
    mesh = pv.read("High-Fieldity-Dev/lambo.stl")

# Scale
mesh.scale([SCALE * NX / mesh.length, SCALE * NX / mesh.length, SCALE * NX / mesh.length], inplace=True)
# Center
center = np.array(mesh.center)
mesh.translate(-center, inplace=True)
# Move
mesh.translate([NX/3.5 if RES_X==256 else NX/4.0, NY/2.0, (mesh.bounds[5]-mesh.bounds[4])/2 + Z_OFFSET_FIX], inplace=True)


# --- 3. PHYSICS EXTRACTION ---
print(" -> Mapping Physics to Surface...")
# Compute normals FIRST so we can use them for offset
mesh.compute_normals(inplace=True, flip_normals=True)
normals = mesh.point_data["Normals"]

# Sample physics slightly *off* the surface to avoid No-Slip (Zero Velocity) issues
# Sample physics slightly *off* the surface to avoid No-Slip (Zero Velocity) issues
# Create a probe shell 1.5 voxels out
shell = mesh.copy()
shell.points += mesh.point_data['Normals'] * 1.5
shell = shell.sample(grid_vol)

# Transfer velocity from shell back to mesh for visualization
mesh.point_data["velocity"] = shell.point_data["velocity"]

# Sample Scalar Data (Density) directly on surface (approx is fine)
mesh_direct = mesh.sample(grid_vol)
mesh.point_data["density"] = mesh_direct.point_data["density"]

# A. Pressure Coefficient (Cp)
rho = mesh.point_data["density"]
pressure = (rho - 1.0) / 3.0
Cp = pressure / (0.5 * 1.0 * WIND_SPEED**2)
mesh.point_data["Cp"] = Cp

# B. Forces & Efficiency
# Normals already computed above
# Drag (Fx) and Downforce (-Fz)
# Note: Normals point OUT. Pressure pushes IN. 
# Note: Normals point OUT. Pressure pushes IN. 
# Force Vector = -Cp * Normal
Fx = -Cp * normals[:, 0]
Fz = -Cp * normals[:, 2]

# "Downforce" is negative Fz (force pointing down)
Downforce = -Fz 

# Efficiency Ratio (L/D)
# We add epsilon to Drag to avoid division by zero
epsilon = 1e-6
Efficiency = Downforce / (np.abs(Fx) + epsilon)
# Cap the efficiency for visualization (tighten range to see details)
# Range: -1 (Bad Lift) to +2 (Good Downforce)
Efficiency = np.clip(Efficiency, -1, 2)

mesh.point_data["Efficiency"] = Efficiency
mesh.point_data["Downforce"] = Downforce

# C. Separation Risk (Oil Flow & Gradient)
# Calculate gradient of Cp
mesh_grad = mesh.compute_derivative(scalars="Cp")
grad_mag = np.linalg.norm(mesh_grad.point_data["gradient"], axis=1)
mesh.point_data["Separation_Risk"] = grad_mag

# Generate Surface Streamlines (Oil Flow)
print(" -> Generating Oil Flow Lines (Volume Integration)...")
try:
    # 1. Use the 'shell' (offset surface) as seeds
    # 2. Subsample seeds to avoid clutter (Reduced to 500 for clarity)
    seed_poly = shell.extract_points(np.random.randint(0, shell.n_points, 500))
    
    # 3. Integrate through the VOLUME (grid_vol), not the surface
    streams = grid_vol.streamlines_from_source(
        seed_poly,
        vectors="velocity",
        integration_direction="both",
        max_time=100.0, # Let them flow long
        surface_streamlines=False 
    )
except Exception as e:
    print(f"Warning: Oil Flow generation failed ({e})")
    streams = None

# D. Aero Moment (Balance)
# Moment about Y-axis (Pitch) = Fz * (x - Cog_x)
# Assessing geometric center as rough CoG
cog_x = (mesh.bounds[0] + mesh.bounds[1]) / 2
moment_arm = mesh.points[:, 0] - cog_x
Pitch_Moment = Downforce * moment_arm
mesh.point_data["Pitch_Moment"] = Pitch_Moment

# --- 4. THE LAMBO DASHBOARD ---
pl = pv.Plotter(shape=(2, 2))
pl.set_background("white")

# TOP LEFT: THE GRIP MAP (Cp)
pl.subplot(0, 0)
pl.add_text("1. GRIP MAP (Cp)\nBlue = Suction (Grip)\nRed = Pressure (Drag)", color="black", font_size=10)
pl.add_mesh(mesh, scalars="Cp", cmap="coolwarm", flip_scalars=True, show_edges=False)

# TOP RIGHT: THE EFFICIENCY MAP (L/D)
pl.subplot(0, 1)
pl.add_text("2. EFFICIENCY (L/D Ratio)\nGreen = Good (Efficient Grip)\nRed = Bad (Parasitic Drag)", color="black", font_size=10)
# We mask out low-force areas to avoid visual noise
pl.add_mesh(mesh, scalars="Efficiency", cmap="RdYlGn", clim=[-1, 2], show_edges=False)

# BOTTOM LEFT: SEPARATION RISK (Oil Flow)
pl.subplot(1, 0)
pl.add_text("3. OIL FLOW (Surface Streamlines)\nLines show air path.\nConvergence = Separation.", color="black", font_size=10)
# Background: Gradient Map
limit_grad = np.percentile(grad_mag, 98) 
pl.add_mesh(mesh, scalars="Separation_Risk", cmap="gray_r", clim=[0, limit_grad], show_edges=False, opacity=0.5)
# Overlay: Oil Flow Lines
if streams and streams.n_points > 0:
    pl.add_mesh(streams, color="orange", line_width=1.5, opacity=0.8)
else:
    pl.add_text("\n(No attached flow lines found)", font_size=8, color="orange", position="lower_left")

# BOTTOM RIGHT: BALANCE (Pitch Moment)
pl.subplot(1, 1)
pl.add_text("4. BALANCE (Pitch Moment)\nBlue = Front Grip (Oversteer)\nRed = Rear Grip (Understeer)", color="black", font_size=10)
# Center the map at 0
limit_moment = np.max(np.abs(Pitch_Moment))
pl.add_mesh(mesh, scalars="Pitch_Moment", cmap="bwr", clim=[-limit_moment, limit_moment], show_edges=False)

print(">>> DASHBOARD READY.")
print(">>> Review the 4 Panes to certify the design.")
pl.link_views()
pl.show()