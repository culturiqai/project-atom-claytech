"""
JAX-LBM Final Design Studio
Author: Ex-NVIDIA Director of Physical AI
Date: Dec 8, 2025

Features:
1. AI Inverse Design (Gradient Descent)
2. 'Clean Air' Physics (No background drag)
3. Scientific Visualization (Vorticity + Pressure)
4. Animation Export (Evolution of the Shape)
"""

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from lbm_oracle import LBMSolverJAX

# --- CONFIG ---
NX, NY = 300, 100
LATENT_NX, LATENT_NY = 40, 20
STEPS = 1000  # Longer run for stable physics
BATCH_SIZE = 1
TAU = 0.53    # Stable fluid settings
INLET_VEL = 0.08

def get_initial_state(solver):
    rho = jnp.ones((BATCH_SIZE, NX, NY))
    u = jnp.full((BATCH_SIZE, NX, NY), INLET_VEL)
    v = jnp.zeros((BATCH_SIZE, NX, NY))
    return solver.equilibrium(rho, u, v)

# --- 1. GEOMETRY ENGINE (With Clean Air Switch) ---
def generate_mask(latent_logits, hard_mode=False):
    """
    Generates the shape. 
    hard_mode=True -> Returns binary 0.0/1.0 mask (For Visualization/Validation)
    hard_mode=False -> Returns soft gradients (For AI Training)
    """
    # Upsample Latents
    latent_grid = jax.nn.sigmoid(latent_logits)
    img = latent_grid.reshape(1, 1, LATENT_NX, LATENT_NY)
    img_high_res = jax.image.resize(img, shape=(1, 1, NX, NY), method='bilinear')
    
    # SDF-like sharpening
    raw_mask = 12.0 * (img_high_res.squeeze() - 0.4)
    mask = jax.nn.sigmoid(raw_mask)
    
    if hard_mode:
        # THE FIX: Clip background noise to absolute zero
        mask = jnp.where(mask < 0.05, 0.0, 1.0)
    else:
        # Training Mode: Keep it slightly soft but suppress background
        mask = jnp.where(mask < 0.01, 0.0, mask)
        
    return mask

# --- 2. PHYSICS LOSS ---
def simulation_loss(latent_logits, solver, init_state):
    # Use Soft Mask for Differentiability
    mask = generate_mask(latent_logits, hard_mode=False)
    mask_batched = jnp.repeat(mask[None, ...], BATCH_SIZE, axis=0)

    # Run Physics
    params = (mask_batched, INLET_VEL, 1.0, TAU)
    final_state, (ux, uy, rho) = solver.run_simulation(init_state, STEPS, params)

    # Objectives
    # A. Minimize Pressure Drop (Efficiency)
    p_in = jnp.mean(rho[:, 10:50, :])
    p_out = jnp.mean(rho[:, -50:, :])
    pressure_drag = (p_in - p_out) * 1000.0
    
    # B. Volume Constraint
    current_vol = jnp.sum(mask)
    target_vol = 0.06 * NX * NY # Target ~6% volume
    vol_loss = (current_vol - target_vol)**2 * 0.05
    
    # C. Centering (Keep it in the middle)
    y_indices, x_indices = jnp.arange(NY), jnp.arange(NX)
    Y, X = jnp.meshgrid(y_indices, x_indices)
    center_x = jnp.sum(mask * X) / (current_vol + 1e-6)
    center_y = jnp.sum(mask * Y) / (current_vol + 1e-6)
    center_loss = ((center_x - NX//3.5)**2 + (center_y - NY//2)**2) * 0.01

    total_loss = pressure_drag + vol_loss + center_loss
    return total_loss, (mask, ux, uy, rho)

# --- 3. MAIN STUDIO ---
def main():
    print(f"🏭 Initializing AI Design Studio ({NX}x{NY})...")
    solver = LBMSolverJAX(NX, NY)
    init_state = get_initial_state(solver)

    # Initialize with Gaussian Seed (The "Egg" DNA)
    key = jax.random.PRNGKey(42)
    y_lat, x_lat = jnp.meshgrid(jnp.arange(LATENT_NY), jnp.arange(LATENT_NX))
    dist_sq = (x_lat - LATENT_NX//2.5)**2 + (y_lat - LATENT_NY//2)**2
    latent_logits = 5.0 * jnp.exp(-dist_sq / 10.0) - 2.0 # Soft seed

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(latent_logits)

    # Compile Training Step
    @jax.jit
    def train_step(logits, opt_state):
        (loss, (mask, ux, uy, rho)), grads = jax.value_and_grad(simulation_loss, has_aux=True)(
            logits, solver, init_state
        )
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_logits = optax.apply_updates(logits, updates)
        return new_logits, new_opt_state, loss, mask

    print("🧠 Optimizing Geometry...")
    t0 = time.time()
    
    history_loss = []
    snapshots = [] # Store masks for animation

    for i in range(121):
        latent_logits, opt_state, loss, mask = train_step(latent_logits, opt_state)
        history_loss.append(loss)
        
        if i % 10 == 0:
            print(f"   Step {i:03d} | Loss: {loss:.4f}")
            snapshots.append(np.array(mask)) # Save CPU copy

    print(f"✅ Optimization Complete ({time.time()-t0:.1f}s)")

    # --- 4. FINAL SCIENTIFIC RENDER ---
    print("🎨 Generating High-Fidelity Physics Render...")
    
    # Generate the FINAL HARD MASK (Correct Visuals)
    final_mask = generate_mask(latent_logits, hard_mode=True)
    mask_batched = jnp.repeat(final_mask[None, ...], BATCH_SIZE, axis=0)
    
    # Run one last high-res simulation
    params = (mask_batched, INLET_VEL, 1.0, TAU)
    final_state, (ux, uy, rho) = solver.run_simulation(init_state, 1500, params)
    
    # Compute Vorticity (Curl)
    du_dy = jnp.gradient(ux[0], axis=1)
    dv_dx = jnp.gradient(uy[0], axis=0)
    vorticity = dv_dx - du_dy
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2)
    
    # A. Evolution Animation (Top Left)
    ax_evo = fig.add_subplot(gs[0, 0])
    ax_evo.set_title("AI Optimization History (Shape Evolution)")
    im_evo = ax_evo.imshow(snapshots[0].T, cmap='gray', origin='lower')
    ax_evo.axis('off')

    def update_evo(frame):
        im_evo.set_data(snapshots[frame].T)
        return [im_evo]
    
    # B. Vorticity (Top Right)
    ax_vort = fig.add_subplot(gs[0, 1])
    ax_vort.set_title("Vorticity (Turbulence Check)")
    ax_vort.imshow(vorticity.T, cmap='seismic', vmin=-0.03, vmax=0.03, origin='lower')
    ax_vort.contour(final_mask.T, colors='black')
    ax_vort.axis('off')

    # C. Pressure Field (Bottom Span)
    ax_press = fig.add_subplot(gs[1, :])
    # Pressure = Density - 1.0
    pressure_field = rho[0] - 1.0
    ax_press.set_title("Pressure Field (Blue = Suction, Red = Drag)")
    im_p = ax_press.imshow(pressure_field.T, cmap='coolwarm', vmin=-0.015, vmax=0.015, origin='lower')
    ax_press.contour(final_mask.T, colors='black')
    ax_press.axis('off')
    plt.colorbar(im_p, ax=ax_press, orientation='vertical', fraction=0.02)

    plt.tight_layout()
    
    # Save Animation
    print("🎥 Saving Evolution Video...")
    ani = animation.FuncAnimation(fig, update_evo, frames=len(snapshots), interval=100)
    ani.save("ai_design_evolution.gif", writer='pillow', fps=10)
    
    print("🚀 Done! Check 'ai_design_evolution.gif' and the plot window.")
    plt.show()

if __name__ == "__main__":
    main()