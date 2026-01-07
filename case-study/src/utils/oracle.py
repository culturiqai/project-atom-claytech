'''
D3Q19 LBM Solver

Note: This is a research-grade solver for homogeneous turbulence. 
The production wind-tunnel kernel is available for enterprise partners.
'''

import jax
# 1. CRITICAL: Enable 64-bit Precision BEFORE other JAX imports
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

# --- D3Q19 LATTICE CONSTANTS ---
# 19 Velocities: 1 Center, 6 Face, 12 Edge
CX = jnp.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
CY = jnp.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1])
CZ = jnp.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1])

# Weights: Center=1/3, Face=1/18, Edge=1/36
W = jnp.array([1/3] + [1/18]*6 + [1/36]*12)

# Opposite indices for Bounce-Back
OPPOSITE = jnp.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])

# Python Integers for unroll
CX_INT = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
CY_INT = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1]
CZ_INT = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1]

class LBMSolverJAX:
    def __init__(self, nx=128, ny=64, nz=64, precision=jnp.float64):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dtype = precision
        
        # Pre-compute broadcast shapes for 3D [1, 19, 1, 1, 1]
        self.CX_broad = CX.reshape(1, 19, 1, 1, 1).astype(self.dtype)
        self.CY_broad = CY.reshape(1, 19, 1, 1, 1).astype(self.dtype)
        self.CZ_broad = CZ.reshape(1, 19, 1, 1, 1).astype(self.dtype)
        self.W_broad = W.reshape(1, 19, 1, 1, 1).astype(self.dtype)

    @partial(jax.jit, static_argnums=(0,))
    def equilibrium(self, rho, u, v, w):
        rho = jnp.expand_dims(rho, axis=1)
        u = jnp.expand_dims(u, axis=1)
        v = jnp.expand_dims(v, axis=1)
        w = jnp.expand_dims(w, axis=1)

        cu = self.CX_broad * u + self.CY_broad * v + self.CZ_broad * w
        usq = u**2 + v**2 + w**2
        
        # D3Q19 Equilibrium Expansion
        feq = rho * self.W_broad * (1.0 + 3.0 * cu + 4.5 * (cu**2) - 1.5 * usq)
        return feq

    @partial(jax.jit, static_argnums=(0,))
    def apply_equilibrium_inlet(self, f, u_in):
        """
        Dirichlet Velocity Inlet for West Boundary (x=0).
        Replaces Zou-He which is algebraically expensive in 3D.
        Sets inlet population to Equilibrium(rho=1.0, u=u_in).
        """
        # Create a macroscopic velocity field for the slice
        # Shape: (Batch, 1, Ny, Nz)
        zeros_slice = jnp.zeros_like(f[:, 0:1, 0, :, :])
        rho_in = jnp.ones_like(zeros_slice)
        
        # Calculate Equilibrium for the inlet slice
        # Note: We squeeze dims to match (Batch, Ny, Nz) for eq calculation then expand back if needed
        # Or better, just utilize broadcasting on the slice
        
        # Slicing at x=0, we need shapes (Batch, Ny, Nz) for equilibrium input
        rho_slice = jnp.ones((f.shape[0], self.ny, self.nz))
        u_slice = jnp.full((f.shape[0], self.ny, self.nz), u_in)
        v_slice = jnp.zeros((f.shape[0], self.ny, self.nz))
        w_slice = jnp.zeros((f.shape[0], self.ny, self.nz))
        
        # Calculate full equilibrium
        # We need a temporary helper to avoid broadcasting issues with the class-level shapes which are 5D
        # We'll just call self.equilibrium but we need to pad inputs to 5D 
        # (Batch, Nx, Ny, Nz) -> but here Nx=1
        
        feq_slice = self.equilibrium(
            rho_slice[:, None, :, :], # Add x-dim back: (B, 1, Y, Z)
            u_slice[:, None, :, :], 
            v_slice[:, None, :, :], 
            w_slice[:, None, :, :]
        )
        
        # feq_slice is (Batch, 19, 1, Ny, Nz). Squeeze x-dim to broadcast into f
        feq_slice = feq_slice[:, :, 0, :, :]
        
        # Apply strict Dirichlet (Hard set)
        f = f.at[:, :, 0, :, :].set(feq_slice)
        return f

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, params):
        f_pop = state
        mask, u_in, rho_global, tau = params
        
        # 1. MACROSCOPIC MOMENTS (3D)
        rho = jnp.sum(f_pop, axis=1)
        ux  = (jnp.sum(f_pop * self.CX_broad, axis=1) / (rho + 1e-15))
        uy  = (jnp.sum(f_pop * self.CY_broad, axis=1) / (rho + 1e-15))
        uz  = (jnp.sum(f_pop * self.CZ_broad, axis=1) / (rho + 1e-15))

        # 2. COLLISION (TRT - Valid for D3Q19)
        f_opp = f_pop[:, OPPOSITE, :, :, :]
        f_plus  = 0.5 * (f_pop + f_opp)
        f_minus = 0.5 * (f_pop - f_opp)
        
        feq = self.equilibrium(rho, ux, uy, uz)
        feq_plus  = 0.5 * (feq + feq[:, OPPOSITE, :, :, :])
        feq_minus = 0.5 * (feq - feq[:, OPPOSITE, :, :, :])
        
        omega_plus = 1.0 / tau
        # Magic Lambda = 1/4 for stability
        omega_minus = 1.0 / (0.25 / (1.0/omega_plus - 0.5) + 0.5)
        
        f_col = (f_plus  - omega_plus  * (f_plus - feq_plus)) + \
                (f_minus - omega_minus * (f_minus - feq_minus))

        # 3. STREAMING (3D Roll)
        f_stream = jnp.zeros_like(f_col)
        for i in range(19):
             shift_x = int(CX_INT[i])
             shift_y = int(CY_INT[i])
             shift_z = int(CZ_INT[i])
             
             # Roll over axes 1, 2, 3 (x, y, z)
             f_stream = f_stream.at[:, i, :, :, :].set(
                 jnp.roll(f_col[:, i, :, :, :], shift=(shift_x, shift_y, shift_z), axis=(1, 2, 3))
             )

        # 4. BOUNDARIES
        # A. Equilibrium Inlet (West)
        f_stream = self.apply_equilibrium_inlet(f_stream, u_in)
        
        # B. Outlet (Neumann) - Copy x=-2 to x=-1
        f_stream = f_stream.at[:, :, -1, :, :].set(f_stream[:, :, -2, :, :])
        
        # C. Obstacle (Simple Bounce-Back)
        f_bounced = f_stream[:, OPPOSITE, :, :, :] 
        
        # Mask needs to be (Batch, 1, X, Y, Z)
        mask_exp = jnp.expand_dims(mask, 1)
        f_next = f_stream * (1 - mask_exp) + f_bounced * mask_exp

        return f_next, (rho, ux, uy, uz)

    @partial(jax.jit, static_argnums=(0, 2))
    def run_simulation(self, initial_state, num_steps, params):
        def scan_fn(carry, _):
            f_pop = carry
            f_new, observables = self.step(f_pop, params)
            return f_new, observables 

        final_state, _ = jax.lax.scan(scan_fn, initial_state, jnp.arange(num_steps))
        
        # Final Macros
        f_pop = final_state
        rho = jnp.sum(f_pop, axis=1)
        ux = jnp.sum(f_pop * self.CX_broad, axis=1) / rho
        uy = jnp.sum(f_pop * self.CY_broad, axis=1) / rho
        uz = jnp.sum(f_pop * self.CZ_broad, axis=1) / rho
        
        return final_state, (ux, uy, uz, rho)

# --- USAGE DEMO (Updated for 3D) ---
def demo_run():
    print("Initializing JAX-LBM D3Q19 (3D Mode)...")
    
    batch_size = 1
    nx, ny, nz = 100, 50, 50
    steps = 500
    solver = LBMSolverJAX(nx, ny, nz)
    
    # Init Data 
    rho_init = jnp.ones((batch_size, nx, ny, nz))
    u_init = jnp.full((batch_size, nx, ny, nz), 0.05)
    v_init = jnp.zeros((batch_size, nx, ny, nz))
    w_init = jnp.zeros((batch_size, nx, ny, nz))
    
    f_init = solver.equilibrium(rho_init, u_init, v_init, w_init)

    # Obstacle (Sphere in 3D)
    # JAX meshgrid for 3D
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    z = jnp.arange(nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Sphere center at (nx/4, ny/2, nz/2)
    radius = ny // 6
    mask = ((X - nx//4)**2 + (Y - ny//2)**2 + (Z - nz//2)**2 < radius**2).astype(jnp.float64)
    mask = jnp.repeat(mask[None, ...], batch_size, axis=0) 
    
    # Params: mask, u_inlet, rho_ref, tau
    params = (mask, 0.05, 1.0, 0.54)
    
    print(f"Starting 3D Simulation (Grid: {nx}x{ny}x{nz}, Steps: {steps})...")
    import time
    t0 = time.time()
    final_f, (ux, uy, uz, rho) = solver.run_simulation(f_init, steps, params)
    ux.block_until_ready()
    t1 = time.time()

# ... inside demo_run, after t1 = time.time() ...

    # Visualize a 2D slice of the 3D wake (Middle of Z-axis)
    mid_z = nz // 2
    velocity_mag = jnp.sqrt(ux**2 + uy**2 + uz**2)
    slice_2d = velocity_mag[0, :, :, mid_z] # Shape (Nx, Ny)

    plt.figure(figsize=(10, 5))
    # Transpose for correct (x, y) orientation in plot
    plt.imshow(slice_2d.T, cmap='jet', origin='lower')
    plt.colorbar(label='Velocity Magnitude')
    plt.title(f'D3Q19 Mid-Slice (Z={mid_z})')
    plt.savefig("d3q19_output.png")
    print("Slice saved to d3q19_output.png")
        
    # Check max velocity magnitude
    vel_mag = jnp.sqrt(ux**2 + uy**2 + uz**2)
    max_vel = jnp.max(vel_mag)
    
    print(f"✅ Simulation Complete in {t1-t0:.4f}s")
    print(f"Max 3D Velocity: {max_vel:.6f}")
    print(f"Mean Density: {jnp.mean(rho):.6f}")

    



if __name__ == "__main__":
    demo_run()
