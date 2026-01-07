"""
JAX-LBM: Differentiable, Batched, GPU-Accelerated Fluid Solver.
Date: Dec 7, 2025

Note: This is a research-grade solver for homogeneous turbulence. 
The production wind-tunnel kernel is available for enterprise partners.

Architecture:
- Method: D2Q9 Lattice Boltzmann Method (TRT Collision + Zou-He Inlet)
- Implementation: Pure JAX (Functional)
- Optimization: XLA JIT Compilation + VMAP for Batching
- Precision: Float64 (High-Fidelity)
"""

# 1. CRITICAL: Enable 64-bit Precision BEFORE other JAX imports
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from functools import partial

# --- D2Q9 LATTICE CONSTANTS ---
CX = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
W = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
OPPOSITE = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Python Integers for unroll
CX_INT = [0, 1, 0, -1, 0, 1, -1, -1, 1]
CY_INT = [0, 0, 1, 0, -1, 1, 1, -1, -1]

class LBMSolverJAX:
    def __init__(self, nx=128, ny=128, precision=jnp.float64):
        self.nx = nx
        self.ny = ny
        self.dtype = precision
        
        # Pre-compute broadcast shapes for performance in 'step'
        self.CX_broad = CX.reshape(1, 9, 1, 1).astype(self.dtype)
        self.CY_broad = CY.reshape(1, 9, 1, 1).astype(self.dtype)
        self.W_broad = W.reshape(1, 9, 1, 1).astype(self.dtype)

    @partial(jax.jit, static_argnums=(0,))
    def equilibrium(self, rho, u, v):
        rho = jnp.expand_dims(rho, axis=1)
        u = jnp.expand_dims(u, axis=1)
        v = jnp.expand_dims(v, axis=1)

        cu = self.CX_broad * u + self.CY_broad * v
        usq = u**2 + v**2
        feq = rho * self.W_broad * (1.0 + 3.0 * cu + 4.5 * (cu**2) - 1.5 * usq)
        return feq

    @partial(jax.jit, static_argnums=(0,))
    def apply_zou_he_inlet(self, f, u_in):
        """
        Zou-He Velocity Inlet for West Boundary (x=0).
        """
        # Extract populations at x=0
        f0, f2, f3, f4, f6, f7 = f[:, 0, 0, :], f[:, 2, 0, :], f[:, 3, 0, :], f[:, 4, 0, :], f[:, 6, 0, :], f[:, 7, 0, :]
        
        # Density at inlet
        rho_in = (f0 + f2 + f4 + 2 * (f3 + f6 + f7)) / (1.0 - u_in)

        # Solve for unknowns (Right-moving: 1, 5, 8)
        f1_new = f3 + (2.0/3.0) * rho_in * u_in
        f5_new = f7 - 0.5 * (f2 - f4) + (1.0/6.0) * rho_in * u_in
        f8_new = f6 + 0.5 * (f2 - f4) + (1.0/6.0) * rho_in * u_in

        # Update
        f = f.at[:, 1, 0, :].set(f1_new)
        f = f.at[:, 5, 0, :].set(f5_new)
        f = f.at[:, 8, 0, :].set(f8_new)
        return f

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, params):
        f_pop = state
        mask, u_in, rho_global, tau = params
        
        # 1. MACROSCOPIC MOMENTS
        rho = jnp.sum(f_pop, axis=1)
        # Avoid div by zero with eps
        ux  = (jnp.sum(f_pop * self.CX_broad, axis=1) / (rho + 1e-15))
        uy  = (jnp.sum(f_pop * self.CY_broad, axis=1) / (rho + 1e-15))

        # 2. COLLISION (TRT)
        f_opp = f_pop[:, OPPOSITE, :, :]
        f_plus  = 0.5 * (f_pop + f_opp)
        f_minus = 0.5 * (f_pop - f_opp)
        
        feq = self.equilibrium(rho, ux, uy)
        feq_plus  = 0.5 * (feq + feq[:, OPPOSITE, :, :])
        feq_minus = 0.5 * (feq - feq[:, OPPOSITE, :, :])
        
        omega_plus = 1.0 / tau
        # Magic Lambda = 1/4 for stability
        omega_minus = 1.0 / (0.25 / (1.0/omega_plus - 0.5) + 0.5)
        
        f_col = (f_plus  - omega_plus  * (f_plus - feq_plus)) + \
                (f_minus - omega_minus * (f_minus - feq_minus))

        # 3. STREAMING
        f_stream = jnp.zeros_like(f_col)
        for i in range(9):
             shift_x = int(CX_INT[i])
             shift_y = int(CY_INT[i])
             f_stream = f_stream.at[:, i, :, :].set(
                 jnp.roll(f_col[:, i, :, :], shift=(shift_x, shift_y), axis=(1, 2))
             )

        # 4. BOUNDARIES
        # A. Zou-He Inlet (West)
        f_stream = self.apply_zou_he_inlet(f_stream, u_in)
        
        # B. Outlet (Neumann)
        f_stream = f_stream.at[:, :, -1, :].set(f_stream[:, :, -2, :])
        
        # C. Obstacle (Simple Half-Way BB)
        # Bouncing back the populations that were about to enter the wall
        f_bounced = f_stream[:, OPPOSITE, :, :] 
        
        mask_exp = jnp.expand_dims(mask, 1)
        f_next = f_stream * (1 - mask_exp) + f_bounced * mask_exp

        return f_next, (rho, ux, uy)

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
        
        return final_state, (ux, uy, rho)

# --- USAGE DEMO ---
def demo_run():
    print("Initializing JAX-LBM (High-Fidelity Mode)...")
    
    batch_size = 1
    nx, ny = 200, 100
    steps = 1000
    solver = LBMSolverJAX(nx, ny)
    
    # Init Data (Perturbation to trigger flow)
    rho_init = jnp.ones((batch_size, nx, ny))
    u_init = jnp.full((batch_size, nx, ny), 0.05) # Initial slight flow
    v_init = jnp.zeros((batch_size, nx, ny))
    f_init = solver.equilibrium(rho_init, u_init, v_init)

    # Obstacle (Cylinder)
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='xy')
    # Transpose mask to match (Nx, Ny) of the solver logic if needed, or adjust meshgrid
    # LBM standard is usually (x, y). Let's assume standard matrix indexing.
    mask = ((x - nx//4)**2 + (y - ny//2)**2 < (ny//9)**2).astype(jnp.float64)
    mask = jnp.repeat(mask[None, ...], batch_size, axis=0) 
    
    # Params: mask, u_inlet, rho_ref, tau
    # Tau = 0.51 (Very low viscosity/High Re) -> Would crash BGK, should run on TRT
    params = (mask, 0.08, 1.0, 0.52)
    
    print(f"Starting Simulation (Steps: {steps}, Tau: {params[3]})...")
    import time
    t0 = time.time()
    final_f, (ux, uy, rho) = solver.run_simulation(f_init, steps, params)
    ux.block_until_ready()
    t1 = time.time()
    
    print(f"✅ Simulation Complete in {t1-t0:.4f}s")
    print(f"Max Velocity: {jnp.max(jnp.sqrt(ux**2 + uy**2))}")

if __name__ == "__main__":
    demo_run()
