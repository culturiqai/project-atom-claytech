"""
JAX-LBM Interactive Wind Tunnel ("God Mode")
Author: Ex-NVIDIA Director of Physical AI
Date: Dec 8, 2025

Instructions:
1. Run script.
2. A window will open showing the flow.
3. LEFT CLICK & DRAG to draw 'walls' (obstacles).
4. RIGHT CLICK to erase.
5. Watch the flow adapt instantly (Real-time CFD).
"""

import jax
import jax.numpy as jnp
import numpy as np
import cv2  # You need: pip install opencv-python
from lbm_oracle import LBMSolverJAX

# --- CONFIG ---
# Reduced resolution slightly for 60 FPS real-time performance on Mac
NX, NY = 200, 100
TAU = 0.53  # Very low viscosity (water-like)
INLET_VEL = 0.1

def main():
    print("⚡️ Initializing Real-Time Wind Tunnel...")
    print("👉 LEFT CLICK to Draw Walls")
    print("👉 RIGHT CLICK to Erase")
    print("👉 'Q' to Quit")

    solver = LBMSolverJAX(NX, NY)
    
    # Initialize Field
    rho = jnp.ones((1, NX, NY))
    u = jnp.full((1, NX, NY), INLET_VEL)
    v = jnp.zeros((1, NX, NY))
    state = solver.equilibrium(rho, u, v)

    # The Mask (Numpy array for easy editing with mouse)
    mask_np = np.zeros((NX, NY), dtype=np.float32)

    # Interaction State
    drawing = False
    erasing = False

    def mouse_callback(event, y, x, flags, param):
        nonlocal drawing, erasing, mask_np
        # Note: CV2 coordinates are (x, y) but array is (NX, NY)
        # We need to swap appropriately.
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False
            
        if drawing or erasing:
            # Brush size
            r = 4
            val = 1.0 if drawing else 0.0
            # Draw circle on mask
            cv2.circle(mask_np, (y, x), r, val, -1)

    # Setup Window
    cv2.namedWindow("JAX-LBM Realtime", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("JAX-LBM Realtime", mouse_callback)
    cv2.resizeWindow("JAX-LBM Realtime", 800, 400)

    # JIT the step function for speed
    step_fn = jax.jit(solver.step)
    
    # Pre-allocate static params
    # We can't JIT the mask since it changes every frame, 
    # so we pass it as a regular argument (or re-JIT? No, that's slow).
    # Trick: We pass mask as a Tracer? No.
    # We will just run step_fn with the new mask each time. 
    # Since shapes match, JAX won't recompile!
    
    print("🚀 Simulation Started!")
    
    while True:
        # 1. Prepare Mask (Push to GPU)
        mask_jax = jnp.array(mask_np)
        mask_batched = jnp.expand_dims(mask_jax, 0)
        
        # 2. Step Physics (Run 5 steps per frame for speedup)
        params = (mask_batched, INLET_VEL, 1.0, TAU)
        for _ in range(5):
            state, (rho_out, ux_out, uy_out) = step_fn(state, params)
        
        # 3. Visualization
        # Pull data to CPU
        ux = np.array(ux_out[0]).T
        uy = np.array(uy_out[0]).T
        mask_vis = mask_np.T
        
        # Compute Curl for "Trippy" visuals
        dy_ux = np.gradient(ux, axis=0)
        dx_uy = np.gradient(uy, axis=1)
        curl = dx_uy - dy_ux
        
        # Color Mapping
        # Speed (Blue/Purple)
        speed = np.sqrt(ux**2 + uy**2)
        img_speed = (np.clip(speed / 0.15, 0, 1) * 255).astype(np.uint8)
        img_color = cv2.applyColorMap(img_speed, cv2.COLORMAP_JET)
        
        # Overlay Obstacles (White)
        img_color[mask_vis > 0.5] = [255, 255, 255]
        
        # Show
        cv2.imshow("JAX-LBM Realtime", img_color)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()