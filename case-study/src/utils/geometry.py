"""
Geometry utilities for building mask generation.

This module converts building parameters into binary masks for LBM simulation.

CRITICAL DESIGN CONSTRAINT (from plan.md):
All functions MUST return generic (128, 128) binary masks.
The AI should NEVER see raw parameters - only pixel bitmaps.
This ensures adding new shapes requires ZERO FNO architecture changes.
"""

import torch
import numpy as np


def generate_building_mask(width, depth, rotation=0.0, shape_type='rectangle', nx=128, ny=128):
    """
    Generate binary mask for a building.
    
    ⚠️ CRITICAL: Output is ALWAYS a (128, 128) binary tensor.
    This abstraction allows adding complex shapes without changing the AI.
    
    Args:
        width: Building width in grid units (10-60)
        depth: Building depth in grid units (10-60)
        rotation: Rotation angle in degrees (0-360)
        shape_type: 'rectangle', 'circle', 'l_shape'
        nx, ny: Grid resolution (default 128x128)
    
    Returns:
        mask: (nx, ny) binary tensor (1 = building, 0 = air)
    """
    if shape_type == 'rectangle':
        return _rectangle(width, depth, rotation, nx, ny)
    elif shape_type == 'circle':
        return _circle(width, nx, ny)
    elif shape_type == 'l_shape':
        return _l_shape(width, depth, rotation, nx, ny)
    else:
        raise ValueError(f"Unknown shape: {shape_type}")


def _rectangle(width, depth, rotation, nx, ny):
    """Rectangle placed at center of domain."""
    mask = torch.zeros((nx, ny), dtype=torch.float32)
    
    # Center coordinates
    cx, cy = nx // 2, ny // 2
    
    # Half dimensions
    hw, hd = int(width // 2), int(depth // 2)
    
    # Create meshgrid
    x = torch.arange(nx).float() - cx
    y = torch.arange(ny).float() - cy
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Apply rotation
    angle_rad = rotation * np.pi / 180.0
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    X_rot = cos_a * X - sin_a * Y
    Y_rot = sin_a * X + cos_a * Y
    
    # Check if inside rectangle
    inside = (torch.abs(X_rot) <= hw) & (torch.abs(Y_rot) <= hd)
    mask[inside] = 1.0
    
    return mask


def _circle(diameter, nx, ny):
    """Circle placed at center of domain."""
    mask = torch.zeros((nx, ny), dtype=torch.float32)
    
    # Center coordinates
    cx, cy = nx // 2, ny // 2
    radius = diameter / 2
    
    # Distance from center
    x = torch.arange(nx).float()
    y = torch.arange(ny).float()
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)
    
    mask[dist <= radius] = 1.0
    
    return mask


def _l_shape(width, depth, rotation, nx, ny):
    """
    L-shaped building (two rectangles joined).
    
    Used for testing asymmetric shapes.
    """
    mask = torch.zeros((nx, ny), dtype=torch.float32)
    
    # First rectangle (horizontal part of L)
    mask1 = _rectangle(width, depth // 2, rotation, nx, ny)
    
    # Second rectangle (vertical part of L)
    # Offset to create L shape
    cx, cy = nx // 2, ny // 2
    offset = int(width // 4)
    
    mask2 = torch.zeros_like(mask)
    x = torch.arange(nx).float() - (cx + offset)
    y = torch.arange(ny).float() - cy
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    hw, hd = int(width // 4), int(depth // 2)
    inside = (torch.abs(X) <= hw) & (torch.abs(Y) <= hd)
    mask2[inside] = 1.0
    
    # Combine (union)
    mask = torch.clamp(mask1 + mask2, max=1.0)
    
    return mask


def visualize_mask(mask, title="Building Mask"):
    """
    Helper to visualize a building mask.
    
    Args:
        mask: (nx, ny) tensor
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask.cpu().numpy().T, origin='lower', cmap='binary')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Building (1) / Air (0)")
    plt.tight_layout()
    plt.show()


# Test cases
if __name__ == "__main__":
    print("Testing geometry.py...")
    
    # Test rectangle
    rect_mask = generate_building_mask(40, 30, rotation=15, shape_type='rectangle')
    print(f"Rectangle mask: {rect_mask.shape}, sum={rect_mask.sum().item()}")
    
    # Test circle
    circ_mask = generate_building_mask(40, 40, shape_type='circle')
    print(f"Circle mask: {circ_mask.shape}, sum={circ_mask.sum().item()}")
    
    # Test L-shape
    l_mask = generate_building_mask(40, 30, rotation=0, shape_type='l_shape')
    print(f"L-shape mask: {l_mask.shape}, sum={l_mask.sum().item()}")
    
    print("\n✅ All shapes generated successfully!")
