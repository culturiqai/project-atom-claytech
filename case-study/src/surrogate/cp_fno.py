"""
Physics-Aware FNO Architecture.
FIXED: Strict Helmholtz Head (Stream Function Only) for Zero Divergence.
FIXED: Forward pass now uses Spectral Convs and correct Permutations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.surrogate.equivariant import EquivariantSpectralConv2d

class SteerableLinear(nn.Module):
    """
    Rotation-Equivariant Linear Layer for Vectors.
    Maps Vector -> Vector using the structure: W = [[a, -b], [b, a]]
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features % 2 == 0 and out_features % 2 == 0
        
        self.in_pairs = in_features // 2
        self.out_pairs = out_features // 2
        
        scale = 1 / (in_features * out_features)
        self.weights = nn.Parameter(scale * torch.randn(self.out_pairs, self.in_pairs, 2))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        input_shape = x.shape
        x_pairs = x.view(*input_shape[:-1], self.in_pairs, 2)
        
        a = self.weights[..., 0] 
        b = self.weights[..., 1]
        
        x_u = x_pairs[..., 0]
        x_v = x_pairs[..., 1]
        
        y_u = torch.einsum('...j,oj->...o', x_u, a) - torch.einsum('...j,oj->...o', x_v, b)
        y_v = torch.einsum('...j,oj->...o', x_u, b) + torch.einsum('...j,oj->...o', x_v, a)
        
        y = torch.stack([y_u, y_v], dim=-1) 
        y = y.view(*input_shape[:-1], -1) + self.bias
        return y

class SteerableLifting(nn.Module):
    """
    Lifts Mask (Scalar) and Wind (Vector) preserving symmetry.
    """
    def __init__(self, out_width):
        super().__init__()
        self.out_width = out_width
        self.scalar_lift = nn.Linear(1, out_width // 2)
        self.vector_lift = SteerableLinear(2, out_width // 2)

    def forward(self, x):
        # x: (B, X, Y, 3)
        mask = x[..., 0:1]
        wind = x[..., 1:3]
        
        h_scalar = self.scalar_lift(mask)
        h_vector = self.vector_lift(wind)
        
        # Output: (B, X, Y, Width)
        return torch.cat([h_scalar, h_vector], dim=-1)

class HelmholtzHead(nn.Module):
    """
    Physics Enforcer: Outputs ONLY Stream Function (psi).
    Velocity is reconstructed as: u = curl(psi)
    """
    def __init__(self, in_width):
        super().__init__()
        self.proj = nn.Linear(in_width, 1) 
        
    def forward(self, x):
        # x: (B, NX, NY, C)
        psi = self.proj(x).squeeze(-1) # (B, NX, NY)
        
        # Pad to handle boundaries (Replicate = Neumann BC approx)
        psi_pad = F.pad(psi, (1,1,1,1), mode='replicate')
        
        # Central Differences
        dy_psi = (psi_pad[:, 2:, 1:-1] - psi_pad[:, :-2, 1:-1]) / 2.0
        dx_psi = (psi_pad[:, 1:-1, 2:] - psi_pad[:, 1:-1, :-2]) / 2.0
        
        u = dy_psi
        v = -dx_psi
        
        return torch.stack([u, v], dim=-1)

class EquivariantCausalFNO(nn.Module):
    def __init__(self, modes_x=12, modes_y=12, width=32, context_dim=0, 
                 in_channels=3, out_channels=3, depth=4):
        super().__init__()
        self.width = width
        self.depth = depth
        
        # 1. Lifting
        self.lifting = SteerableLifting(width)
        
        # 2. Backbone (Spectral Convs)
        self.spectral_convs = nn.ModuleList([
            EquivariantSpectralConv2d(width, width, modes_x, modes_y, 
                                      context_dim=context_dim) 
            for _ in range(depth)
        ])
        # Skip connection weights
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.InstanceNorm2d(width) for _ in range(depth)])
        self.activation = nn.GELU()
        
        # 3. Heads
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 128) # Extra layer for capacity
        
        # Physics Head (Strict Helmholtz) for U, V
        self.helmholtz = HelmholtzHead(128)
        # Scalar Head for Rho
        self.fc_rho = nn.Linear(128, 1)
        # Uncertainty
        self.fc_sigma = nn.Linear(128, 3)

    def forward(self, x, context=None):
        # x: (B, nx, ny, 3)
        
        # 1. Lifting -> (B, nx, ny, width)
        x = self.lifting(x)
        
        # 2. Backbone (FNO Blocks)
        # Iterate through layers
        for i in range(self.depth):
            x_in = x
            
            # Spectral Conv expects (B, nx, ny, C)
            x_spec = self.spectral_convs[i](x, context=context)
            
            # Linear Skip (Conv1x1) - Needs (B, C, nx, ny)
            x_skip = x.permute(0, 3, 1, 2)
            x_skip = self.ws[i](x_skip)
            x_skip = x_skip.permute(0, 2, 3, 1)
            
            # Combine
            x = x_spec + x_skip
            
            # Norm & Activation (Norm needs B, C, X, Y)
            x = x.permute(0, 3, 1, 2)
            x = self.norms[i](x)
            x = x.permute(0, 2, 3, 1)
            
            x = self.activation(x)
        
        # 3. Projection Heads
        # x is (B, nx, ny, width)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        # Predict Vector Field via Stream Function
        pred_uv = self.helmholtz(x)
        pred_rho = self.fc_rho(x)
        pred_sigma = F.softplus(self.fc_sigma(x)) + 1e-6
        
        # --- GLOBAL CONSERVATION FIX (CP-FNO) ---
        if context is not None and 'total_mass' in context:
            current_mean = torch.mean(pred_rho, dim=(1, 2), keepdim=True)
            target_mean = context['total_mass'].view(-1, 1, 1, 1)
            pred_rho = pred_rho - current_mean + target_mean
        # ----------------------------------------
        
        # Combine [Ux, Uy, Rho]
        pred_mean = torch.cat([pred_uv, pred_rho], dim=-1)
        
        return pred_mean, pred_sigma
