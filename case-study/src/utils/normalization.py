import torch
import torch.nn as nn

class UnitGaussianNormalizer:
    """
    Normalizes data to zero mean and unit variance.
    Essential for FNO performance.
    """
    def __init__(self, x, eps=1e-5):
        # x shape: (N_samples, *spatial_dims, N_features)
        # Compute mean and std along sample dimension (dim 0)
        self.mean = torch.mean(x, dim=0, keepdim=True)
        self.std = torch.std(x, dim=0, keepdim=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)

    def decode(self, x):
        return x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

class NormalizedModel(nn.Module):
    """
    Wraps a model to automatically normalize inputs and denormalize outputs.
    """
    def __init__(self, model, x_normalizer, y_normalizer):
        super().__init__()
        self.model = model
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        # Expose context_dim if available in the wrapped model
        if hasattr(model, 'context_dim'):
            self.context_dim = model.context_dim

    def forward(self, x, context=None):
        x_norm = self.x_normalizer.encode(x)
        # Context is usually not normalized or normalized separately. 
        # For now assuming context is already suitable or handled by model.
        out_norm = self.model(x_norm, context)
        
        if isinstance(out_norm, tuple):
            # Assume (mean, sigma)
            mean_norm, sigma_norm = out_norm
            mean = self.y_normalizer.decode(mean_norm)
            # Sigma should be scaled by std, but NOT shifted by mean
            sigma = sigma_norm * (self.y_normalizer.std.to(sigma_norm.device) + self.y_normalizer.eps)
            return mean, sigma
        else:
            return self.y_normalizer.decode(out_norm)

