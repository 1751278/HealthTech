import torch

class VLAD:
    def __init__(self, descriptor_dim=64, n_clusters=32, device='cuda'):
        self.k = n_clusters
        self.d = descriptor_dim
        self.device = device
        self.centroids = None  # (k, d), set by fit()

    def fit(self, sample_descriptors: torch.Tensor, iters=25):
        # sample_descriptors: (N, d) pooled from many frames' local descriptors
        x = sample_descriptors.to(self.device)
        idx = torch.randperm(x.shape[0])[:self.k]
        centroids = x[idx].clone()
        for _ in range(iters):
            d = torch.cdist(x, centroids)          # (N, k)
            assign = d.argmin(dim=1)
            for c in range(self.k):
                mask = assign == c
                if mask.any():
                    centroids[c] = x[mask].mean(dim=0)
        self.centroids = centroids

    def encode(self, descriptors: torch.Tensor) -> torch.Tensor:
        # descriptors: (N, d) for ONE frame -> returns a single (k*d,) global vector
        x = descriptors.to(self.device)
        d = torch.cdist(x, self.centroids)          # (N, k)
        assign = d.argmin(dim=1)
        vlad = torch.zeros(self.k, self.d, device=self.device)
        for c in range(self.k):
            mask = assign == c
            if mask.any():
                vlad[c] = (x[mask] - self.centroids[c]).sum(dim=0)
        vlad = torch.sign(vlad) * torch.sqrt(vlad.abs() + 1e-12)  # power-norm
        vlad = vlad.flatten()
        return vlad / (vlad.norm() + 1e-12)                       # L2-norm