import torch

def get_best_device():
    if torch.cuda.is_available():
        # covers both NVIDIA CUDA and AMD ROCm
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # covers apple silicon mps
        return torch.device("mps") 
    else:
        return torch.device("cpu")
