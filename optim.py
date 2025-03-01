### Done ###
import torch

def create_optimizer(name, net, lr):
    if name == 'adam':
        return torch.optim.Adam(
            params=net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-4,
            amsgrad=False
        )
    elif name == 'adamw':
        return torch.optim.AdamW(
            params=net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-4,
            amsgrad=False
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")