import torch
import platform

def get_device() : 
    if platform.system() == "Darwin" :
        if torch.backends.mps.is_available() :
            return torch.device("mps")
    elif torch.cuda.is_available() :
        return torch.device("cuda")
    else :
        raise RuntimeError("No valid gpu device found")

device = get_device()
x = torch.zeros((3,3))
x = x.to(device)
print(x)