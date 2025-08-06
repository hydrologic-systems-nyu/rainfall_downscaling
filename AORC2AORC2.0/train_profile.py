import torch
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
import numpy as np
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from fss import fss_batch
import xarray as xr
import pandas as pd
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

def check_model_dtype(model):
    dtypes = set()
    for name, param in model.named_parameters():
        dtypes.add(param.dtype)
    return dtypes


def train(input_, target_):
    ''' parameter: epochs, loss_function, optimizer, batch_size. '''
    epochs=4
    batch_size=1
    scaler = torch.amp.GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    D = Discriminator().to(device)
    gen_opt = optim.AdamW(G.parameters(), lr=1e-4, betas=(0.0, 0.999))
    disc_opt = optim.AdamW(D.parameters(), lr=2e-4, betas=(0.0, 0.5))
    full_step = 0
    for epoch in range(0, epochs):
        perm = np.random.permutation(input_.shape[0])
        x_lr = input_[perm]
        x_hr = target_[perm]
        G.train()
        D.train()
        loss_epoch = 0
        for i in tqdm(range(0, 1, batch_size)):
            input_image = torch.as_tensor(x_lr[i:i+batch_size], device=device).float()            
            target = torch.as_tensor(x_hr[i:i+batch_size], device=device).float()
            gen_opt.zero_grad(set_to_none=True)
            # === 1. Train Generator ===
            #with torch.autocast(device_type="cuda", dtype=torch.float16):
                ## generate multiple ensemble prediction-
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                with record_function("model_inference"):
                    o = G(input_image)
                    print(check_model_dtype(G))
            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=30))
            with open("profiler_output.txt", "w") as f:
                f.write(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=100))
            with open("profiler_output.csv", "w") as f:
                f.write(prof.key_averages().table(
                    sort_by="cuda_memory_usage",  # or "self_cuda_memory_usage"
                    row_limit=100))
            return




