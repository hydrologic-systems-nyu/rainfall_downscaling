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

def weighted_l1_loss(pred, target, rain_threshold=1.0, rain_weight=5.0, dry_weight=1.0):
    # Create weight mask: more weight if target has rain
    weight = torch.where(target > rain_threshold,
                         torch.tensor(rain_weight, device=target.device),
                         torch.tensor(dry_weight, device=target.device))
    
    loss = (weight * torch.abs(pred - target)).mean()
    return loss

def validation(G, input_image, target):
    ensemble_size = 1
    batch_split = 1
    G.eval()
    outputs = []

    for _ in range(ensemble_size):
        batch_outputs = []
        for i in range(0, input_image.size(0), batch_split):
            batch = input_image[i:i+batch_split]
            with torch.no_grad():
                out = G(batch)
            batch_outputs.append(out)
        outputs.append(torch.cat(batch_outputs, dim=0))

    # cat along dimension 1 for ensemble
    target = target.cpu().detach().numpy()
    ensemble_output = torch.cat(outputs, dim=1)
    ensemble_output = ensemble_output.cpu().detach().numpy()


    # save the output to nc file
    da = xr.DataArray(
    ensemble_output[0, 0],
    dims=["time", "lat", "lon"],
    name="pr")  # or temperature, etc.)

    # Convert to Dataset if needed
    ds = da.to_dataset()

    # Save to NetCDF
    ds.to_netcdf("output_2011_new.nc")

    fss = fss_batch(ensemble_output, target)
    print(f'fss: {fss:.5f}')

    input_image = input_image.cpu().detach().numpy()

    input_T = input_image.shape[2]
    output_T = target.shape[2] 
    fig, ax = plt.subplots(3, output_T, figsize=(2 * output_T, 8))

    for t in range(input_T):
        ax[0, t].imshow(input_image[0, 0, t], cmap='turbo')
        ax[0, t].set_title(f'input t={t}')
        ax[0, t].axis('off')

    for t in range(output_T):
        ax[1, t].imshow(target[0, 0, t], vmin=0, vmax=10, cmap='turbo')
        ax[1, t].set_title(f'target t={t}')
        ax[1, t].axis('off')
        
    for t in range(output_T):
        ax[2, t].imshow(ensemble_output[0, 0, t], vmin=0, vmax=10, cmap='turbo')
        ax[2, t].set_title(f'output t={t}')
        ax[2, t].axis('off')

    plt.tight_layout()
    plt.show()

def train(input_, target_):
    ''' parameter: epochs, loss_function, optimizer, batch_size. '''
    epochs=100
    batch_size=2
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
        for i in tqdm(range(0, len(x_lr), batch_size)):
            input_image = torch.as_tensor(x_lr[i:i+batch_size], device=device).float()            
            target = torch.as_tensor(x_hr[i:i+batch_size], device=device).float()
            gen_opt.zero_grad(set_to_none=True)
            
            # === 1. Train Generator ===
            with torch.autocast(device_type="cuda", dtype=torch.float16):

                ## generate multiple ensemble prediction-
                gen_ensemble = torch.cat([G(input_image) for _ in range(2)], dim=1)

                pred = gen_ensemble[:,0:1] # single prediction

                # calculate ensemble mean
                ensemble_mean = torch.sum(gen_ensemble, dim=1, keepdim=True) / 2
                
                # Classify all fake batch with D
                disc_fake_output = D(input_image, ensemble_mean)
                gen_gan_loss = criterion(disc_fake_output, torch.ones_like(disc_fake_output))
                
                # can also be changed to crps loss, using the indv. ensemble members
                l1loss = weighted_l1_loss(ensemble_mean, target)
                #l1loss = nn.L1Loss()(ensemble_mean, target)
                loss_epoch = loss_epoch + l1loss

                loss = (l1loss * 1.0 + gen_gan_loss)
            scaler.scale(loss).backward()
            scaler.step(gen_opt)
            scaler.update()


            
            # === 2. Train Discriminator ===
            disc_opt.zero_grad(set_to_none=True)
            ensemble_mean = ensemble_mean.detach()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                
                # discriminator prediction
                disc_real_output = D(input_image, target)
                disc_real = criterion(disc_real_output, torch.ones_like(disc_real_output))

                
                # Classify all fake batch with D
                disc_fake_output = D(input_image, ensemble_mean) # 
                # Calculate D's loss on the all-fake batch
                disc_fake = criterion(disc_fake_output, torch.zeros_like(disc_fake_output))


            scaler.scale(disc_fake + disc_real).backward()
            # Gradient Norm Clipping
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=2.0, norm_type=2)
            scaler.step(disc_opt)
            scaler.update()
            full_step = full_step + 1
            if full_step % (100) == 0:
                validation(G, input_image, target)
                torch.save(G.state_dict(), 'model_weights/models_gcm_aorc.pth')
        print(f'epoch:{epoch}, loss:{loss_epoch:.5f}')





