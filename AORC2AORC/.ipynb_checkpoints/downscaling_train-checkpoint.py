import torch
from model import Generator  
from train import train, validation
#from train_profile import train
from model import Generator, Discriminator
import torch.optim as optim

from data import readFile, make_temporal_batches
import xarray as xr
import numpy as np
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_data(year):

    input = readFile(f'/scratch/jl14811/AORC_1981-2011/AORC_21_31/APCP_surface_{year}*.nc', 'APCP_surface', 24, 21, 31)
    input = input.reshape(-1, 1, 24, 21, 31).sum(axis=2) / 24
    mean_val = np.nanmean(input)
    input = np.where(np.isnan(input), 0.0, input)
    print(input.shape)
    
    target = readFile(f'/scratch/jl14811/AORC_1981-2011/AORC_126_186/APCP_surface_{year}*.nc', 'APCP_surface', 24, 126, 186)
    target = target.reshape(-1, 4, 6, 126, 186).sum(axis=2) / 6
    mean_val = np.nanmean(target)
    target = np.where(np.isnan(target), 0.0, target)
    print(target.shape)
    
    
    input, target = make_temporal_batches(input, target, True)
    print(input.shape)
    print(target.shape)
    return input, target



''' parameter: epochs, loss_function, optimizer, batch_size. '''
epochs=60
batch_size=4
scaler = torch.amp.GradScaler()
criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)
gen_opt = optim.AdamW(G.parameters(), lr=1e-4, betas=(0.0, 0.999))
disc_opt = optim.AdamW(D.parameters(), lr=2e-4, betas=(0.0, 0.5))

for epoch in range(0, epochs):
    for year in range(1981, 2012, 1):
        loss_ = 0.0
        print(f'This is the {epoch}, and it is training the year of {year}.')
        input, target = get_data(year)
        loss_epoch = train(G, D, batch_size, gen_opt, disc_opt, scaler, criterion, input, target, device)
        loss_ = loss_ + loss_epoch
        print(f'epoch:{epoch}, loss:{loss_:.5f}')
        












