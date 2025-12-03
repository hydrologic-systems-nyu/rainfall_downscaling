import os
import torch
from model import Generator
from train import train
#from train_profile import train
from model import Generator, Discriminator
import torch.optim as optim
from data import readFile, make_temporal_batches
import xarray as xr
import numpy as np
from einops import rearrange
import datetime
import matplotlib.pyplot as plt
import matplotlib
import argparse



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_data(year, end_year, cotinuous_year, input_source, target_source, mode, temporal_factor, spatial_factor):
    input = []
    if mode == 'GCM':
        for y in range(year, year + cotinuous_year):
            if y > end_year:
                break
            x = readFile(f'{input_source}/pr_{y}*.nc', 'pr', 1, 21, 31) * 3600
            if x.shape[0] == 366:
                x = np.delete(x, 59, axis=0)
            input.append(x)
    else:
        for y in range(year, year + cotinuous_year):
            if y > end_year:
                break
            x = readFile(f'{input_source}/APCP_surface_{y}*.nc', 'APCP_surface', 24, 21, 31)
            x = x.reshape(-1, 1, 24, 21, 31).sum(axis=2) / 24
            if x.shape[0] == 366:
                x = np.delete(x, 59, axis=0)
            input.append(x)
    input = np.concatenate(input, axis=0)
    mean_val = np.nanmean(input)
    input = np.where(np.isnan(input), 0.0, input)
    print(input.shape)

    target = []
    for y in range(year, year + cotinuous_year):
        if y > end_year:
            break
        x = readFile(f'{target_source}/APCP_surface_{y}*.nc', 'APCP_surface', 24, 126, 186)
        x = x.reshape(-1, temporal_factor, 24 // temporal_factor, 126, 186).sum(axis=2) / (24 // temporal_factor)
        ############### coarsen ###############
        factor = 6 // spatial_factor
        B, C, H, W = x.shape
        H_new, W_new = H // factor, W // factor
        x = x.reshape(B, C, H_new, factor, W_new, factor).mean(axis=(3,5))
        #######################################
        if x.shape[0] == 366:
            x = np.delete(x, 59, axis=0)
        target.append(x)
    target = np.concatenate(target, axis=0)
    mean_val = np.nanmean(target)
    target = np.where(np.isnan(target), 0.0, target)
    print(target.shape)
    
    
    input, target = make_temporal_batches(input, target, temporal_factor)
    print(input.shape)
    print(target.shape)
    return input, target


def validation(G, input_image, target, validation_file, start_date):
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
    input_image = input_image.cpu().detach().numpy()
    input_T = input_image.shape[2]
    output_T = target.shape[2] 
    end_date = start_date + datetime.timedelta(days=2)
    filename = f"{start_date}_{end_date}.nc"
    os.makedirs(f"/scratch/jl14811/{validation_file}", exist_ok=True)
    ds_input = xr.Dataset(
    {
        "pr": (("time", "lat", "lon"), input_image[0, 0][1:-1]) 
    },
    )
    ds_input.to_netcdf(f"/scratch/jl14811/{validation_file}/input_{filename}")
    ds_output = xr.Dataset(
    {
        "pr": (("time", "lat", "lon"), ensemble_output[0, 0]) 
    },
    )
    ds_output.to_netcdf(f"/scratch/jl14811/{validation_file}/output_{filename}")
    ds_target = xr.Dataset(
    {
        "pr": (("time", "lat", "lon"), target[0, 0]) 
    },
    )
    ds_target.to_netcdf(f"/scratch/jl14811/{validation_file}/target_{filename}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAN model validation with arguments")
    parser.add_argument('--start_year', type=int, required=True, help='Year to process')
    parser.add_argument('--end_year', type=int, required=True, help='Year to process')
    parser.add_argument('--cotinuous_year', type=int, required=True, help='Continuous Years')
    parser.add_argument('--input_source', type=str, required=True, help='Input data source directory')
    parser.add_argument('--target_source', type=str, required=True, help='Target data source directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('--step', type=int, default=3, help='Date increment step in days (default=3)')
    parser.add_argument('--mode', type=str, default='AORC', help='Date increment step in days (default=3)')
    parser.add_argument('--validation_file', type=str, default='validation', help='Date increment step in days (default=3)')


    
    parser.add_argument('--temporal_factor', type=int, default=1, help='temporal_downscale_factor')
    parser.add_argument('--spatial_factor', type=int, default=1, help='spatial_downscale_factor')

    

    args = parser.parse_args()
    model = Generator(temporal_factor=args.temporal_factor, spatial_factor=args.spatial_factor).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    for year in range(args.start_year, args.end_year + 1, args.cotinuous_year):
        print(year)
        start_date = datetime.date(year, 1, 2)
        input, target = get_data(year, args.end_year, args.cotinuous_year, args.input_source, args.target_source, args.mode, 
                                 args.temporal_factor, args.spatial_factor)
        x = torch.tensor(input, device=device).float()
        y = torch.tensor(target, device=device).float()
        for i in range(0, x.shape[0], 1):
            validation(model, x[i:i+1], y[i:i+1], args.validation_file, start_date)
            start_date = start_date + datetime.timedelta(days=3)


