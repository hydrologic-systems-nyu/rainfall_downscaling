import xarray as xr
import numpy as np
import glob

def readInput(data_file, value, t, h, w):
    nc_files = sorted(glob.glob(data_file)) # read and sort data by time

    all_data = []

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        data = ds[value].values
        all_data.append(data)

    all_data = np.array(all_data)  # shape: (num_files, 1, W, H)

    days = all_data.shape[0] // t

    # reshape
    final_data = all_data[:days*24].reshape(days, t, h, w)

    print("final shape", final_data.shape)
    
    return final_data

def readFile(data_file, value, t, h, w):
    all_data = []
    nc_files = sorted(glob.glob(data_file))
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        data = ds[value].values  # shape: (time, H, W)
        # reshape (time → days, 24, H, W)
        time = data.shape[0]
        if time % t != 0:
            continue
    
        reshaped = data.reshape(-1, t, h, w)
        all_data.append(reshaped)
    
    final_data = np.concatenate(all_data, axis=0)
    print("final shape", final_data.shape)
    return final_data

def make_temporal_batches(x, y):
    window_size = 3
    stride = 3
    Tx, Cx, Hx, Wx = x.shape
    Ty, Cy, Hy, Wy = y.shape
    num_windows = (Tx - window_size) // stride + 1
    windowed_x = []
    windowed_y = []

    for i in range(0, Tx - window_size + 1, stride):
        window_x = x[i:i+window_size] # shape: (3, 6, w, h)
        window_x = window_x.reshape(18, Hx, Wx) # shape: (18, w, h)
        window_x = np.expand_dims(window_x, axis=0)
        # dimension changes
        #window_x = window_x.transpose(1, 0, 2, 3) # shape: (9, 5, w, h)
        windowed_x.append(window_x) 

    
        window_y = y[i:i+window_size] # shape: (3, 24, w, h)
        window_y = window_y.reshape(72, Hy, Wy) # shape: (72, w, h)
        window_y = np.expand_dims(window_y, axis=0)
        # dimension changes
        #window_y = window_y.transpose(1, 0, 2, 3) # shape: (1, 5, w, h)
        windowed_y.append(window_y) 


    windowed_x = np.stack(windowed_x, axis=0)  # (num_windows, 9, 5, Hx, Wx)

    windowed_y = np.stack(windowed_y, axis=0)  # (num_windows, 1, 5, Hy, Wy)
    return windowed_x, windowed_y


def block_average_pooling(x, block_size=30):
    B, T, H, W = x.shape
    assert H % block_size == 0 and W % block_size == 0, "Height and width must be divisible by block size"
    
    x = x.reshape(B, T, H // block_size, block_size, W // block_size, block_size)
    x = x.mean(axis=(3, 5))
    return x