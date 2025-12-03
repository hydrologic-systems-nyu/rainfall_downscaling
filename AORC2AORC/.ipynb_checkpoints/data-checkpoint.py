import xarray as xr
import rioxarray
import numpy as np
import glob


def readFile(data_file, value, t, h, w):
    all_data = []
    nc_files = sorted(glob.glob(data_file))
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        ds = ds.rio.write_crs("EPSG:4326")
        data = ds[value].values  # shape: (time, H, W)
        time = data.shape[0]
        if time % t != 0:
            continue
    
        reshaped = data.reshape(-1, t, h, w)
        all_data.append(reshaped) 
    final_data = np.concatenate(all_data, axis=0)
    print("final shape", final_data.shape)
    return final_data

def make_temporal_batches(x, y, temporal_factor):
    window_size = 5
    stride = 3
    Tx, Cx, Hx, Wx = x.shape
    Ty, Cy, Hy, Wy = y.shape
    num_windows = (Tx - window_size) // stride + 1
    windowed_x = []
    windowed_y = []

    for i in range(0, Tx - window_size + 1, stride):
        window_x = x[i:i+window_size] # shape: (day_x, time_x, w, h)
        window_y = y[i:i+window_size] # shape: (day_y, time_y, w, h)
        window_x = window_x.reshape(window_size * Cx, Hx, Wx) # shape: (Tx * Cx, w, h)
        window_x = np.expand_dims(window_x, axis=0)
        window_y = window_y.reshape(window_size * Cy, Hy, Wy) # shape: (Ty * Cy, w, h)
        window_y = np.expand_dims(window_y, axis=0)
        windowed_x.append(window_x) 
        windowed_y.append(window_y) 

        

    windowed_x = np.stack(windowed_x, axis=0)  # (num_windows, 1, time_x, Hx, Wx)

    windowed_y = np.stack(windowed_y, axis=0)[:, :, temporal_factor:-temporal_factor]  # (num_windows, 1, time_y, Hy, Wy)
    return windowed_x, windowed_y
