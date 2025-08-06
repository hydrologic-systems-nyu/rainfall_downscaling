import os
import glob
import xarray as xr
from datetime import datetime

def extract_datetime_from_filename(filename):
    try:
        parts = os.path.basename(filename).split('_')
        date_str = parts[3]      # e.g., '20141101'
        hour_str = parts[4][:4]  # e.g., '0000'
        return datetime.strptime(date_str + hour_str, "%Y%m%d%H%M")
    except Exception as e:
        print(f"Skip file {filename}: {e}")
        return None

def load_file_with_time(file_path):
    ds = xr.open_dataset(file_path)
    dt = extract_datetime_from_filename(file_path)
    if dt is None:
        return None
    da = ds["APCP_surface"]
    # Downsample from 630x930 to 21x31 using mean pooling
    da = da.coarsen(latitude=30, longitude=30, boundary="trim").mean()
    return da.expand_dims(time=[dt])

def merge_monthly_by_year(input_dir, output_dir, year):
    os.makedirs(output_dir, exist_ok=True)
    all_files = sorted(glob.glob(os.path.join(input_dir, f"APCP_surface_{year}_*.nc")))
    if not all_files:
        print(f"❌ Not find the {year} data")
        return
    
    monthly_data = {month: [] for month in range(1, 13)}
    
    for f in all_files:
        dt = extract_datetime_from_filename(f)
        if dt is None or dt.year != year:
            continue
        da = load_file_with_time(f)
        if da is not None:
            monthly_data[dt.month].append(da)

    for month in range(1, 13):
        data_list = monthly_data[month]
        if not data_list:
            print(f"⚠️ {year}-{month:02d} No Data")
            continue
        print(f"📦 Processing {year}-{month:02d}：{len(data_list)} hours")
        
        # combined all data
        combined = xr.concat(data_list, dim="time")
        combined = combined.sortby("time")

        final_ds = xr.Dataset({"APCP_surface": combined})

        out_file = os.path.join(output_dir, f"APCP_surface_{year}-{month:02d}.nc")
        final_ds.to_netcdf(out_file)
        print(f"✅ save: {out_file}")


for y in range(1981, 2012, 1):
    merge_monthly_by_year(
        input_dir="./training/AORC/Hourly/",
        output_dir="./AORC_21_31/",
        year=y
    )