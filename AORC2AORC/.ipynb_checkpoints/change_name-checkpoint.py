import os
from datetime import datetime, timedelta


def is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


models = ["CMCC-CM2-SR5", "FGOALS-g3", "NorESM2-LM", "BCC-CSM2-MR", "GFDL-CM4_gr2",
        "INM-CM5-0", "TaiESM1", "CanESM5", "GFDL-ESM4",  "CESM2", "KIOST-ESM"]

for model in models:
    folder_path = f"/scratch/jl14811/validation/{model}"
    files = sorted(os.listdir(folder_path))
    for fname in files:
        name_no_ext = fname.replace(".nc", "")
        class_str, start_str, end_str = name_no_ext.split("_")
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        if is_leap(start_date.year) and start_date.month == 2 and start_date.day == 28:
            end_date += timedelta(days=1)
            new_name = f"{class_str}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.nc"
            os.rename(
                os.path.join(folder_path, fname),
                os.path.join(folder_path, new_name)
            )
        elif is_leap(start_date.year) and start_date.month > 2:
            start_date += timedelta(days=1)
            end_date += timedelta(days=1)
            new_name = f"{class_str}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.nc"
            os.rename(
                os.path.join(folder_path, fname),
                os.path.join(folder_path, new_name)
            )

    
