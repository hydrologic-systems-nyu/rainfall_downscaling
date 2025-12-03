#!/bin/bash


models=("ACCESS-CM2" "ACCESS-ESM1-5" "BCC-CSM2-MR" "CanESM5" "CESM2" "CMCC-CM2-SR5"
    "CNRM-ESM2-1" "EC-Earth3-Veg-LR" "FGOALS-g3" "GFDL-CM4_gr2" "GFDL-ESM4"
    "INM-CM5-0" "IPSL-CM6A-LR" "KIOST-ESM" "MIROC-ES2L" "MIROC6" "MPI-ESM1-2-HR"
    "MPI-ESM1-2-LR" "MRI-ESM2-0" "NESM3" "NorESM2-LM" "TaiESM1")

for model in "${models[@]}"
do
  echo "Running validation for model: $model"
  
  python -u validation.py \
    --start_year 1981 \
    --end_year 2011 \
    --cotinuous_year 5\
    --input_source "/scratch/jl14811/GCM_1981-2011/${model}" \
    --target_source "/scratch/jl14811/AORC_1981-2011/AORC_126_186" \
    --model_path model_weights/models_aorc_aorc_1-3.pth \
    --step 3 \
    --validation_file "validation/1_3_3-train/${model}" \
    --mode "GCM" \
    --temporal_factor 1 \
    --spatial_factor 3
    
  echo "Finished validation for $model"
  echo "-----------------------------"s
done
