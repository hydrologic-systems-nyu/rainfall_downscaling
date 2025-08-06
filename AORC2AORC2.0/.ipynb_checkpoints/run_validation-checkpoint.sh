#!/bin/bash

#!/bin/bash

models=("ACCESS-CM2" "MIROC6" "ACCESS-ESM1-5" "MIROC-ES2L" "MPI-ESM1-2-HR" \
        "CNRM-ESM2-1" "IPSL-CM6A-LR" "MPI-ESM1-2-LR" "MRI-ESM2-0" \
        "EC-Earth3-Veg-LR" "NESM3")

for model in "${models[@]}"
do
  echo "Running validation for model: $model"
  
  python validation.py \
    --start_year 1981 \
    --end_year 2011 \
    --input_source "GCM_1981-2011/${model}/" \
    --target_source "AORC_1981-2011/AORC_126_186" \
    --model_path model_weights/models_aorc_aorc.pth \
    --step 3 \
    --validation_file "validation/${model}" \
    --mode "GCM"
    
  echo "Finished validation for $model"
  echo "-----------------------------"
done

