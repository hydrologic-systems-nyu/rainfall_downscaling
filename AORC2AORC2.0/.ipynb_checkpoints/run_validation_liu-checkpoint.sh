#!/bin/bash

#!/bin/bash

models=("CMCC-CM2-SR5" "FGOALS-g3" "NorESM2-LM" "BCC-CSM2-MR" "GFDL-CM4_gr2" \
        "INM-CM5-0" "TaiESM1" "CanESM5" "GFDL-ESM4"  "CESM2" "KIOST-ESM")

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
