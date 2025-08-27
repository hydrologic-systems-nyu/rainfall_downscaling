#!/bin/bash


models=("ACCESS-CM2" "MIROC6")
        
for model in "${models[@]}"
do
  echo "Running validation for model: $model"
  
  python -u validation.py \
    --start_year 1981 \
    --end_year 2011 \
    --cotinuous_year 5\
    --input_source "GCM_1981-2011/${model}" \
    --target_source "AORC_1981-2011/AORC_126_186" \
    --model_path model_weights/models_aorc_aorc_ecdfloss.pth \
    --step 3 \
    --validation_file "validation/${model}_ecdfloss_5" \
    --mode "GCM"
    
  echo "Finished validation for $model"
  echo "-----------------------------"s
done

