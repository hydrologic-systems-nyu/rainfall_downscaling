#!/bin/bash



echo "Running validation for model: train"
  
python validation.py \
    --start_year 2011 \
    --end_year 2011 \
    --cotinuous_year 5\
    --input_source "AORC_1981-2011/AORC_21_31" \
    --target_source "AORC_1981-2011/AORC_126_186" \
    --model_path model_weights/models_aorc_aorc_ecdfloss.pth \
    --step 3 \
    --validation_file "validation/train_ecdfloss_6" \
    --mode "AORC"
    
echo "Finished validation for train"
echo "-----------------------------"
