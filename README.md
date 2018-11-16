

#### Usage (raw images)
python antispoofing/mcnns/scripts/mcnnsantispoofing.py \
    --dataset 7 \
    --augmentation 0 \
    --dataset_path $DATASET_PATH \
    --ground_truth_path $GT_PATH \
    --iris_location $IRIS_LOCATION \
    --output_path $OUTPUT_PATH \
    --n_jobs 6 \
    --classification \
    --operation segment \
    --max_axis 260 \
    --bs 32 \
    --epochs $EPOCHS \
    --lr 0.001 \
    --decay 0.0 \
    --last_layer softmax \
    --loss_function 2 \
    --optimizer 1 \
    --reg 0.1 \
    --device_number $CUDA_VISIBLE_DEVICES

#### Usage (bsif images)
python antispoofing/mcnns/scripts/mcnnsantispoofing.py \
    --dataset 7 \
    --augmentation 0 \
    --dataset_path $DATASET_PATH \
    --ground_truth_path $GT_PATH \
    --iris_location $IRIS_LOCATION \
    --output_path $OUTPUT_PATH \
    --n_jobs 6 \
    --feature_extraction \
    --descriptor bsif \
    --desc_params "[3x3x8]" \
    --classification \
    --operation segment \
    --max_axis 260 \
    --bs 32 \
    --epochs $EPOCHS \
    --lr 0.001 \
    --decay 0.0 \
    --last_layer softmax \
    --loss_function 2 \
    --optimizer 1 \
    --reg 0.1 \
    --device_number $CUDA_VISIBLE_DEVICES

#### Usage (weighted voting)
python antispoofing/mcnns/scripts/mcnnsantispoofing_fusion.py \
    "${SCRIPT_PATH}/weighted_votingconfig.json" \
    --augmentation 0 \
    --weighttype acc \
    --device_number ${CUDA_VISIBLE_DEVICES}
