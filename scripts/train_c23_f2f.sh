EXPID=$(date +"%Y%m%d_%H%M%S")

HOST='127.0.0.1'
PORT='5'

NUM_GPU=1

YOUR_DATA_PATH="./data"
YOUR_RESULT_PATH="./results"


CUDA_VISIBLE_DEVICES="1" python train.py \
    --results_path ${YOUR_RESULT_PATH} \
    --config 'configs/bottleneck_vit_base_patch16_224_spatial.json' \
    --data_dir "${YOUR_DATA_PATH}/FaceForensicspp_RECCE" \
    --dataset_name 'FaceForensicspp_RECCE_c23' \
    --dataset_split 'youtube_Face2Face' \
    --test_dataset_name 'youtube_Face2Face' \
    --launcher pytorch \
    --rank 0 \
    --log_num ${EXPID} \
    --dist-url tcp://${HOST}:2338${PORT} \
    --world_size $NUM_GPU \
    --ffn_adapt \
    --val_epoch 1 \
    --manual_seed 42