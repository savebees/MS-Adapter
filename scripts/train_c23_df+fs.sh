EXPID=$(date +"%Y%m%d_%H%M%S")

HOST='127.0.0.1'
PORT='5'

NUM_GPU=2

YOUR_DATA_PATH="./data"
YOUR_RESULT_PATH="./results"

CUDA_VISIBLE_DEVICES="0,1" python train.py \
    --results_path ${YOUR_RESULT_PATH} \
    --config 'configs/bottleneck_vit_base_patch16_224_spatial.json' \
    --data_dir "${YOUR_DATA_PATH}/FaceForensicspp_RECCE" \
    --dataset_name 'FaceForensicspp_RECCE_c23' \
    --dataset_split 'df+fs' \
    --test_dataset_name 'youtube_Deepfakes youtube_Face2Face youtube_FaceSwap youtube_NeuralTextures' \
    --launcher pytorch \
    --rank 0 \
    --log_num ${EXPID} \
    --dist-url tcp://${HOST}:3234${PORT} \
    --world_size $NUM_GPU \
    --ffn_adapt \
    --val_epoch 2
    --resume 'results/FaceForensicspp_RECCE_c23/alltype/vit_base_patch16_224_spatial/bottleneck/log20250109_051256/snapshots/model-75.pt'