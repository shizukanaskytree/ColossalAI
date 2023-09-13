mkdir -p act_scales
mkdir -p logs

git lfs install
git clone https://huggingface.co/datasets/mit-han-lab/pile-val-backup

export MODEL_NAME='facebook/opt-125m'
export ACT_SCALES_PT_FILE='./act_scales/opt-125m.pt'
export DATASET_PATH='pile-val-backup/val.jsonl.zst'

python generate_act_scales.py \
    --model-name $MODEL_NAME \
    --output-path $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH\
    --num-samples 512 \
    --seq-len 512 \
    2>&1 | tee logs/generate_act_scales_$(date +"%Y-%m-%d_%H-%M-%S").log

python export_int8_model.py \
    --model-name $MODEL_NAME \
    --num-samples 1000 \
    --seq-len 512 \
    --act-scales $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH \
    --output-path 'int8_models' \
    2>&1 | tee logs/export_int8_model_$(date +"%Y-%m-%d_%H-%M-%S").log
