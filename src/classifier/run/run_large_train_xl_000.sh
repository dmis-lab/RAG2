DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODELNAME=flant5
LLM=llama3
MODEL=/classifier/model/updated_flan_t5_model
DATASET_NAME=llama3_5%-train
BENCHMARK=MEDQA
GPU=1

for EPOCH in 40
do
    # train
    TRAIN_OUTPUT_DIR=classifier/model/outputs/LLM/${LLM}/BENCHMARK/${BENCHMARK}/DATASET_NAME/${DATASET_NAME}/epoch/${EPOCH}/${DATE}
    mkdir -p ${TRAIN_OUTPUT_DIR}
    
    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${MODEL} \
        --train_file classifier/model/data/medqa/llama3_cot/5%-train.json \
        --question_column question \
        --answer_column answer \
        --checkpointing_steps epoch \
        --learning_rate 3e-5 \
        --max_seq_length 512 \
        --doc_stride 128 \
        --per_device_train_batch_size 16 \
        --output_dir ${TRAIN_OUTPUT_DIR} \
        --overwrite_cache \
        --train_column 'train' \
        --do_train \
        --num_train_epochs ${EPOCH}
done