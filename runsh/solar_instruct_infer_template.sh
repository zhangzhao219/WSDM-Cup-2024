CUDA_VISIBLE_DEVICES=$1 \
python llm_infer.py \
    --model_type solar-10-7b-instruct-v1 \
    --model_cache_dir pretrained/upstage/SOLAR-10.7B-Instruct-v1.0 \
    --ckpt_dir $2 \
    --custom_val_dataset_path data/wsdm/model/ori/release_phase2_test_data_wo_gt.json \
    --val_dataset_sample 3588 \
    --stream false \
    --verbose false \
    --sft_type lora \
    --infer_backend vllm \
    --template_type llama \
    --seed 42 \
    --max_length 3072 \
    --max_new_tokens 512 \
    --repetition_penalty 1.00 \
    --do_sample false \
    --merge_lora_and_save true

