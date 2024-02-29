CUDA_VISIBLE_DEVICES=0 \
python llm_infer.py \
    --model_type solar-10-7b-instruct-v1 \
    --model_cache_dir pretrained/upstage/SOLAR-10.7B-Instruct-v1.0 \
    --ckpt_dir checkpoints/v08-20240205-114459 \
    --custom_val_dataset_path release_phase2_test_data_wo_gt.json \
    --val_dataset_sample 3588 \
    --stream false \
    --verbose false \
    --sft_type lora \
    --infer_backend vllm \
    --template_type llama \
    --seed 42 \
    --max_length 3072 \
    --max_new_tokens 512 \
    --repetition_penalty 1.02 \
    --do_sample false \
    --merge_lora_and_save true

cp checkpoints/v08-20240205-114459-merged/infer_result_*.jsonl merge/v08-20240205-114459.jsonl
