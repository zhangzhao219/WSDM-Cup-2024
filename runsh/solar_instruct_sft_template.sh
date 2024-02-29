PORT=0
#判断当前端口是否被占用，没被占用返回0，反之1
function Listening {
    TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
    UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
    (( Listeningnum = TCPListeningnum + UDPListeningnum ))
    if [ $Listeningnum == 0 ]; then
        echo "0"
    else
        echo "1"
    fi
}

while [ $PORT == 0 ]; do
    temp1=`shuf -i 10000-50000 -n1`
    if [ `Listening $temp1` == 0 ] ; then
        PORT=$temp1
    fi
done

# -----需要自行配置-----
nproc_per_node=2
CUDA_VISIBLE_DEVICES=6,7 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port ${PORT} \
    llm_sft.py \
    --model_type solar-10-7b-instruct-v1 \
    --model_cache_dir pretrained/upstage/SOLAR-10.7B-Instruct-v1.0 \
    --sft_type lora \
    --tuner_backend swift \
    --template_type llama \
    --output_dir output \
    --add_output_dir_suffix true \
    --ddp_backend nccl \
    --seed 42 \
    --dtype fp16 \
    --dataset_seed 42 \
    --dataset_test_ratio 0.01 \
    --train_dataset_sample -1 \
    --max_length 3072 \
    --truncation_strategy delete \
    --check_dataset_strategy warning \
    --custom_train_dataset_path data/wsdm/model/Pseudo/phase_1/best_eval_1.01/release_train_data.json \
    --quantization_bit 0 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --num_train_epochs 2 \
    --max_steps -1 \
    --weight_decay 0.01 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_total_limit -1 \
    --only_save_model true \
    --logging_steps 100 \
    --push_to_hub false \
    --deepspeed_config_path config/zero2.json \
    --check_model_is_latest false