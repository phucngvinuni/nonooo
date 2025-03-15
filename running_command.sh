# running_command.sh
TF_ENABLE_ONEDNN_OPTS=0 CUDA_LAUNCH_BLOCKING=1 python3 run_class_main.py \
    --model ViT_FIM_model_S \
    --output_dir ckpt_record \
    --data_set fish \
    --data_path fish_image/ \
    --batch_size 2 \
    --input_size 224 \
    --lr 3e-5 \
    --epochs 1 \
    --opt_betas 0.95 0.99 \
    --save_freq 2 \
    --mask_ratio 0.0 \
    --train_type fim_train \
    --if_attack_train \
    --if_attack_test