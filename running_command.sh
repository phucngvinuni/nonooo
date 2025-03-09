TF_ENABLE_ONEDNN_OPTS=0 CUDA_VISIBLE_DEVICES=2 python3 run_class_main.py \
    --model ViT_FIM_model_S \
    --output_dir ckpt_record \
    --data_set cifar_S32 \
    --batch_size 50 \
    --input_size 224 \
    --lr 3e-5 \
    --epochs 50 \
    --opt_betas 0.95 0.99 \
    --save_freq 2 \
    --mask_ratio 0.0 \
    --train_type fim_train \
    --if_attack_train \
    --if_attack_test \
