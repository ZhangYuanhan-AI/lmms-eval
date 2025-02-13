accelerate launch --main_process_port=12345 --num_processes=8 \
-m lmms_eval \
--model llava_vid \
--model_args pretrained=/mnt/bn/tiktok-mm-4/aiic/users/wujinming/checkpoints/LLaVA-NeXT-Video-7B-Qwen2/,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
--tasks hardvideo \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_vid \
--output_path ./logs/