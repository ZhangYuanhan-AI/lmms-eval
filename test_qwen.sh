accelerate launch --main_process_port=12346 --num_processes=8 \
-m lmms_eval \
--model qwen2_5_vl \
--model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,use_custom_video_loader=True,max_image_size=1024,max_num_frames=64,use_flash_attention_2=True \
--tasks hardvideo_single_mc \
--batch_size 1 \
--log_samples \
--log_samples_suffix qwen2_5_vl_72B \
--output_path ./logs/