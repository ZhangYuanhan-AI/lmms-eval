accelerate launch --main_process_port=12345 --num_processes=8 \
-m lmms_eval \
--model internvl2 \
--model_args pretrained=rhymes-ai/Aria,max_frames_num=64  \
--tasks hardvideo_single_mc \
--batch_size 1 \
--log_samples \
--log_samples_suffix internvl2 \
--output_path ./logs/