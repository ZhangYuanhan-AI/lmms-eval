accelerate launch --main_process_port=12345 --num_processes=1 \
-m lmms_eval \
--model internvl2 \
--model_args pretrained=OpenGVLab/InternVL2_5-38B,modality="video",num_frame=64,device_map=1 \
--tasks hardvideo_single_mc \
--batch_size 1 \
--log_samples \
--log_samples_suffix internvl2 \
--output_path ./logs/