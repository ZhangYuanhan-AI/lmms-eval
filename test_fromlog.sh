accelerate launch --main_process_port=12345 --num_processes=1 \
-m lmms_eval \
--model from_log \
--model_args logs=/opt/tiger/yx/ \
--tasks hardvideo_all \
--batch_size 1 \
--log_samples \
--log_samples_suffix human \
--output_path ./logs/

