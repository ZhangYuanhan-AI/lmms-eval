# accelerate launch --main_process_port=12345 --num_processes=1 \
# -m lmms_eval \
# --model openai_client \
# --model_args model=gpt-4o-2024-05-13,max_frames=120,fps=5 \
# --tasks hardvideo_single_mc \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix gpt4o \
# --output_path ./logs/



# accelerate launch --main_process_port=12345 --num_processes=1 \
# -m lmms_eval \
# --model openai_client \
# --model_args model=gemini-1.5-pro-preview,max_frames=64,fps=5 \
# --tasks hardvideo_single_mc \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix gpt4o \
# --output_path ./logs/


accelerate launch --main_process_port=12345 --num_processes=1 \
-m lmms_eval \
--model openai_client \
--model_args model=llava_next_video_7b,max_frames=64,fps=1 \
--tasks hardvideo_single_mc \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_next_video_7b_api \
--output_path ./logs/