# accelerate launch --main_process_port=12345 --num_processes=1 \
# -m lmms_eval \
# --model openai_client \
# --model_args model=gpt-4o-2024-05-13,max_frames=64,fps=1,process_num=10 \
# --tasks hardvideo_single_mzc \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix gpt4o \
# --output_path ./logs/





accelerate launch --main_process_port=12345 --num_processes=1 \
-m lmms_eval \
--model openai_client \
--model_args model=gemini-1.5-pro-preview,max_frames=64,fps=1,process_num=1 \
--tasks hardvideo_all \
--batch_size 1 \
--log_samples \
--log_samples_suffix gemini-1.5-pro \
--output_path ./logs/

# accelerate launch --main_process_port=12345 --num_processes=1 \
# -m lmms_eval \
# --model openai_client \
# --model_args model=Qwen2-VL-72B-Instruct,max_frames=64,fps=1,process_num=10,is_even_frame=True \
# --tasks hardvideo_all \
# --batch_size 1 \
# --log_samples \
# --log_samples_suffix qwen2_vl_72b_api \
# --output_path ./logs/