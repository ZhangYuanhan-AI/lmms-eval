import whisper
import os
import subprocess

# Paths (update as needed)
video_dir = '/mnt/bn/tiktok-mm-4/aiic/public/model/huggingface/hardvideo/Benchmark-AllVideos-LQ/'  # Directory containing the videos
audio_dir = './audio_files_human'  # Directory to save extracted audio
transcript_dir = './audio_human'  # Directory to save transcripts
model = whisper.load_model("base")
# Create output directories if needed
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(transcript_dir, exist_ok=True)

# Function to extract audio using ffmpeg (from video to mp3)
def extract_audio(video_path, audio_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn', '-acodec', 'mp3', '-y',
        '-ar', '44100', '-ab', '192k',
        audio_path
    ]
    subprocess.run(command, check=True)

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

# Process all videos in directory
def process_all_videos(video_dir, audio_dir, transcript_dir):
    video_extensions = {'.mp4', '.mkv', '.webm', '.mov', '.avi'}

    for file_name in os.listdir(video_dir):
        if not any(file_name.lower().endswith(ext) for ext in video_extensions):
            continue  # Skip non-video files

        video_path = os.path.join(video_dir, file_name)
        base_name, _ = os.path.splitext(file_name)
        audio_path = os.path.join(audio_dir, f"{base_name}.mp3")
        transcript_path = os.path.join(transcript_dir, f"{base_name}.txt")

        try:
            # Extract audio from video
            extract_audio(video_path, audio_path)
            
            # Transcribe audio
            transcript = transcribe_audio(audio_path)

            # Save transcript to file
            with open(transcript_path, 'w') as f:
                f.write(transcript)

            print(f"Processed {file_name} -> Transcript saved to {transcript_path}")

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

# Run the pipeline
process_all_videos(video_dir, audio_dir, transcript_dir)
