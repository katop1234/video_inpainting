import os
import subprocess
from moviepy.editor import VideoFileClip

def convert_and_move_videos(input_dir, output_dir):
    '''
    Move all videos from directory A to directory B, converting to mp4 in the process.
    '''
    video_extensions = ['.MOV', '.mp4', ".webm"]
    count = 0
    print(os.getcwd(), input_dir, output_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            print(file)
            if os.path.splitext(file)[1] in video_extensions:
                # Construct full file paths
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, f'{str(count).zfill(5)}.mp4')

                # Use moviepy to convert to mp4
                clip = VideoFileClip(input_file)
                clip.write_videofile(output_file)

                count += 1


def delete_files_with_extension(directory, extension):
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            os.remove(os.path.join(directory, filename))

def convert_avi_to_mp4(avi_file_path, mp4_file_path):
    try:
        clip = VideoFileClip(avi_file_path)
        clip.write_videofile(mp4_file_path, codec='libx264')

    except Exception as e:
        print(f"Unable to convert {avi_file_path} to {mp4_file_path}. Exception: {e}")

