import os
import subprocess

'''
Move all videos from directory A to directory B, converting to mp4 in the process.
'''

input_dir = '/shared/katop1234/Datasets/DirA/'
output_dir = '/shared/katop1234/Datasets/DirB/'
video_extensions = ['.MOV', '.mp4', ".webm"]
count = 0

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if os.path.splitext(file)[1] in video_extensions:
            # Construct full file paths
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_dir, f'{str(count).zfill(5)}.mp4')

            # Run the ffmpeg command to convert to mp4
            # Including a scale filter to ensure dimensions are divisible by 2
            subprocess.call(['ffmpeg', '-i', input_file, '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', output_file])

            count += 1
