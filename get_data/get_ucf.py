import os
import subprocess
import time
import shutil
import rarfile
from moviepy.editor import VideoFileClip
from helpers import delete_files_with_extension, convert_avi_to_mp4

# Check if 'unrar' is installed on the system
if shutil.which("unrar") is None:
    print("'unrar' is not installed on your system. Installing it now...")
    os.system('sudo apt-get update -y')
    os.system('sudo apt-get install unrar -y')
else:
    print("'unrar' is installed on your system.")

# Base directory for your operations
base_dir = os.path.join(os.getcwd(), "get_data/")

# Modify this path to be relative to your get_data directory
download_dir = os.path.join(base_dir, "data/ucf/") 

# # Create download_dir if it doesn't exist
# os.makedirs(download_dir, exist_ok=True)

# # Download the rar file
# url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
# filename = os.path.join(download_dir, "UCF101.rar")

# print("WARNING: This script will download the file without checking the SSL certificate. This operation will start in 10 seconds.")
# time.sleep(10)

# # Download the file using wget with no SSL check
# subprocess.run(["wget", "--no-check-certificate", "-O", filename, url])

# # Open the rar file
# rar = rarfile.RarFile(filename)

# # Extract the rar file
# rar.extractall(path=download_dir)

# # Close the rar file
# rar.close()

base_path = os.path.join(download_dir, 'UCF-101/')
output_path = download_dir

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Recursive traversal
for dirpath, dirs, files in os.walk(base_path):
    for filename in files:
        fname = os.path.join(dirpath, filename)
        if fname.endswith('.avi'):
            # Convert avi to mp4 and move to the main ucf/ dir
            target_fname = os.path.join(output_path, os.path.splitext(filename)[0] + '.mp4')
            convert_avi_to_mp4(fname, target_fname)

# Check each mp4 for frame length and delete if necessary
for filename in os.listdir(output_path):
    if filename.endswith('.mp4'):
        video = VideoFileClip(os.path.join(output_path, filename))
        total_frames = video.duration * video.fps
        if total_frames < 64 / 30 * video.fps:
            os.remove(os.path.join(output_path, filename))

# Clean up non-mp4 files
delete_files_with_extension(output_path, '.avi')
delete_files_with_extension(output_path, '.rar')
# add any other file extensions you want to remove

# Recursively remove empty directories
for dirpath, dirs, files in os.walk(base_path):
    if not dirs and not files:
        os.rmdir(dirpath)
