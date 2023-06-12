import os
import cv2
import glob
import wget
import tarfile
import shutil
from moviepy.editor import ImageSequenceClip

### WARNING call python get_data/get_atari.py or cwd will be wrong

# Base directory for your operations
base_dir = os.path.join(os.getcwd(), "get_data/")

# Modify this path to be relative to your get_data directory
download_dir = os.path.join(base_dir, "data/atari/") 

# Create download_dir if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Download and extract the tar.gz file
url = "https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/full.tar.gz"
filename = wget.download(url, out=os.path.join(download_dir, "full.tar.gz"))

tar = tarfile.open(filename, "r:gz")
tar.extractall(path=download_dir)
tar.close()

base_path = os.path.join(download_dir, 'atari_v1/screens/')
output_path = download_dir

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

def get_subdirs(base_path):
    with os.scandir(base_path) as entries:
        for entry in entries:
            if entry.is_dir():
                yield entry.path
                yield from get_subdirs(entry.path)

for subdir in get_subdirs(base_path):
    print(subdir)
    all_images = sorted(glob.glob(os.path.join(subdir, "*.png")))

    if len(all_images) < 300:
        continue

    for i in range(0, len(all_images), 300):
        chunk = all_images[i:i+300]
        if len(chunk) < 300:
            break

        frames = []
        for img in chunk:
            frame = cv2.imread(img)
            if frame is None:
                print(f"Could not read image {img}")
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        clip = ImageSequenceClip(frames, fps=120)

        output_file = os.path.join(output_path, f"{os.path.basename(os.path.dirname(subdir))}_{os.path.basename(subdir)}_{i//300}.mp4")

        # Create an absolute path to the output file
        abs_output_file = os.path.abspath(output_file)
        clip.write_videofile(abs_output_file, fps=120)
        print("wrote", abs_output_file)

# Clear temp files needed to get mp4
atari_dir = os.path.join(download_dir, 'atari_v1')
shutil.rmtree(atari_dir)

# Delete the tar.gz file named full.tar.gz
os.remove(os.path.join(download_dir, "full.tar.gz"))
