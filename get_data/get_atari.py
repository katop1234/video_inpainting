import os
import cv2
import glob
import wget
import tarfile
import shutil
from moviepy.editor import ImageSequenceClip

download_dir = "data/atari/" # Can specify this path

# Create download_dir if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Download and extract the tar.gz file
url = "https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/full.tar.gz"
filename = wget.download(url, out=download_dir)
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

        clip.write_videofile(output_file, fps=120)
        print("wrote", output_file)

# Clear temp files needed to get mp4
atari_dir = os.path.join(download_dir, 'atari_v1')
shutil.rmtree(atari_dir)
