import os
import cv2
import glob
from moviepy.editor import ImageSequenceClip

# Directories
base_path = "/shared/katop1234/Datasets/atari/atari_v1/screens/"
output_path = "/shared/katop1234/Datasets/atari_mp4s_120fps/"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Recursive function to yield subdirectories
def get_subdirs(base_path):
    with os.scandir(base_path) as entries:
        for entry in entries:
            if entry.is_dir():
                yield entry.path
                yield from get_subdirs(entry.path)

# Iterate over all subdirectories
for subdir in get_subdirs(base_path):
    print(subdir)
    
    # Get all the .png files in the current directory
    all_images = sorted(glob.glob(os.path.join(subdir, "*.png")))
    
    # Check if there are less than 300 images, if so, continue to next folder
    if len(all_images) < 300:
        continue
    
    # Iterate over all images in chunks of 300
    for i in range(0, len(all_images), 300):
        chunk = all_images[i:i+300]
        
        # If there are less than 300 images in the current chunk, break the loop
        if len(chunk) < 300:
            break
        
        # Convert images to frames
        frames = []
        for img in chunk:
            frame = cv2.imread(img)
            if frame is None:
                print(f"Could not read image {img}")
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        # Create a clip from the frames
        clip = ImageSequenceClip(frames, fps=120)
        
        # Generate output file path
        output_file = os.path.join(output_path, f"{os.path.basename(os.path.dirname(subdir))}_{os.path.basename(subdir)}_{i//300}.mp4")
        
        # Debug info
        print("Clip duration:", clip.duration)
        print("Clip fps:", clip.fps)
        print("Clip size:", clip.size)

        # Write the clip to the output file
        clip.write_videofile(output_file, fps=120)
        print("wrote", output_file)

