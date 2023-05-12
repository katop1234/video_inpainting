import os
import shutil

def clear_directory(path):
    os.chdir(path)

    output_dir = 'output_dir'  # replace with your actual directory name
    output_dir_path = os.path.join(path, output_dir)

    if os.path.exists(output_dir_path) and os.path.isdir(output_dir_path):
        shutil.rmtree(output_dir_path)
        
def clear_output_dir():
    clear_directory("/shared/katop1234/video_inpainting/video_inpainting/output_dir/")
