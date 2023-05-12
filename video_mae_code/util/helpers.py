import os
import shutil

def clear_directory(output_dir_path="/shared/katop1234/video_inpainting/video_inpainting/output_dir/"):
    if os.path.exists(output_dir_path) and os.path.isdir(output_dir_path):
        shutil.rmtree(output_dir_path)
        print(f"Directory '{output_dir_path}' has been cleared.")

    # recreate the directory after clearing
    os.makedirs(output_dir_path, exist_ok=True)
    print(f"Directory '{output_dir_path}' has been created.")

