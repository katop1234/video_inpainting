import os
import subprocess
import time
import rarfile

# Base directory for your operations
base_dir = os.path.join(os.getcwd(), "get_data/")

# Modify this path to be relative to your get_data directory
download_dir = os.path.join(base_dir, "data/ucf/") 

# Create download_dir if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Download the rar file
url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
filename = os.path.join(download_dir, "UCF101.rar")

print("WARNING: This script will download the file without checking the SSL certificate. This operation will start in 10 seconds.")
time.sleep(10)

# Download the file using wget with no SSL check
subprocess.run(["wget", "--no-check-certificate", "-O", filename, url])

# Open the rar file
rar = rarfile.RarFile(filename)

# Extract the rar file
rar.extractall(path=download_dir)

# Close the rar file
rar.close()

base_path = os.path.join(download_dir, 'UCF101/')
output_path = download_dir

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)
