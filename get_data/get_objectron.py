from helpers import convert_and_move_videos
import os
import subprocess
import shutil

# Check if 'gsutil' exists.
if shutil.which("gsutil") is None:
    print("'gsutil' is not installed on your system. Installing it now...")
    
    # Provide the absolute path to your bash script
    bash_script = 'get_data/download_gcloud.sh'

    # Change the permission of the bash script to make it executable
    os.chmod(bash_script, 0o755)

    # Execute the bash script
    subprocess.run([bash_script], check=True)
    
    print("One last step, in your terminal, type in")
    print("gcloud init")
    print("and follow the instructions there.")

else:
    print("'gsutil' is installed on your system.")
    
# Test a basic gcloud command to check if the user has correctly set up their account
account = subprocess.run(['gcloud', 'config', 'get-value', 'account'], capture_output=True, text=True).stdout.strip()
if account:
    print(f"gcloud is correctly set up for account {account}")
else:
    print("gcloud is not set up correctly. Please run 'gcloud init' and follow the instructions. Then rerun this script with")
    print("python get_data/get_objectron.py")
    exit(0)

# Execute the gsutil command to download the objectron dataset
os.system('mkdir -p get_data/data/objectron_temp/')
subprocess.run(['gsutil', '-m', 'cp', '-r', 'gs://objectron', 'get_data/data/objectron_temp/'], check=True)

# TODO implement this based on the above results
# convert_and_move_videos(input_dir='data/objectron_temp/objectron', 
#                     output_dir='/shared/katop1234/Datasets/objectron/')