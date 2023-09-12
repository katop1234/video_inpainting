import csv
import os
import glob

# File extensions
video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.webm']
image_extensions = ['.png', '.jpg', '.JPEG']

#Facebook Paths
imagenet_fb_path = "/datasets01/imagenet_full_size/061417/train/"
cvf_fb_path = "/private/home/amirbar/datasets/CVF/arxiv_resized_train_val_split/train/"

crosstask_fb_path = "/datasets01/CrossTask/053122/raw_video/"
kinetics_fb_path = "/datasets01/kinetics/092121/400/train_288px/"
egoobj_fb_path = "/datasets01/ego-object/012422/raw/"
ssv2_fb_path = "/datasets01/SSV2/videos/"
ucf101_fb_path = "/datasets01/ucf101/112018/data"
csv_fb_path = "/private/home/amirbar/datasets/CSV/"


imagenet_fb_csv = "imagenet_images_fb.csv"
cvf_fb_csv = "cvf_images_fb.csv"

crosstask_fb_csv = "crosstask_videos_fb.csv"
kinetics_fb_csv = "kinetics_videos_fb.csv"
egoobj_fb_csv = "egoobj_videos_fb.csv"
ssv2_fb_csv = "ssv2_videos_fb.csv"
ucf101_fb_csv = "ucf101_videos_fb.csv"
csv_fb_csv = "csv_videos_fb.csv"

fb_make = [[crosstask_fb_path, crosstask_fb_csv, video_extensions], [kinetics_fb_path, kinetics_fb_csv, video_extensions], [egoobj_fb_path, egoobj_fb_csv, video_extensions], 
                 [ssv2_fb_path, ssv2_fb_csv, video_extensions], [ucf101_fb_path, ucf101_fb_csv, video_extensions], [csv_fb_path, csv_fb_csv, video_extensions],
                 [imagenet_fb_path, imagenet_fb_csv, image_extensions], [cvf_fb_path, cvf_fb_csv, image_extensions]]

#BAIR Paths
imagenet_path = "/shared/group/ilsvrc/train"
cvf_path = "/shared/amir/dataset/arxiv_resized_train_val_split/train"

crosstask_path = "/shared/katop1234/Datasets/CrossTask_vids/"
kinetics_path = "/shared/group/kinetics/train_256/"
objectron_path = "/shared/katop1234/Datasets/Objectron/"
ssv2_path = "/shared/katop1234/Datasets/SSV2_videos/"
ucf101_path = "/shared/katop1234/Datasets/UCF101/"
csv_path = "/shared/dannyt123/Datasets/CSV"


imagenet_csv = "imagenet_images.csv"
cvf_csv = "cvf_images.csv"

crosstask_csv = "crosstask_videos.csv"
kinetics_csv = "kinetics_videos.csv"
objectron_csv = "objectron_videos.csv"
ssv2_csv = "ssv2_videos.csv"
ucf101_csv = "ucf101_videos.csv"
csv_csv = "csv_videos.csv"

bair_make = [[crosstask_path, crosstask_csv, video_extensions], [kinetics_path, kinetics_csv, video_extensions], [objectron_path, objectron_csv, video_extensions], 
                 [ssv2_path, ssv2_csv, video_extensions], [ucf101_path, ucf101_csv, video_extensions], [csv_path, csv_csv, video_extensions],
                 [imagenet_path, imagenet_csv, image_extensions], [cvf_path, cvf_csv, image_extensions]]

def create_csv(dataset_path, dataset_csv_path, extensions):
    print('dataset_path: ', dataset_path)
    with open(dataset_csv_path, 'w') as f:
        writer = csv.writer(f)
        # Use glob to find video files with specified extensions recursively
        for curr_extension in extensions:
            videos = glob.glob(os.path.join(dataset_path, '**', f'*{curr_extension}'), recursive=True)
            for video in videos:
                if video[0] == '"':
                    video = video[1:]
                if video[-1] == '"':
                    video = video[:-1]
                writer.writerow([video])
                    
def main():
    for dataset_path, dataset_csv_path, extensions in fb_make:
        create_csv(dataset_path, dataset_csv_path, extensions)
        
    
if __name__ == "__main__":
    main()