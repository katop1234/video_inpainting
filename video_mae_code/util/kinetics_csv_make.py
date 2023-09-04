import argparse
import csv
import os

PARSER = argparse.ArgumentParser()
#
PARSER.add_argument('--kinetics_path', type=str,
                    help='path to kinetics')
PARSER.add_argument('--kinetics_videos_fb_csv', type=str,
                    help='csv for kinetics videos for fb')


def create_csv(kinetics_path, kinetics_videos_fb_csv):
    with open(kinetics_videos_fb_csv, 'w') as f:
        writer = csv.writer(f)
        for _, dirnames, _ in os.walk(kinetics_path):
            for dirname in dirnames:
                curr_case = os.path.join(kinetics_path, dirname)
                for filename in os.listdir(curr_case):
                    if filename.endswith('.mp4'):
                        curr_file = str(os.path.join(curr_case, filename)) + ' 0'
                        curr_row = [curr_file]
                        writer.writerow(curr_row)
                    
def main():
    args = PARSER.parse_known_args()[0]
    create_csv(args.kinetics_path, args.kinetics_videos_fb_csv)
    
if __name__ == "__main__":
    main()