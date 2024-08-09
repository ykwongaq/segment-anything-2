import os
import shutil

data_dir = "/mnt/hdd/davidwong/data/trash/TrashCan"
output_dir = "/mnt/hdd/davidwong/data/trash_video"

frame_files = []
for split in ["train", "val"]:
    split_dir = os.path.join(data_dir, split)
    for video_name in os.listdir(split_dir):
        video_dir = os.path.join(split_dir, video_name)
        frame_files.append(video_dir)

video_dataset = {}
for frame_file in frame_files:
    frame_filename = os.path.basename(frame_file)
    video_name = os.path.splitext(frame_filename)[0]
    video_name = video_name.split("_")[1]
    if video_name not in video_dataset:
        video_dataset[video_name] = []
    video_dataset[video_name].append(frame_file)

for video_name, frame_files in video_dataset.items():
    output_folder = os.path.join(output_dir, video_name)
    os.makedirs(output_folder, exist_ok=True)
    for frame_file in frame_files:
        shutil.copy(frame_file, output_folder)
