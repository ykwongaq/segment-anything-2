import argparse
import os
import shutil
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import yaml
from mmdet.apis import DetInferencer
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*antialias.*")

class TrashDetector:
    def __init__(self, config):
        self.config = config
        self.model = self.prepare_detection_model()
        self.target_classes = self.config["detection"]["target_classes"]
        self.score_threshold = self.config["detection"]["score_threshold"]

    def detect(self, frame_file: str) -> List[Dict]:
        output = self.model(frame_file)
        predictions = output["predictions"][0]
        output_list = []
        for label, score, bbox in zip(
            predictions["labels"], predictions["scores"], predictions["bboxes"]
        ):
            if not label in self.target_classes:
                continue
            if score < self.score_threshold:
                continue

            bbox = [int(b) for b in bbox]
            output_list.append({"label": label, "score": score, "bbox": bbox})

        return output_list

    def prepare_detection_model(self):
        detection_config_file = self.config["detection"]["config_file"]
        detection_model_checkpoint = self.config["detection"]["checkpoint_file"]
        device = self.config["detection"]["device"]
        detection_model = DetInferencer(
            model=detection_config_file,
            weights=detection_model_checkpoint,
            device=device,
            show_progress=False,
        )
        return detection_model


class TrashTracker:
    def __init__(self, config):
        self.config = config
        model_config_file = self.config["segmentation"]["config_file"]
        checkpoint_file = self.config["segmentation"]["checkpoint_file"]

        # self.image_predictor_device = self.config["segmentation"]["image_predictor_device"]
        self.sam1_model = SAM2ImagePredictor(
            build_sam2(model_config_file, checkpoint_file)
        )
        # self.video_predictor_device = self.config["segmentation"]["video_predictor_device"]
        self.sam2_model = build_sam2_video_predictor(model_config_file, checkpoint_file)

    def predict_mask(self, image: np.ndarray, bbox: List[int]) -> torch.Tensor:
        bbox = np.array(bbox)
        self.sam1_model.set_image(image)
        masks, _, _ = self.sam1_model.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )
        mask = masks[0]
        return mask

    def predict_video(
        self, frame_folder: str, idx_to_bbox_outputs: Dict
    ) -> List[torch.Tensor]:
        idx_to_mask_outputs = {}
        frame_files = extract_frame_files(frame_folder)
        for idx, frame_file in enumerate(frame_files):
            image = read_image(frame_file)
            bbox_outputs = idx_to_bbox_outputs[idx]
            masks_output = []
            for bbox_output in bbox_outputs:
                mask = self.predict_mask(image, bbox_output["bbox"])
                masks_output.append((mask, bbox_output["label"]))
            idx_to_mask_outputs[idx] = masks_output

        del self.sam1_model
        torch.cuda.empty_cache()

        # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        with torch.inference_mode():
            inference_state = self.sam2_model.init_state(
                video_path=frame_folder,
            )
            has_prompt = False
            for frame_idx, mask_outputs in idx_to_mask_outputs.items():
                for mask, label in mask_outputs:
                    has_prompt = True
                    _, _, _ = self.sam2_model.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=label,
                        mask=mask,
                    )

            video_segments = {}
            if not has_prompt:
                return video_segments
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.sam2_model.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        del self.sam2_model
        torch.cuda.empty_cache()

        # Ensure that every frame have not Null value
        for frame_idx in range(len(frame_files)):
            if frame_idx not in video_segments:
                video_segments[frame_idx] = {}

        return video_segments


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def is_image(file: str) -> bool:
    try:
        Image.open(file)
        return True
    except:
        return False


def read_config(config_file: str) -> dict:
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


def extract_frame_files(frame_folder: str) -> List[str]:
    frame_files = [
        os.path.join(frame_folder, f)
        for f in os.listdir(frame_folder)
        if is_image(os.path.join(frame_folder, f))
    ]
    frame_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return frame_files


def combine_image_and_mask(
    image: np.ndarray, mask: np.ndarray, obj_id: int, alpha: float = 0.6
) -> np.ndarray:
    cmap = plt.get_cmap("tab10")
    color = np.array(cmap(obj_id)[:3])
    color = color * 255
    color = color.astype(np.int8)

    mask = mask.squeeze()
    binary_mask = mask > 0

    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask] = color
    # binary_mask = np.expand_dims(binary_mask, axis=-1)

    vis_image = np.copy(image)
    vis_image[binary_mask] = (1 - alpha) * vis_image[
        binary_mask
    ] + alpha * colored_mask[binary_mask]
    return vis_image


def frames_to_video(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


def process(config, detector, tracker, frame_folder, output_video_name):

    frame_files = extract_frame_files(frame_folder)
    frame_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    print("Detecting objects in the video ...")
    idx_to_bbox_output = {}
    for idx, frame_file in enumerate(frame_files):
        bboxes = detector.detect(frame_file)
        idx_to_bbox_output[idx] = bboxes
    print(f"There are {len(idx_to_bbox_output)} frames in the folder.")

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Segmenting objects in the video ...")
    video_segments = tracker.predict_video(frame_folder, idx_to_bbox_output)
    if len(video_segments) == 0:
        print("No objects detected in the video.")

    masked_frames = []
    for frame_id, frame_path in enumerate(frame_files):
        image = read_image(frame_path)
        vis_image = np.copy(image)
        if len(video_segments) > 0:
            for obj_id, mask in video_segments[frame_id].items():
                vis_image = combine_image_and_mask(vis_image, mask, obj_id)
        masked_frames.append(vis_image)

    # Save the masked frames as a video
    output_dir = config["output_folder"]
    os.makedirs(output_dir, exist_ok=True)
    fps = config.get("fps", 30)
    video_name = output_video_name + ".mp4"
    video_path = os.path.join(output_dir, video_name)
    print(f"Saving the video to: {video_path}")
    frames_to_video(masked_frames, video_path, fps)


def move_frames(frame_folder: str, output_folder: str) -> None:
    frame_files = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder)]
    frame_files.sort()
    for idx, frame_filename in enumerate(frame_files):
        frame_file = os.path.join(frame_folder, frame_filename)
        output_file = os.path.join(output_folder, f"{idx:06d}.jpg")
        shutil.copy(frame_file, output_file)


def main(args):
    config_file = args.config_file

    print(f"Reading config file: {config_file}")
    config = read_config(config_file)
    input_frame_folder = config["frame_folder"]
    is_frame_folders_contain_sub_folder = config["is_sub_folder"]

    frame_folders = []
    if is_frame_folders_contain_sub_folder:
        for sub_folder in os.listdir(input_frame_folder):
            frame_folder = os.path.join(input_frame_folder, sub_folder)
            frame_folders.append(frame_folder)
    else:
        frame_folders = [input_frame_folder]
    frame_folders.sort()

    detector = TrashDetector(config)

    output_dir = config["output_folder"]
    os.makedirs(output_dir, exist_ok=True)

    for frame_folder in frame_folders:
        tracker = TrashTracker(config)
        print(f"Processing frames in folder: {frame_folder}")
        temp_folder = os.path.join(output_dir, "temp")
        os.makedirs(temp_folder, exist_ok=True)
        move_frames(frame_folder, temp_folder)
        video_name = os.path.basename(frame_folder)
        process(config, detector, tracker, temp_folder, video_name)
        shutil.rmtree(temp_folder)


if __name__ == "__main__":
    DEFAULT_CONFIG_FILE = "config/trash.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help=f"Path to config file. Default: {DEFAULT_CONFIG_FILE}",
    )
    args = parser.parse_args()
    main(args)
