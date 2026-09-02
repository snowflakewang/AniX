import argparse
import os
import re

import cv2


def merge_videos(input_dir: str, output_path: str | None = None) -> str:
    """Merge videos by trailing index, keeping only the last 96 frames after index 0."""
    output_path = output_path or os.path.join(input_dir, "merged.mp4")

    videos = []
    for name in os.listdir(input_dir):
        match = re.search(r"_(\d+)\.mp4$", name)
        if match:
            videos.append((int(match.group(1)), os.path.join(input_dir, name)))
    videos.sort(key=lambda item: item[0])

    if not videos:
        raise ValueError(f"No videos ending with _INDEX.mp4 found in {input_dir}")

    first = cv2.VideoCapture(str(videos[0][1]))
    fps = first.get(cv2.CAP_PROP_FPS)
    size = (int(first.get(cv2.CAP_PROP_FRAME_WIDTH)), int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    first.release()

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    try:
        for idx, path in videos:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            if idx >= 1:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 96))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != size:
                    raise ValueError(f"Video resolution does not match: {path}")
                writer.write(frame)
            capture.release()
    finally:
        writer.release()

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge videos by index")
    parser.add_argument("--input_dir", help="Directory containing the videos")
    parser.add_argument("--output", help="Output video path")
    args = parser.parse_args()
    print(merge_videos(args.input_dir, args.output))

    # python video_merge.py --input_dir "output/orangeRobot_futureUtopia"