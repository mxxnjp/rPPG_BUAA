import argparse
import csv
import json
import subprocess
import tempfile
import os

import cv2
import imageio_ffmpeg

from datasets.roi_utils import detect_face_bbox, compute_roi_boxes


def read_first_frame(video_path):
    """
    cv2.VideoCapture()로 직접 비디오 코덱을 여는 대신 ffmpeg CLI로
    첫 프레임만 PNG로 뽑아 cv2.imread()로 읽는다.

    (mediapipe와 같은 프로세스에서 cv2.VideoCapture를 쓰면 네이티브
    라이브러리 충돌로 segfault가 나는 걸 확인해서 우회한 것)
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "frame.png")

        cmd = [
            ffmpeg,
            "-loglevel", "error",
            "-y",
            "-i", video_path,
            "-vframes", "1",
            out_path,
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            return None

        return cv2.imread(out_path)


def read_labels(path):
    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for r in reader:
            rows.append(r)

    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--labels",
        default="buaa_experiment/labels.csv"
    )

    ap.add_argument(
        "--out",
        default="buaa_experiment/roi_boxes.json"
    )

    ap.add_argument(
        "--ref_lux",
        type=float,
        default=100.0,
        help="brightest lux level used as the reference frame for face detection"
    )

    args = ap.parse_args()

    rows = read_labels(args.labels)

    subject_video = {}

    for r in rows:
        if float(r["lux"]) == args.ref_lux:
            subject_video.setdefault(
                r["subject"],
                r["video_path"]
            )

    result = {}

    for subject, video_path in sorted(subject_video.items()):

        frame = read_first_frame(video_path)

        if frame is None:
            print(f"[WARN] cannot read frame for subject {subject}: {video_path}")
            continue

        bbox = detect_face_bbox(frame)

        if bbox is None:
            print(f"[WARN] no face detected for subject {subject}: {video_path}")
            continue

        roi = compute_roi_boxes(bbox)

        result[subject] = {
            "video_path": video_path,
            "face_bbox": list(bbox),
            "roi_boxes": {
                k: list(v)
                for k, v in roi.items()
            },
        }

        print(f"[OK] subject {subject}: face_bbox={bbox}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nsaved {len(result)} / {len(subject_video)} subjects to {args.out}")


if __name__ == "__main__":
    main()
