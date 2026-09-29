import os
import csv
import glob
import json
import subprocess
import tempfile

import cv2
import torch
import numpy as np
import imageio_ffmpeg

from torch.utils.data import Dataset

from datasets.roi_utils import build_mask


class BUAARppgDataset(Dataset):
    """
    BUAA-MIHR rPPG Dataset

    반환:
        {
            "frames":  (T, 3, 112, 112)
            "bvp":     (T,)
            "hr":      (T,)
            "quality": {
                "blur_score": (T,)
                "brightness": (T,)
                "dark_ratio": (T,)
                "lux":        (T,)
            }
            "subject": str
            "lux": float
        }
    """

    def __init__(
        self,
        labels_csv,
        window_size=30,
        stride=15,
        image_size=112,
        cache_dir="/tmp/buaa_dataset_cache",
        roi_mode="full",
        roi_boxes_path=None,
    ):
        self.labels_csv = labels_csv
        self.window_size = window_size
        self.stride = stride
        self.image_size = image_size
        self.cache_dir = cache_dir
        self.roi_mode = roi_mode

        os.makedirs(self.cache_dir, exist_ok=True)

        self.roi_boxes = {}

        if self.roi_mode == "multi":

            if roi_boxes_path is None:
                raise ValueError(
                    "roi_boxes_path is required when roi_mode='multi'"
                )

            with open(roi_boxes_path, "r") as f:
                self.roi_boxes = json.load(f)

        self.rows = self._read_labels(labels_csv)
        self.windows = self._build_windows()

        print(
            f"[BUAA Dataset] rows={len(self.rows)}, "
            f"windows={len(self.windows)}, "
            f"window={self.window_size}, stride={self.stride}"
        )

    def _read_labels(self, path):
        rows = []

        with open(path, "r") as f:
            reader = csv.DictReader(f)

            for r in reader:
                rows.append({
                    "subject": r["subject"],
                    "lux": float(r["lux"]),
                    "video_path": r["video_path"],
                    "frame_idx": int(r["frame_idx"]),
                    "timestamp": float(r["timestamp"]),
                    "bvp": float(r["bvp"]),
                    "hr_bpm": float(r["hr_bpm"]),
                })

        return rows

    def _build_windows(self):
        """
        같은 video 안에서만 window를 구성.
        """
        groups = {}

        for i, r in enumerate(self.rows):
            key = r["video_path"]
            groups.setdefault(key, []).append(i)

        windows = []

        for video_path, indices in groups.items():
            indices = sorted(
                indices,
                key=lambda idx: self.rows[idx]["frame_idx"]
            )

            n = len(indices)

            for start in range(
                0,
                n - self.window_size + 1,
                self.stride
            ):
                win = indices[start:start + self.window_size]

                # 연속 frame인지 확인
                frame_ids = [
                    self.rows[idx]["frame_idx"]
                    for idx in win
                ]

                expected = list(
                    range(
                        frame_ids[0],
                        frame_ids[0] + self.window_size
                    )
                )

                if frame_ids != expected:
                    continue

                windows.append(win)

        return windows

    def __len__(self):
        return len(self.windows)
    def _extract_one_frame_ffmpeg(self, video_path, frame_idx):
        """
        FAST VERSION

        영상 하나를 처음 사용할 때 전체 frame을 한 번에 PNG로 추출.
        이후에는 필요한 frame을 cache에서 cv2.imread().
        """

        import hashlib

        video_key = hashlib.md5(
            video_path.encode("utf-8")
        ).hexdigest()

        video_cache = os.path.join(
            self.cache_dir,
            video_key
        )

        done_file = os.path.join(
            video_cache,
            ".done"
        )

        os.makedirs(
            video_cache,
            exist_ok=True
        )

        if not os.path.exists(done_file):

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

            output_pattern = os.path.join(
                video_cache,
                "%06d.png"
            )

            cmd = [
                ffmpeg,
                "-loglevel", "error",
                "-y",
                "-i", video_path,
                "-vsync", "0",
                output_pattern,
            ]

            print(
                f"[CACHE] extracting full video:\n"
                f"        {video_path}"
            )

            result = subprocess.run(cmd)

            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg 전체 frame 추출 실패: {video_path}"
                )

            with open(done_file, "w") as f:
                f.write("done")

        # ffmpeg numbering은 1부터 시작
        image_path = os.path.join(
            video_cache,
            f"{frame_idx + 1:06d}.png"
        )

        img = cv2.imread(image_path)

        if img is None:
            raise RuntimeError(
                f"cache frame 읽기 실패: {image_path}"
            )

        return img


    def _get_roi_mask(self, subject):
        """
        roi_mode='multi'일 때 subject별 forehead+cheek 마스크를
        (image_size, image_size)로 반환. subject의 ROI 박스가 없으면
        (얼굴 검출 실패 등) None을 반환해 Full Face로 fallback.
        """
        if self.roi_mode != "multi":
            return None

        subj_key = str(subject).zfill(2)

        entry = self.roi_boxes.get(subj_key)

        if entry is None:
            return None

        return build_mask(
            (self.image_size, self.image_size),
            entry["roi_boxes"]
        )

    def _preprocess_frame(self, frame, roi_mask=None):
        """
        전체 frame을 112x112로 resize.
        roi_mask가 주어지면 (multi-ROI 모드) forehead+cheek
        영역 밖은 0으로 마스킹한다.
        """
        frame = cv2.resize(
            frame,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = (
            frame.astype(np.float32)
            / 255.0
        )

        if roi_mask is not None:
            frame = frame * roi_mask[..., None]

        frame = np.transpose(
            frame,
            (2, 0, 1)
        )

        return torch.from_numpy(frame)

    def _extract_quality(self, frame_bgr, lux):
        gray = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2GRAY
        )

        brightness = float(
            gray.mean()
        )

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        dark_ratio = float(
            (gray < 40).mean()
        )

        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "dark_ratio": dark_ratio,
            "lux": float(lux),
        }

    def __getitem__(self, idx):
        win = self.windows[idx]

        frames = []
        bvps = []
        hrs = []

        blur_scores = []
        brightnesses = []
        dark_ratios = []
        lux_values = []

        first_row = self.rows[win[0]]

        subject = first_row["subject"]
        lux = first_row["lux"]
        video_path = first_row["video_path"]

        roi_mask = self._get_roi_mask(subject)

        for row_idx in win:
            r = self.rows[row_idx]

            frame = self._extract_one_frame_ffmpeg(
                video_path,
                r["frame_idx"]
            )

            q = self._extract_quality(
                frame,
                lux
            )

            x = self._preprocess_frame(
                frame,
                roi_mask
            )

            frames.append(x)

            bvps.append(
                r["bvp"]
            )

            hrs.append(
                r["hr_bpm"]
            )

            blur_scores.append(
                q["blur_score"]
            )

            brightnesses.append(
                q["brightness"]
            )

            dark_ratios.append(
                q["dark_ratio"]
            )

            lux_values.append(
                q["lux"]
            )

        frames = torch.stack(
            frames,
            dim=0
        )

        bvp = torch.tensor(
            bvps,
            dtype=torch.float32
        )

        hr = torch.tensor(
            hrs,
            dtype=torch.float32
        )

        quality = {
            "blur_score": torch.tensor(
                blur_scores,
                dtype=torch.float32
            ),
            "brightness": torch.tensor(
                brightnesses,
                dtype=torch.float32
            ),
            "dark_ratio": torch.tensor(
                dark_ratios,
                dtype=torch.float32
            ),
            "lux": torch.tensor(
                lux_values,
                dtype=torch.float32
            ),
        }

        return {
            "frames": frames,
            "bvp": bvp,
            "hr": hr,
            "quality": quality,
            "subject": subject,
            "lux": torch.tensor(
                lux,
                dtype=torch.float32
            ),
        }

