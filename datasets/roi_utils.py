import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_MODEL_PATH = "models/face_landmarker.task"

_landmarker = None


def _get_landmarker():
    global _landmarker

    if _landmarker is None:
        base_options = mp_python.BaseOptions(
            model_asset_path=_MODEL_PATH
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )

        _landmarker = mp_vision.FaceLandmarker.create_from_options(
            options
        )

    return _landmarker


def detect_face_bbox(frame_bgr):
    """
    frame_bgr: HxWx3 uint8 BGR image (as read by cv2)

    Returns normalized (x1,y1,x2,y2) in [0,1], or None if no face found.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = _get_landmarker().detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    return (
        float(min(xs)),
        float(min(ys)),
        float(max(xs)),
        float(max(ys)),
    )


def compute_roi_boxes(bbox):
    """
    bbox: normalized face (x1,y1,x2,y2) in full-frame coordinates.

    Returns dict of normalized (x1,y1,x2,y2) boxes, still in
    full-frame coordinates, approximating forehead / left cheek /
    right cheek via fixed fractions of the face bounding box.

    NOTE: heuristic approximation (no hairline/cheek landmarks used
    directly) -- good enough for a Full-Face vs Multi-ROI ablation,
    not meant to be anatomically precise.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    forehead = (
        x1 + 0.25 * w,
        y1 + 0.05 * h,
        x1 + 0.75 * w,
        y1 + 0.25 * h,
    )

    left_cheek = (
        x1 + 0.05 * w,
        y1 + 0.45 * h,
        x1 + 0.35 * w,
        y1 + 0.75 * h,
    )

    right_cheek = (
        x1 + 0.65 * w,
        y1 + 0.45 * h,
        x1 + 0.95 * w,
        y1 + 0.75 * h,
    )

    return {
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
    }


def build_mask(image_size, roi_boxes):
    """
    image_size: (H, W)
    roi_boxes: dict of normalized (x1,y1,x2,y2) boxes in full-frame
               coordinates (same convention regardless of the actual
               pixel resolution of the frame being masked).

    Returns float32 mask (H, W) with 1.0 inside the ROI union,
    0.0 elsewhere.
    """
    H, W = image_size

    mask = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2) in roi_boxes.values():
        px1 = max(0, int(round(x1 * W)))
        py1 = max(0, int(round(y1 * H)))
        px2 = min(W, int(round(x2 * W)))
        py2 = min(H, int(round(y2 * H)))

        mask[py1:py2, px1:px2] = 1.0

    return mask
