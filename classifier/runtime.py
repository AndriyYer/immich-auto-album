"""Shared YOLO classification primitives.

Used by both the one-shot `backlog.py` runner (typically on a GPU host) and the
`image_classifier` source handler (in the polling sidecar, typically CPU-only).
The Windows GPU DLL setup below is a no-op on Linux, so the same module loads
correctly in both environments.
"""

import os
import site

# Windows + onnxruntime-gpu: bundled NVIDIA wheel DLLs are not on PATH by
# default. Has to run before `import onnxruntime`. No-op on Linux.
if hasattr(os, 'add_dll_directory'):
    for _sp in site.getsitepackages():
        _nvidia = os.path.join(_sp, 'nvidia')
        if not os.path.isdir(_nvidia):
            continue
        for _sub in os.listdir(_nvidia):
            _bin = os.path.join(_nvidia, _sub, 'bin')
            if os.path.isdir(_bin):
                os.add_dll_directory(_bin)
                os.environ['PATH'] = _bin + os.pathsep + os.environ.get('PATH', '')

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = 640

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]
CLASS_TO_ID = {name: i for i, name in enumerate(COCO_CLASSES)}


def load_session(model_path, prefer_gpu=True):
    providers = []
    if prefer_gpu:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    return ort.InferenceSession(model_path, providers=providers)


def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = new_shape - new_h, new_shape - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def preprocess(img_bgr):
    img = letterbox(img_bgr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)[None]


def score_per_class(session, input_name, img_bgr):
    """Run YOLO11/v8-format inference and return max conf per COCO class (length 80)."""
    inp = preprocess(img_bgr)
    out = session.run(None, {input_name: inp})[0]
    # out shape: (1, 4+nc, num_anchors). Drop the 4 box coords, take per-anchor max per class.
    class_scores = out[0, 4:, :]
    return class_scores.max(axis=1).astype(float)


def parse_duration(duration_str):
    """'00:01:23.500000' -> 83.5 seconds. Empty/None -> 0."""
    if not duration_str:
        return 0.0
    h, m, rest = duration_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(rest)
