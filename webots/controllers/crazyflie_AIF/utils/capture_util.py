import os
from datetime import datetime

import numpy as np
from PIL import Image


def save_capture(image, width, height, timestamp=None, output_dir=None, prefix="capture"):
    if image is None or width is None or height is None:
        return None

    frame = np.frombuffer(image, dtype=np.uint8)
    expected = int(width * height * 4)
    if frame.size != expected:
        return None

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "logs", "current_run_captures")
    os.makedirs(output_dir, exist_ok=True)

    frame = frame.reshape((height, width, 4))
    frame = frame[:, :, :3]  # BGRA -> BGR
    frame = frame[:, :, ::-1]  # BGR -> RGB

    if timestamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    else:
        stamp = f"{timestamp:.6f}".replace(".", "_")

    filename = f"{prefix}_{stamp}.png"
    path = os.path.join(output_dir, filename)
    Image.fromarray(frame).save(path)
    return path
