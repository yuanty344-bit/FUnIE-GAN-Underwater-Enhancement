"""
视频帧提取工具
"""
import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frame(video_path: str, frame_index: int = 0) -> Optional[np.ndarray]:
    """从视频提取指定帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_frames(video_path: str, start: int = 0, end: int = -1,
                   step: int = 1) -> List[np.ndarray]:
    """提取帧范围"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end < 0 or end > total:
        end = total

    frames = []
    for idx in range(start, end, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def get_video_info(video_path: str) -> Optional[dict]:
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
    }
    cap.release()
    return info


def extract_thumbnail(video_path: str, output_path: str, time_sec: float = 1.0) -> bool:
    """提取视频缩略图"""
    info = get_video_info(video_path)
    if info is None:
        return False
    fps = info["fps"]
    frame_idx = int(time_sec * fps)
    frame = extract_frame(video_path, frame_idx)
    if frame is None:
        return False
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(output_path, bgr)
