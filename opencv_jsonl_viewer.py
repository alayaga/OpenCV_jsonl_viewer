#!/usr/bin/env python3
"""
OpenCV + Tkinter 视频检测结果查看器。

左侧播放 match_1_clip 中某一路 m3u8 视频，并把 output_0.jsonl 中对应帧、
对应相机的 players / balls 检测结果画到画面上。

右侧显示当前帧当前相机的 JSON 原始数据。
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


# =========================
# 直接修改下面这些参数即可
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# 输入视频目录：里面应包含 101.m3u8 ... 124.m3u8。
VIDEO_DIR = PROJECT_DIR / "videos" / "match_1_clip"

# 2D pipeline 输出的 JSONL。
JSONL_PATH = PROJECT_DIR / "output_0.jsonl"

# 按 pts 命名的 JPEG 帧目录。
FRAME_IMAGE_DIR = PROJECT_DIR / "videos" / "match_1_clip_jpeg_frames"

# 默认打开的相机。
DEFAULT_CAMERA = 101

# 播放窗口中的视频显示尺寸。只影响显示，不改变原始视频和 bbox 坐标。
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# 视频显示大小滑块默认比例。1.00 对应上面的默认显示尺寸。
DEFAULT_VIDEO_SCALE = 1.00

# 叠加信息默认文字大小。运行时可以在界面里调整。
DEFAULT_LABEL_SCALE = 0.48

# 自动播放时的刷新间隔，单位毫秒。30fps 约等于 33ms。
PLAY_INTERVAL_MS = 33

# 排错模式下的异常阈值。
ABNORMAL_DET_THRESHOLD = 0.60
ABNORMAL_TEAM_SCORE_THRESHOLD = 0.60
ABNORMAL_ID_SCORE_THRESHOLD = 0.60


TEAM_COLORS_BGR = {
    "team_red": (60, 80, 255),
    "team_black": (190, 190, 190),
    "goalkeeper_lightblue": (255, 210, 80),
    "goalkeeper_darkblue": (255, 110, 80),
    None: (230, 230, 230),
}

BALL_COLOR_BGR = (0, 220, 255)
KEYPOINT_COLOR_BGR = (120, 255, 120)
HEADER_TEXT_COLOR_BGR = (255, 255, 255)


@dataclass
class CameraAnnotation:
    camera_id: int
    items: list[dict[str, Any]] = field(default_factory=list)
    pts_values: list[int] = field(default_factory=list)
    pts_to_index: dict[int, int] = field(default_factory=dict)


@dataclass
class AnnotationStore:
    cameras: dict[int, CameraAnnotation] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def require_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if Image is None or ImageTk is None:
        missing.append("pillow")

    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "缺少运行依赖："
            + ", ".join(missing)
            + "\n请先安装：\n"
            + f"  pip install {packages}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="播放 m3u8 并叠加 JSONL 检测结果")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR,
        help=f"视频目录，仅视频模式需要，默认：{VIDEO_DIR}",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=JSONL_PATH,
        help=f"JSONL 文件，默认：{JSONL_PATH}",
    )
    parser.add_argument(
        "--frame-image-dir",
        type=Path,
        default=FRAME_IMAGE_DIR,
        help=f"按 pts 命名的 JPEG 帧目录，默认：{FRAME_IMAGE_DIR}",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA,
        help=f"默认相机编号，默认：{DEFAULT_CAMERA}",
    )
    return parser.parse_args()


def camera_id_from_item(item: dict[str, Any]) -> int | None:
    input_url = item.get("input_url") or ""
    filename = input_url.rsplit("/", 1)[-1]
    if filename.endswith(".m3u8"):
        stem = filename[:-5]
        if stem.isdigit():
            return int(stem)

    cam_idx = item.get("result", {}).get("cam_idx")
    if isinstance(cam_idx, int):
        return cam_idx + 101
    return None


def load_annotations(jsonl_path: Path) -> AnnotationStore:
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"找不到 JSONL 文件：{jsonl_path}")

    store = AnnotationStore()
    total_lines = 0
    bad_lines = 0
    player_count = 0
    ball_count = 0
    camera_line_counts: dict[int, int] = defaultdict(int)

    print(f"正在读取 JSONL：{jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            camera_id = camera_id_from_item(item)
            pts = item.get("pts")
            if camera_id is None or pts is None:
                bad_lines += 1
                continue

            total_lines += 1
            camera_annotation = store.cameras.setdefault(
                camera_id,
                CameraAnnotation(camera_id=camera_id),
            )
            camera_annotation.items.append(item)
            camera_annotation.pts_values.append(int(pts))
            camera_annotation.pts_to_index[int(pts)] = len(camera_annotation.items) - 1
            camera_line_counts[camera_id] += 1

            result = item.get("result") or {}
            player_count += len(result.get("players") or [])
            ball_count += len(result.get("balls") or [])

    for camera_annotation in store.cameras.values():
        order = sorted(range(len(camera_annotation.items)), key=lambda idx: camera_annotation.pts_values[idx])
        camera_annotation.items = [camera_annotation.items[idx] for idx in order]
        camera_annotation.pts_values = [camera_annotation.pts_values[idx] for idx in order]
        camera_annotation.pts_to_index = {
            pts: idx
            for idx, pts in enumerate(camera_annotation.pts_values)
        }

    store.summary = {
        "jsonl_path": str(jsonl_path),
        "total_lines": total_lines,
        "bad_lines": bad_lines,
        "camera_count": len(store.cameras),
        "camera_line_counts": dict(sorted(camera_line_counts.items())),
        "player_count": player_count,
        "ball_count": ball_count,
    }
    print(f"JSONL 读取完成：{total_lines} 行，{len(store.cameras)} 路相机。")
    return store


def nearest_index_by_pts(annotation: CameraAnnotation, pts: int) -> int:
    if not annotation.pts_values:
        return 0

    if pts in annotation.pts_to_index:
        return annotation.pts_to_index[pts]

    position = bisect.bisect_left(annotation.pts_values, pts)
    if position <= 0:
        return 0
    if position >= len(annotation.pts_values):
        return len(annotation.pts_values) - 1

    before = annotation.pts_values[position - 1]
    after = annotation.pts_values[position]
    return position - 1 if abs(pts - before) <= abs(after - pts) else position


def player_id_text(player: dict[str, Any]) -> str:
    player_id = player.get("player_id")
    if player_id is None:
        return "null"
    return str(player_id)


def label_metrics(
    text: str,
    font_scale: float = 0.72,
    thickness: int = 2,
) -> tuple[int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    rect_width = text_width + 6
    rect_height = text_height + 2 * baseline + 6
    baseline_offset = text_height + baseline + 4
    return rect_width, rect_height, baseline_offset


def label_rect_from_top_left(
    frame: Any,
    text: str,
    left: float,
    top: float,
    font_scale: float,
    thickness: int,
) -> tuple[int, int, tuple[int, int, int, int]]:
    rect_width, rect_height, baseline_offset = label_metrics(text, font_scale, thickness)
    max_left = max(0, frame.shape[1] - rect_width)
    max_top = max(0, frame.shape[0] - rect_height)
    clamped_left = max(0, min(int(round(left)), max_left))
    clamped_top = max(0, min(int(round(top)), max_top))
    baseline_y = clamped_top + baseline_offset
    rect = (
        clamped_left,
        clamped_top,
        clamped_left + rect_width,
        clamped_top + rect_height,
    )
    return clamped_left, baseline_y, rect


def draw_label(
    frame: Any,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.72,
    thickness: int = 2,
) -> tuple[int, int, int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    _, _, baseline_offset = label_metrics(text, font_scale, thickness)
    x, y, rect = label_rect_from_top_left(
        frame,
        text,
        x,
        y - baseline_offset,
        font_scale,
        thickness,
    )
    cv2.rectangle(
        frame,
        (rect[0], rect[1]),
        (rect[2], rect[3]),
        (0, 0, 0),
        -1,
    )
    cv2.putText(frame, text, (x + 3, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return rect


def text_rect(
    frame: Any,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
) -> tuple[int, int, int, int]:
    _, _, baseline_offset = label_metrics(text, font_scale, thickness)
    return label_rect_from_top_left(frame, text, x, y - baseline_offset, font_scale, thickness)[2]


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def expanded_rect(rect: tuple[int, int, int, int], padding: int) -> tuple[int, int, int, int]:
    return (
        rect[0] - padding,
        rect[1] - padding,
        rect[2] + padding,
        rect[3] + padding,
    )


def overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def rect_center(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)


def closest_point_on_rect(
    rect: tuple[int, int, int, int],
    point: tuple[int, int],
) -> tuple[int, int]:
    x, y = point
    return (
        max(rect[0], min(x, rect[2])),
        max(rect[1], min(y, rect[3])),
    )


def draw_leader_line(
    frame: Any,
    bbox_rect: tuple[int, int, int, int],
    label_rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    bbox_center = rect_center(bbox_rect)
    label_center = rect_center(label_rect)
    start = closest_point_on_rect(bbox_rect, label_center)
    end = closest_point_on_rect(label_rect, bbox_center)

    if abs(start[0] - end[0]) <= 2 and abs(start[1] - end[1]) <= 2:
        return

    line_thickness = max(1, thickness)
    cv2.line(frame, start, end, color, line_thickness, cv2.LINE_AA)
    cv2.circle(frame, start, max(2, line_thickness + 1), color, -1, cv2.LINE_AA)


def find_label_position(
    frame: Any,
    text: str,
    bbox_rect: tuple[int, int, int, int],
    occupied_rects: list[tuple[int, int, int, int]],
    font_scale: float,
    thickness: int,
) -> tuple[int, int]:
    frame_h, frame_w = frame.shape[:2]
    rect_width, rect_height, _ = label_metrics(text, font_scale, thickness)
    padding = max(6, int(round(10 * font_scale)))
    gap = max(3, int(round(5 * font_scale)))
    x1, y1, x2, y2 = bbox_rect
    center_x = (x1 + x2 - rect_width) / 2
    center_y = (y1 + y2 - rect_height) / 2
    above_y = y1 - rect_height - padding
    below_y = y2 + padding
    left_x = x1 - rect_width - padding
    right_x = x2 + padding

    candidates: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    def add_candidate(left: float, top: float) -> None:
        key = (int(round(left)), int(round(top)))
        if key in seen:
            return
        seen.add(key)
        candidates.append((left, top))

    local_lefts = [x1, center_x, x2 - rect_width, left_x, right_x]
    local_tops = [above_y, below_y, center_y, y1 + padding, y2 - rect_height - padding]
    for top in local_tops:
        for left in local_lefts:
            add_candidate(left, top)

    vertical_step = rect_height + padding
    horizontal_step = max(rect_width // 2, 48)
    for ring in range(1, 9):
        vertical_offset = ring * vertical_step
        horizontal_offset = ring * horizontal_step
        for left in local_lefts:
            add_candidate(left, above_y - vertical_offset)
            add_candidate(left, below_y + vertical_offset)
        for top in [above_y, below_y, center_y]:
            add_candidate(left_x - horizontal_offset, top)
            add_candidate(right_x + horizontal_offset, top)

    max_left = max(0, frame_w - rect_width)
    max_top = max(0, frame_h - rect_height)
    row_step = max(rect_height + padding, 18)
    col_step = max(rect_width + padding, 80)
    rows = list(range(0, max_top + 1, row_step))
    if max_top not in rows:
        rows.append(max_top)
    cols = list(range(0, max_left + 1, col_step))
    if max_left not in cols:
        cols.append(max_left)

    preferred_top = max(0, min(above_y, max_top))
    preferred_left = max(0, min(x1, max_left))
    rows.sort(key=lambda row: (abs(row - preferred_top), row))
    cols.sort(key=lambda col: (abs(col - preferred_left), col))
    for left in cols:
        for top in rows:
            add_candidate(left, top)

    best_position: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None
    for left, top in candidates:
        label_x, label_y, rect = label_rect_from_top_left(frame, text, left, top, font_scale, thickness)
        padded_rect = expanded_rect(rect, gap)
        overlap_score = sum(
            overlap_area(padded_rect, expanded_rect(occupied, gap))
            for occupied in occupied_rects
        )
        distance_score = abs(rect[0] - x1) + abs(rect[1] - above_y)
        edge_score = 1 if left < 0 or top < 0 or left > max_left or top > max_top else 0
        score = (overlap_score, edge_score, distance_score)
        if best_score is None or score < best_score:
            best_score = score
            best_position = (label_x, label_y)
        if overlap_score == 0:
            return label_x, label_y

    if best_position is not None:
        return best_position
    return x1, max(0, y1 - padding)


def safe_score(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_abnormal_player(player: dict[str, Any]) -> bool:
    return is_abnormal_player_with_thresholds(
        player,
        ABNORMAL_DET_THRESHOLD,
        ABNORMAL_TEAM_SCORE_THRESHOLD,
        ABNORMAL_ID_SCORE_THRESHOLD,
    )


def is_abnormal_player_with_thresholds(
    player: dict[str, Any],
    det_threshold: float,
    team_score_threshold: float,
    id_score_threshold: float,
) -> bool:
    player_id = player.get("player_id")
    player_id_missing = player_id is None or str(player_id) == "-1"
    return (
        player_id_missing
        or safe_score(player.get("score")) < det_threshold
        or safe_score(player.get("reid_score")) < team_score_threshold
        or safe_score(player.get("player_id_score")) < id_score_threshold
    )


def build_player_label(
    player: dict[str, Any],
    label_mode: str,
    show_id: bool,
    show_team: bool,
    show_det: bool,
    show_team_score: bool,
    show_id_score: bool,
    det_threshold: float,
    team_score_threshold: float,
    id_score_threshold: float,
) -> str:
    player_id = player_id_text(player)
    if label_mode == "排错模式":
        if not is_abnormal_player_with_thresholds(
            player,
            det_threshold,
            team_score_threshold,
            id_score_threshold,
        ):
            return f"id:{player_id}"
        return (
            f"id:{player_id} "
            f"{player.get('reid_team')} "
            f"det:{safe_score(player.get('score')):.2f} "
            f"team_s:{safe_score(player.get('reid_score')):.2f} "
            f"id_s:{safe_score(player.get('player_id_score')):.2f}"
        )

    parts: list[str] = []
    if show_id:
        parts.append(f"id:{player_id}")
    if show_team:
        parts.append(str(player.get("reid_team")))
    if show_det:
        parts.append(f"det:{safe_score(player.get('score')):.2f}")
    if show_team_score:
        parts.append(f"team_s:{safe_score(player.get('reid_score')):.2f}")
    if show_id_score:
        parts.append(f"id_s:{safe_score(player.get('player_id_score')):.2f}")
    return " ".join(parts)


def draw_detections(
    frame: Any,
    item: dict[str, Any],
    show_players: bool,
    show_balls: bool,
    show_keypoints: bool,
    scale_x: float,
    scale_y: float,
    label_scale: float,
    label_mode: str,
    show_label_id: bool,
    show_label_team: bool,
    show_label_det: bool,
    show_label_team_score: bool,
    show_label_id_score: bool,
    det_threshold: float,
    team_score_threshold: float,
    id_score_threshold: float,
) -> None:
    result = item.get("result") or {}
    box_thickness = max(1, int(round(label_scale * 3)))
    point_radius = max(1, int(round(label_scale * 3)))
    label_thickness = max(1, int(round(label_scale * 3)))
    header_reserved_height = max(34, int(round(56 * label_scale)))
    occupied_labels: list[tuple[int, int, int, int]] = [
        (0, 0, int(frame.shape[1] * 0.72), header_reserved_height)
    ]

    if show_players:
        for player in result.get("players") or []:
            bbox = player.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [
                int(round(float(bbox[0]) * scale_x)),
                int(round(float(bbox[1]) * scale_y)),
                int(round(float(bbox[2]) * scale_x)),
                int(round(float(bbox[3]) * scale_y)),
            ]
            team = player.get("reid_team")
            color = TEAM_COLORS_BGR.get(team, (230, 230, 230))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            label = build_player_label(
                player,
                label_mode,
                show_label_id,
                show_label_team,
                show_label_det,
                show_label_team_score,
                show_label_id_score,
                det_threshold,
                team_score_threshold,
                id_score_threshold,
            )
            if label:
                label_x, label_y = find_label_position(
                    frame,
                    label,
                    (x1, y1, x2, y2),
                    occupied_labels,
                    label_scale,
                    label_thickness,
                )
                rect = text_rect(frame, label, label_x, label_y, label_scale, label_thickness)
                draw_leader_line(
                    frame,
                    (x1, y1, x2, y2),
                    rect,
                    color,
                    max(1, label_thickness),
                )
                rect = draw_label(
                    frame,
                    label,
                    label_x,
                    label_y,
                    color,
                    font_scale=label_scale,
                    thickness=label_thickness,
                )
                occupied_labels.append(rect)

            if show_keypoints:
                for point in player.get("2d_keypoints") or []:
                    if not isinstance(point, list) or len(point) < 2:
                        continue
                    px = int(round(float(point[0]) * scale_x))
                    py = int(round(float(point[1]) * scale_y))
                    cv2.circle(frame, (px, py), point_radius, KEYPOINT_COLOR_BGR, -1)

    if show_balls:
        for ball in result.get("balls") or []:
            bbox = ball.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [
                int(round(float(bbox[0]) * scale_x)),
                int(round(float(bbox[1]) * scale_y)),
                int(round(float(bbox[2]) * scale_x)),
                int(round(float(bbox[3]) * scale_y)),
            ]
            cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR_BGR, max(2, box_thickness + 1))
            label = f"ball {float(ball.get('score', 0)):.2f}"
            label_x, label_y = find_label_position(
                frame,
                label,
                (x1, y1, x2, y2),
                occupied_labels,
                label_scale,
                label_thickness,
            )
            rect = text_rect(frame, label, label_x, label_y, label_scale, label_thickness)
            draw_leader_line(
                frame,
                (x1, y1, x2, y2),
                rect,
                BALL_COLOR_BGR,
                max(1, label_thickness),
            )
            rect = draw_label(
                frame,
                label,
                label_x,
                label_y,
                BALL_COLOR_BGR,
                font_scale=label_scale,
                thickness=label_thickness,
            )
            occupied_labels.append(rect)


class OpenCvJsonlViewer:
    def __init__(
        self,
        root: tk.Tk,
        video_dir: Path,
        frame_image_dir: Path,
        store: AnnotationStore,
        initial_camera: int,
    ):
        self.root = root
        self.video_dir = video_dir
        self.frame_image_dir = frame_image_dir
        self.store = store
        self.current_camera = initial_camera
        self.current_index = 0
        self.capture: Any | None = None
        self.capture_next_index = 0
        self.current_raw_frame = None
        self.current_frame_image = None
        self.canvas_image_id: int | None = None
        self.json_offset_frames = 0
        self.is_updating_progress = False
        self.is_playing = False
        self.last_tick_time = time.time()
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT

        self.camera_var = tk.StringVar(value=str(initial_camera))
        self.mode_var = tk.StringVar(value="图片模式")
        self.pts_var = tk.StringVar(value="")
        self.frame_var = tk.StringVar(value="1")
        self.progress_var = tk.DoubleVar(value=1)
        self.json_offset_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="")
        self.display_width_var = tk.StringVar(value=str(DISPLAY_WIDTH))
        self.display_height_var = tk.StringVar(value=str(DISPLAY_HEIGHT))
        self.video_scale_var = tk.DoubleVar(value=DEFAULT_VIDEO_SCALE)
        self.video_scale_text_var = tk.StringVar(value=f"{DEFAULT_VIDEO_SCALE:.2f}")
        self.show_players = tk.BooleanVar(value=True)
        self.show_balls = tk.BooleanVar(value=True)
        self.show_keypoints = tk.BooleanVar(value=False)
        self.label_mode_var = tk.StringVar(value="详细模式")
        self.show_label_id = tk.BooleanVar(value=True)
        self.show_label_team = tk.BooleanVar(value=True)
        self.show_label_det = tk.BooleanVar(value=True)
        self.show_label_team_score = tk.BooleanVar(value=True)
        self.show_label_id_score = tk.BooleanVar(value=True)
        self.label_scale_var = tk.DoubleVar(value=DEFAULT_LABEL_SCALE)
        self.label_scale_text_var = tk.StringVar(value=f"{DEFAULT_LABEL_SCALE:.2f}")
        self.current_det_threshold = ABNORMAL_DET_THRESHOLD
        self.current_team_score_threshold = ABNORMAL_TEAM_SCORE_THRESHOLD
        self.current_id_score_threshold = ABNORMAL_ID_SCORE_THRESHOLD
        self.det_threshold_var = tk.StringVar(value=f"{ABNORMAL_DET_THRESHOLD:.2f}")
        self.team_score_threshold_var = tk.StringVar(value=f"{ABNORMAL_TEAM_SCORE_THRESHOLD:.2f}")
        self.id_score_threshold_var = tk.StringVar(value=f"{ABNORMAL_ID_SCORE_THRESHOLD:.2f}")

        self.root.title("OpenCV JSONL 检测结果查看器")
        self.root.geometry("1500x900")
        self.root.minsize(1100, 700)

        self.build_layout()
        self.open_camera(initial_camera)

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)

        row1 = ttk.Frame(toolbar)
        row1.grid(row=0, column=0, sticky="ew")
        row2 = ttk.Frame(toolbar)
        row2.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        row3 = ttk.Frame(toolbar)
        row3.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        progress_row = ttk.Frame(toolbar)
        progress_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.threshold_row = ttk.Frame(toolbar)
        self.threshold_row.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        status_row = ttk.Frame(toolbar)
        status_row.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        status_row.columnconfigure(0, weight=1)

        row1.columnconfigure(14, weight=1)
        row2.columnconfigure(16, weight=1)
        row3.columnconfigure(16, weight=1)
        progress_row.columnconfigure(1, weight=1)
        self.threshold_row.columnconfigure(8, weight=1)

        ttk.Label(row1, text="相机").grid(row=0, column=0, padx=(0, 4))
        cameras = [str(camera_id) for camera_id in sorted(self.store.cameras)]
        camera_box = ttk.Combobox(
            row1,
            width=8,
            textvariable=self.camera_var,
            values=cameras,
            state="readonly",
        )
        camera_box.grid(row=0, column=1, padx=4)
        camera_box.bind("<<ComboboxSelected>>", lambda _event: self.change_camera())

        ttk.Label(row1, text="模式").grid(row=0, column=2, padx=(10, 4))
        mode_box = ttk.Combobox(
            row1,
            width=10,
            textvariable=self.mode_var,
            values=["视频模式", "图片模式"],
            state="readonly",
        )
        mode_box.grid(row=0, column=3, padx=4)
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self.change_mode())

        ttk.Button(row1, text="播放/暂停", command=self.toggle_play).grid(row=0, column=4, padx=(14, 4))
        ttk.Button(row1, text="上一帧", command=self.prev_frame).grid(row=0, column=5, padx=4)
        ttk.Button(row1, text="第一帧", command=self.first_frame).grid(row=0, column=6, padx=4)
        ttk.Button(row1, text="下一帧", command=self.next_frame).grid(row=0, column=7, padx=4)

        ttk.Label(row1, text="帧序号").grid(row=0, column=8, padx=(14, 4))
        ttk.Entry(row1, width=10, textvariable=self.frame_var).grid(row=0, column=9, padx=4)
        ttk.Button(row1, text="跳帧", command=self.jump_frame).grid(row=0, column=10, padx=4)

        ttk.Label(row1, text="PTS").grid(row=0, column=11, padx=(14, 4))
        ttk.Entry(row1, width=16, textvariable=self.pts_var).grid(row=0, column=12, padx=4)
        ttk.Button(row1, text="跳 PTS", command=self.jump_pts).grid(row=0, column=13, padx=4)

        ttk.Checkbutton(row2, text="players", variable=self.show_players, command=self.render_current).grid(row=0, column=0, padx=(0, 4))
        ttk.Checkbutton(row2, text="balls", variable=self.show_balls, command=self.render_current).grid(row=0, column=1, padx=4)
        ttk.Checkbutton(row2, text="keypoints", variable=self.show_keypoints, command=self.render_current).grid(row=0, column=2, padx=4)

        ttk.Label(row2, text="视频大小").grid(row=0, column=3, padx=(14, 4))
        video_scale = ttk.Scale(
            row2,
            from_=0.25,
            to=1.2,
            variable=self.video_scale_var,
            orient=tk.HORIZONTAL,
            command=self.on_video_scale_change,
        )
        video_scale.grid(row=0, column=4, padx=4, sticky="ew")
        ttk.Label(row2, textvariable=self.video_scale_text_var, width=5).grid(row=0, column=5, padx=(0, 4))

        ttk.Label(row2, text="显示宽").grid(row=0, column=6, padx=(10, 4))
        ttk.Entry(row2, width=8, textvariable=self.display_width_var).grid(row=0, column=7, padx=4)
        ttk.Label(row2, text="显示高").grid(row=0, column=8, padx=4)
        ttk.Entry(row2, width=8, textvariable=self.display_height_var).grid(row=0, column=9, padx=4)
        ttk.Button(row2, text="应用显示", command=self.apply_display_settings).grid(row=0, column=10, padx=4)

        ttk.Label(row2, text="文字大小").grid(row=0, column=11, padx=(14, 4))
        label_scale = ttk.Scale(
            row2,
            from_=0.25,
            to=1.2,
            variable=self.label_scale_var,
            orient=tk.HORIZONTAL,
            command=self.on_label_scale_change,
        )
        label_scale.grid(row=0, column=12, padx=4, sticky="ew")
        ttk.Label(row2, textvariable=self.label_scale_text_var, width=5).grid(row=0, column=13, padx=(0, 4))

        ttk.Label(row3, text="标签模式").grid(row=0, column=0, padx=(0, 4))
        label_mode_box = ttk.Combobox(
            row3,
            width=10,
            textvariable=self.label_mode_var,
            values=["排错模式", "详细模式"],
            state="readonly",
        )
        label_mode_box.grid(row=0, column=1, padx=4)
        label_mode_box.bind("<<ComboboxSelected>>", lambda _event: self.on_label_mode_change())

        self.label_option_widgets: list[ttk.Checkbutton] = []
        option_specs = [
            ("id", self.show_label_id),
            ("team", self.show_label_team),
            ("det", self.show_label_det),
            ("team_s", self.show_label_team_score),
            ("id_s", self.show_label_id_score),
        ]
        for offset, (text, variable) in enumerate(option_specs, start=2):
            checkbox = ttk.Checkbutton(
                row3,
                text=text,
                variable=variable,
                command=self.render_current,
            )
            checkbox.grid(row=0, column=offset, padx=4)
            self.label_option_widgets.append(checkbox)

        ttk.Label(row3, text="JSON偏移").grid(row=0, column=8, padx=(18, 4))
        ttk.Button(row3, text="←", width=3, command=lambda: self.step_json_offset(-1)).grid(row=0, column=9, padx=2)
        json_offset_entry = ttk.Entry(row3, width=7, textvariable=self.json_offset_var)
        json_offset_entry.grid(row=0, column=10, padx=2)
        json_offset_entry.bind("<Return>", lambda _event: self.apply_json_offset())
        ttk.Button(row3, text="→", width=3, command=lambda: self.step_json_offset(1)).grid(row=0, column=11, padx=2)
        ttk.Button(row3, text="应用", command=self.apply_json_offset).grid(row=0, column=12, padx=(6, 2))
        ttk.Button(row3, text="归零", command=self.reset_json_offset).grid(row=0, column=13, padx=2)

        ttk.Label(progress_row, text="播放进度").grid(row=0, column=0, padx=(0, 8))
        self.progress_scale = ttk.Scale(
            progress_row,
            from_=1,
            to=1,
            variable=self.progress_var,
            orient=tk.HORIZONTAL,
            command=self.on_progress_drag,
        )
        self.progress_scale.grid(row=0, column=1, sticky="ew", padx=4)
        self.progress_scale.bind("<ButtonRelease-1>", lambda _event: self.apply_progress_seek())
        self.progress_scale.bind("<Return>", lambda _event: self.apply_progress_seek())

        ttk.Label(self.threshold_row, text="异常阈值").grid(row=0, column=0, padx=(0, 8))
        ttk.Label(self.threshold_row, text="det").grid(row=0, column=1, padx=(0, 4))
        det_entry = ttk.Entry(self.threshold_row, width=7, textvariable=self.det_threshold_var)
        det_entry.grid(row=0, column=2, padx=(0, 10))
        ttk.Label(self.threshold_row, text="team_s").grid(row=0, column=3, padx=(0, 4))
        team_score_entry = ttk.Entry(self.threshold_row, width=7, textvariable=self.team_score_threshold_var)
        team_score_entry.grid(row=0, column=4, padx=(0, 10))
        ttk.Label(self.threshold_row, text="id_s").grid(row=0, column=5, padx=(0, 4))
        id_score_entry = ttk.Entry(self.threshold_row, width=7, textvariable=self.id_score_threshold_var)
        id_score_entry.grid(row=0, column=6, padx=(0, 10))
        ttk.Button(self.threshold_row, text="应用阈值", command=self.apply_threshold_settings).grid(row=0, column=7, padx=4)
        for entry in [det_entry, team_score_entry, id_score_entry]:
            entry.bind("<Return>", lambda _event: self.apply_threshold_settings())

        ttk.Label(status_row, textvariable=self.status_var, anchor="w").grid(row=0, column=0, sticky="ew")
        self.update_label_option_state()

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=3)
        body.add(right, weight=2)

        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        canvas_frame = ttk.Frame(left)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(
            canvas_frame,
            background="#111111",
            highlightthickness=0,
            cursor="fleur",
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")
        canvas_yscroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.video_canvas.yview)
        canvas_xscroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.video_canvas.xview)
        canvas_yscroll.grid(row=0, column=1, sticky="ns")
        canvas_xscroll.grid(row=1, column=0, sticky="ew")
        self.video_canvas.configure(
            xscrollcommand=canvas_xscroll.set,
            yscrollcommand=canvas_yscroll.set,
        )
        self.video_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.video_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.video_canvas.bind("<MouseWheel>", self.on_canvas_mousewheel)

        hint = ttk.Label(
            left,
            text="快捷键：空格播放/暂停，← 上一帧，→ 下一帧，q 退出；左键拖动画面，滚轮上下移动，Shift+滚轮左右移动",
            anchor="center",
        )
        hint.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="当前叠加用 JSON 原数据", font=("", 11, "bold")).grid(row=0, column=0, sticky="w")

        self.json_text = tk.Text(right, wrap=tk.NONE)
        self.json_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        yscroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.json_text.yview)
        xscroll = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.json_text.xview)
        yscroll.grid(row=1, column=1, sticky="ns", pady=(6, 0))
        xscroll.grid(row=2, column=0, sticky="ew")
        self.json_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.prev_frame())
        self.root.bind("<Right>", lambda _event: self.next_frame())
        self.root.bind("q", lambda _event: self.root.destroy())

    def get_annotation(self) -> CameraAnnotation:
        annotation = self.store.cameras.get(self.current_camera)
        if annotation is None:
            raise KeyError(f"JSONL 中没有相机 {self.current_camera} 的数据")
        return annotation

    def current_display_pts(self) -> int | None:
        try:
            annotation = self.get_annotation()
        except KeyError:
            return None
        if not annotation.pts_values:
            return None
        index = max(0, min(self.current_index, len(annotation.pts_values) - 1))
        return annotation.pts_values[index]

    def update_progress_range(self) -> None:
        annotation = self.get_annotation()
        upper = max(1, len(annotation.items))
        self.progress_scale.configure(from_=1, to=upper)

    def on_progress_drag(self, _value: str) -> None:
        if self.is_updating_progress:
            return
        annotation = self.get_annotation()
        if not annotation.items:
            return
        frame_number = int(round(self.progress_var.get()))
        frame_number = max(1, min(frame_number, len(annotation.items)))
        self.frame_var.set(str(frame_number))
        self.pts_var.set(str(annotation.pts_values[frame_number - 1]))

    def apply_progress_seek(self) -> None:
        annotation = self.get_annotation()
        if not annotation.items:
            return
        self.is_playing = False
        frame_number = int(round(self.progress_var.get()))
        frame_number = max(1, min(frame_number, len(annotation.items)))
        self.progress_var.set(frame_number)
        self.seek_and_render(frame_number - 1)

    def open_camera(self, camera_id: int, target_pts: int | None = None) -> None:
        self.is_playing = False
        if camera_id not in self.store.cameras:
            messagebox.showerror("相机不存在", f"JSONL 中没有相机 {camera_id} 的数据")
            return

        self.current_camera = camera_id
        self.camera_var.set(str(camera_id))
        self.current_raw_frame = None

        if self.is_video_mode():
            playlist_path = self.video_dir / f"{camera_id}.m3u8"
            if not playlist_path.is_file():
                messagebox.showerror("视频不存在", f"找不到视频文件：{playlist_path}")
                return

            if not self.reopen_capture():
                return
        else:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.capture_next_index = 0

        annotation = self.get_annotation()
        self.update_progress_range()
        if target_pts is None:
            frame_index = max(0, min(self.current_index, len(annotation.items) - 1))
        else:
            frame_index = nearest_index_by_pts(annotation, target_pts)
        self.seek_and_render(frame_index)

    def is_video_mode(self) -> bool:
        return self.mode_var.get() == "视频模式"

    def change_mode(self) -> None:
        self.is_playing = False
        self.current_raw_frame = None
        current_pts = self.current_display_pts()
        self.open_camera(self.current_camera, target_pts=current_pts)

    def reopen_capture(self) -> bool:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        playlist_path = self.video_dir / f"{self.current_camera}.m3u8"
        self.capture = cv2.VideoCapture(str(playlist_path))
        self.capture_next_index = 0

        if not self.capture.isOpened():
            messagebox.showerror(
                "打开失败",
                f"OpenCV 无法打开视频：{playlist_path}\n"
                "如果 m3u8 无法读取，可以先确认 opencv-python 是否带 ffmpeg 支持。",
            )
            return False

        return True

    def change_camera(self) -> None:
        try:
            camera_id = int(self.camera_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "相机编号必须是整数")
            return
        current_pts = self.current_display_pts()
        self.open_camera(camera_id, target_pts=current_pts)

    def toggle_play(self) -> None:
        if not self.is_video_mode():
            # 图片模式下也允许自动翻页，只是本质上是连续切图。
            pass
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.last_tick_time = time.time()
            self.play_tick()

    def play_tick(self) -> None:
        if not self.is_playing:
            return

        self.step_frame(1)
        self.root.after(PLAY_INTERVAL_MS, self.play_tick)

    def prev_frame(self) -> None:
        self.is_playing = False
        self.seek_and_render(self.current_index - 1)

    def first_frame(self) -> None:
        self.is_playing = False
        self.seek_and_render(0)

    def next_frame(self) -> None:
        self.is_playing = False
        self.seek_and_render(self.current_index + 1)

    def jump_frame(self) -> None:
        self.is_playing = False
        try:
            frame_number = int(self.frame_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "帧序号必须是整数")
            return
        self.seek_and_render(frame_number - 1)

    def jump_pts(self) -> None:
        self.is_playing = False
        try:
            pts = int(self.pts_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "PTS 必须是整数")
            return
        annotation = self.get_annotation()
        self.seek_and_render(nearest_index_by_pts(annotation, pts))

    def apply_display_settings(self) -> None:
        try:
            width = int(self.display_width_var.get())
            height = int(self.display_height_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "显示宽和显示高必须是整数")
            return

        if width < 320 or height < 180:
            messagebox.showerror("输入错误", "显示宽高太小，建议至少 320x180")
            return
        if width > 3840 or height > 2160:
            messagebox.showerror("输入错误", "显示宽高太大，建议不要超过 3840x2160")
            return

        self.display_width = width
        self.display_height = height
        self.sync_video_scale_from_size(width, height)
        self.render_current()

    def on_video_scale_change(self, _value: str) -> None:
        scale = float(self.video_scale_var.get())
        width = max(320, int(round(DISPLAY_WIDTH * scale)))
        height = max(180, int(round(DISPLAY_HEIGHT * scale)))
        self.display_width = width
        self.display_height = height
        self.display_width_var.set(str(width))
        self.display_height_var.set(str(height))
        self.video_scale_text_var.set(f"{scale:.2f}")
        self.render_current()

    def sync_video_scale_from_size(self, width: int, height: int) -> None:
        width_scale = width / DISPLAY_WIDTH
        height_scale = height / DISPLAY_HEIGHT
        scale = min(max((width_scale + height_scale) / 2, 0.25), 1.2)
        self.video_scale_var.set(scale)
        self.video_scale_text_var.set(f"{scale:.2f}")

    def on_label_scale_change(self, _value: str) -> None:
        self.label_scale_text_var.set(f"{float(self.label_scale_var.get()):.2f}")
        self.render_current()

    def on_label_mode_change(self) -> None:
        self.update_label_option_state()
        self.render_current()

    def update_label_option_state(self) -> None:
        # 排错模式固定显示策略，复选框只在详细模式下生效。
        is_detail_mode = self.label_mode_var.get() == "详细模式"
        state = tk.NORMAL if is_detail_mode else tk.DISABLED
        for widget in self.label_option_widgets:
            widget.configure(state=state)
        if is_detail_mode:
            self.threshold_row.grid_remove()
        else:
            self.threshold_row.grid()

    def apply_threshold_settings(self) -> None:
        try:
            det_threshold = float(self.det_threshold_var.get())
            team_score_threshold = float(self.team_score_threshold_var.get())
            id_score_threshold = float(self.id_score_threshold_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "异常阈值必须是数字")
            return

        values = {
            "det": det_threshold,
            "team_s": team_score_threshold,
            "id_s": id_score_threshold,
        }
        for name, value in values.items():
            if value < 0 or value > 1:
                messagebox.showerror("输入错误", f"{name} 阈值必须在 0 到 1 之间")
                return

        self.current_det_threshold = det_threshold
        self.current_team_score_threshold = team_score_threshold
        self.current_id_score_threshold = id_score_threshold
        self.det_threshold_var.set(f"{det_threshold:.2f}")
        self.team_score_threshold_var.set(f"{team_score_threshold:.2f}")
        self.id_score_threshold_var.set(f"{id_score_threshold:.2f}")
        self.render_current()

    def apply_json_offset(self) -> None:
        try:
            offset = int(self.json_offset_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "JSON 偏移帧数必须是整数")
            return
        self.json_offset_frames = offset
        self.json_offset_var.set(str(offset))
        self.render_current()

    def step_json_offset(self, step: int) -> None:
        self.json_offset_frames += step
        self.json_offset_var.set(str(self.json_offset_frames))
        self.render_current()

    def reset_json_offset(self) -> None:
        self.json_offset_frames = 0
        self.json_offset_var.set("0")
        self.render_current()

    def step_frame(self, step: int) -> None:
        self.seek_and_render(self.current_index + step)

    def seek_and_render(self, frame_index: int) -> None:
        annotation = self.get_annotation()
        if not annotation.items:
            return

        frame_index = max(0, min(frame_index, len(annotation.items) - 1))

        frame = self.read_frame_data(frame_index)
        if frame is None:
            source_name = "视频帧" if self.is_video_mode() else "图片帧"
            self.status_var.set(f"读取{source_name}失败：camera={self.current_camera}, frame={frame_index + 1}")
            return

        self.current_index = frame_index
        self.current_raw_frame = frame
        self.render_frame(frame)

    def render_current(self) -> None:
        if self.current_raw_frame is not None:
            self.render_frame(self.current_raw_frame)
        else:
            self.seek_and_render(self.current_index)

    def read_video_frame(self, frame_index: int) -> Any | None:
        if self.capture is None:
            return None

        # HEVC/HLS 不能每帧都随机 seek；否则容易缺参考帧并出现灰屏。
        # 所以向前播放时连续 read，向后/随机跳转时重新打开并从头解码到目标帧。
        if frame_index < self.capture_next_index:
            if not self.reopen_capture():
                return None

        frame = None
        while self.capture_next_index <= frame_index:
            ok, frame = self.capture.read()
            if not ok:
                return None
            self.capture_next_index += 1

        return frame

    def read_image_frame(self, frame_index: int) -> Any | None:
        annotation = self.get_annotation()
        pts = annotation.pts_values[frame_index]
        image_path = self.frame_image_dir / str(self.current_camera) / f"{pts}.jpg"
        if not image_path.is_file():
            self.show_json(
                {
                    "error": "找不到图片帧",
                    "camera": self.current_camera,
                    "frame_index": frame_index,
                    "pts": pts,
                    "expected_path": str(image_path),
                }
            )
            return None
        return cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    def read_frame_data(self, frame_index: int) -> Any | None:
        if self.is_video_mode():
            if self.capture is None:
                return None
            return self.read_video_frame(frame_index)
        return self.read_image_frame(frame_index)

    def render_frame(self, frame: Any) -> None:
        annotation = self.get_annotation()
        json_index = max(0, min(self.current_index + self.json_offset_frames, len(annotation.items) - 1))
        item = annotation.items[json_index]
        pts = annotation.pts_values[self.current_index]
        json_pts = annotation.pts_values[json_index]

        frame_h, frame_w = frame.shape[:2]
        scale_x = self.display_width / frame_w
        scale_y = self.display_height / frame_h
        display_frame = cv2.resize(
            frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        label_scale = float(self.label_scale_var.get())

        draw_detections(
            display_frame,
            item,
            show_players=self.show_players.get(),
            show_balls=self.show_balls.get(),
            show_keypoints=self.show_keypoints.get(),
            scale_x=scale_x,
            scale_y=scale_y,
            label_scale=label_scale,
            label_mode=self.label_mode_var.get(),
            show_label_id=self.show_label_id.get(),
            show_label_team=self.show_label_team.get(),
            show_label_det=self.show_label_det.get(),
            show_label_team_score=self.show_label_team_score.get(),
            show_label_id_score=self.show_label_id_score.get(),
            det_threshold=self.current_det_threshold,
            team_score_threshold=self.current_team_score_threshold,
            id_score_threshold=self.current_id_score_threshold,
        )
        self.draw_overlay_header(display_frame, item, pts, json_index, json_pts, label_scale)
        self.show_frame(display_frame)
        self.show_json(item)

        result = item.get("result") or {}
        players = len(result.get("players") or [])
        balls = len(result.get("balls") or [])
        threshold_status = ""
        if self.label_mode_var.get() == "排错模式":
            threshold_status = (
                f"阈值 det={self.current_det_threshold:.2f}, "
                f"team_s={self.current_team_score_threshold:.2f}, "
                f"id_s={self.current_id_score_threshold:.2f} | "
            )
        self.is_updating_progress = True
        self.progress_var.set(self.current_index + 1)
        self.is_updating_progress = False
        self.frame_var.set(str(self.current_index + 1))
        self.pts_var.set(str(pts))
        self.status_var.set(
            f"{self.mode_var.get()} | "
            f"相机 {self.current_camera} | "
            f"帧 {self.current_index + 1}/{len(annotation.items)} | "
            f"图片 PTS {pts} | JSON偏移 {self.json_offset_frames:+d} | "
            f"JSON帧 {json_index + 1}/{len(annotation.items)} | JSON PTS {json_pts} | "
            f"players {players} | balls {balls} | "
            f"原始尺寸 {frame_w}x{frame_h} -> 显示尺寸 {self.display_width}x{self.display_height} | "
            f"标签 {self.label_mode_var.get()} | "
            f"{threshold_status}"
            f"文字 {label_scale:.2f} | "
            f"解码位置 {self.capture_next_index}"
        )

    def draw_overlay_header(
        self,
        frame: Any,
        item: dict[str, Any],
        pts: int,
        json_index: int,
        json_pts: int,
        label_scale: float,
    ) -> None:
        result = item.get("result") or {}
        text = (
            f"camera {self.current_camera}  "
            f"cam_idx {result.get('cam_idx')}  "
            f"frame {self.current_index + 1}  "
            f"pts {pts}  "
            f"json_offset {self.json_offset_frames:+d}  "
            f"json_frame {json_index + 1}  "
            f"json_pts {json_pts}  "
            f"players {len(result.get('players') or [])}  "
            f"balls {len(result.get('balls') or [])}"
        )
        draw_label(
            frame,
            text,
            18,
            max(28, int(38 * label_scale)),
            HEADER_TEXT_COLOR_BGR,
            font_scale=max(label_scale + 0.12, 0.35),
            thickness=max(1, int(round(label_scale * 3))),
        )

    def show_frame(self, frame: Any) -> None:
        xview = self.video_canvas.xview() if hasattr(self, "video_canvas") else (0.0, 1.0)
        yview = self.video_canvas.yview() if hasattr(self, "video_canvas") else (0.0, 1.0)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.current_frame_image = photo
        if self.canvas_image_id is None:
            self.canvas_image_id = self.video_canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            self.video_canvas.itemconfigure(self.canvas_image_id, image=photo)
        self.video_canvas.configure(scrollregion=(0, 0, frame.shape[1], frame.shape[0]))
        self.video_canvas.xview_moveto(xview[0])
        self.video_canvas.yview_moveto(yview[0])

    def on_canvas_press(self, event: tk.Event) -> None:
        self.video_canvas.scan_mark(event.x, event.y)

    def on_canvas_drag(self, event: tk.Event) -> None:
        self.video_canvas.scan_dragto(event.x, event.y, gain=1)

    def on_canvas_mousewheel(self, event: tk.Event) -> None:
        units = int(-1 * (event.delta / 120)) if event.delta else 0
        if units == 0:
            units = -1 if event.delta > 0 else 1
        if event.state & 0x0001:
            self.video_canvas.xview_scroll(units, "units")
        else:
            self.video_canvas.yview_scroll(units, "units")

    def show_json(self, item: dict[str, Any]) -> None:
        self.json_text.configure(state=tk.NORMAL)
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(item, ensure_ascii=False, indent=2))
        self.json_text.configure(state=tk.DISABLED)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.root.destroy()


def main() -> None:
    args = parse_args()
    try:
        require_dependencies()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

    video_dir = args.video_dir.expanduser().resolve()
    frame_image_dir = args.frame_image_dir.expanduser().resolve()
    jsonl_path = args.jsonl.expanduser().resolve()
    if not video_dir.is_dir():
        print(f"提示：视频目录不存在，视频模式暂不可用：{video_dir}")
    if not frame_image_dir.is_dir():
        print(f"提示：图片帧目录不存在，图片模式暂不可用：{frame_image_dir}")

    store = load_annotations(jsonl_path)
    if args.camera not in store.cameras:
        available = ", ".join(str(camera_id) for camera_id in sorted(store.cameras))
        raise ValueError(f"JSONL 中没有相机 {args.camera}，可用相机：{available}")

    root = tk.Tk()
    app = OpenCvJsonlViewer(root, video_dir, frame_image_dir, store, args.camera)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
