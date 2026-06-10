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
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import av
except ImportError:
    av = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


# =========================
# 可修改下面这些参数
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# 输入视频目录：里面应包含 101.m3u8 ... 124.m3u8 或 101.ts ... 124.ts。
VIDEO_DIR = PROJECT_DIR / "videos" / "match_1_clip"

# 2D pipeline 输出的 JSONL。
JSONL_PATH = "https://sense-omni.tos-cn-shanghai.volces.com/hocky/lym/data_saved/0526/output/jsonl_without_players/20260526_164647/output_0_no_players.jsonl"

# 按 pts 命名的 JPEG 帧目录。
FRAME_IMAGE_DIR = "https://sense-omni.tos-cn-shanghai.volces.com/hocky/lym/data_saved/0526/frame_clips/20260526_164647/"

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
DEFAULT_FRAME_STEP = 1

# 视频模式关键帧 seek 缓存。4K BGR 一帧约 24MB，不宜缓存太多。
VIDEO_CACHE_BEFORE = 3
VIDEO_CACHE_AFTER = 36
VIDEO_CACHE_MAX_FRAMES = 240
VIDEO_CACHE_KEEP_BEHIND = 6
VIDEO_PREFETCH_AFTER = 90
VIDEO_PREFETCH_TARGET_AHEAD = 180
VIDEO_RESULT_POLL_MS = 30

# 网络图片模式预加载范围。只对 tos/http 图片目录生效。
IMAGE_PREFETCH_BEFORE = 12
IMAGE_PREFETCH_AFTER = 70
IMAGE_CACHE_MAX_FRAMES = 300
IMAGE_CONCURRENT_DOWNLOADS = 16

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
MANUAL_BALL_COLOR_BGR = (255, 0, 255)
KEYPOINT_COLOR_BGR = (120, 255, 120)
HEADER_TEXT_COLOR_BGR = (255, 255, 255)
MANUAL_BALL_SOURCE = "manual"
MANUAL_BALL_SCORE = 2.0
ANNOTATION_WORK_SUFFIX = "_ball_annotation"
MIN_ANNOTATION_BOX_SIZE = 3
ANNOTATION_HIT_TOLERANCE = 6
ANNOTATION_SAVE_DEBOUNCE_MS = 2000
WORK_MODE_ANNOTATION = "标注模式"
WORK_MODE_INSPECT = "检查模式"
WORK_MODE_VIEW = "观看模式"
INPUT_MODE_VIDEO = "视频模式"
INPUT_MODE_IMAGE = "图片模式"


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


@dataclass
class VideoKeyframeIndex:
    camera_id: int
    keyframe_indices: list[int] = field(default_factory=list)
    keyframe_pts_values: list[int] = field(default_factory=list)


class PyAvVideoFrameReader:
    """用 PyAV 持久打开一路视频，并从最近关键帧解码到目标帧附近。"""

    def __init__(
        self,
        playlist_path: Path,
        annotation: CameraAnnotation,
        keyframe_index: VideoKeyframeIndex,
    ):
        self.playlist_path = playlist_path
        self.annotation = annotation
        self.keyframe_index = keyframe_index
        self.container: Any | None = None
        self.stream: Any | None = None

    def open(self) -> None:
        if av is None:
            raise RuntimeError("缺少 PyAV 依赖，请先安装：pip install av")

        self.close()
        self.container = av.open(str(self.playlist_path))
        self.stream = self.container.streams.video[0]

        try:
            self.stream.thread_type = "AUTO"
        except Exception:
            try:
                self.stream.codec_context.thread_type = "AUTO"
            except Exception:
                pass

    def close(self) -> None:
        if self.container is not None:
            self.container.close()
            self.container = None
        self.stream = None

    def keyframe_start_index_for_target(self, frame_index: int) -> int:
        pos = bisect.bisect_right(self.keyframe_index.keyframe_indices, frame_index) - 1
        if pos < 0:
            return 0
        return self.keyframe_index.keyframe_indices[pos]

    def seek_to_keyframe(self, keyframe_index: int) -> None:
        if self.container is None or self.stream is None:
            self.open()

        if keyframe_index <= 0:
            self.open()
            return

        key_pts = self.annotation.pts_values[keyframe_index]
        try:
            self.container.seek(
                key_pts, stream=self.stream, backward=True, any_frame=False
            )
        except Exception:
            self.open()
            self.container.seek(
                key_pts, stream=self.stream, backward=True, any_frame=False
            )

    def decode_cache_window(
        self,
        target_frame_index: int,
        before: int,
        after: int,
        max_frames: int,
        step: int = 1,
    ) -> dict[int, Any]:
        if not self.annotation.pts_values:
            return {}

        step = max(1, int(step))
        keyframe_index = self.keyframe_start_index_for_target(target_frame_index)
        desired_start = max(keyframe_index, target_frame_index - before)
        desired_end = min(
            len(self.annotation.pts_values) - 1, target_frame_index + after
        )

        self.seek_to_keyframe(keyframe_index)
        cache: dict[int, Any] = {}

        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue

            pts = int(frame.pts)
            frame_index = self.annotation.pts_to_index.get(pts)
            if frame_index is None:
                frame_index = nearest_index_by_pts(self.annotation, pts)

            if frame_index < keyframe_index:
                continue
            if frame_index > desired_end:
                break

            if (
                desired_start <= frame_index <= desired_end
                and (frame_index - target_frame_index) % step == 0
            ):
                cache[frame_index] = frame.to_ndarray(format="bgr24")
                if len(cache) >= max_frames:
                    break

        return cache


REMOTE_SCHEMES = ("http://", "https://", "tos://")
DEFAULT_TOS_HTTP_TEMPLATE = "https://{bucket}.tos-cn-beijing.volces.com/{key}"


def resource_to_text(resource: Any) -> str:
    return str(resource).strip()


def is_remote_resource(resource: Any) -> bool:
    text = resource_to_text(resource).lower()
    return text.startswith(REMOTE_SCHEMES)


def is_tos_resource(resource: Any) -> bool:
    return resource_to_text(resource).lower().startswith("tos://")


def display_resource_name(resource: Any) -> str:
    text = resource_to_text(resource)
    if is_remote_resource(text):
        parsed = urlparse(text)
        name = Path(parsed.path.rstrip("/")).name
        return name or parsed.netloc or text
    return Path(text).name


def local_path_from_resource(resource: Any) -> Path:
    return Path(resource_to_text(resource)).expanduser()


def normalize_local_or_remote_resource(
    resource: Any, *, must_be_file: bool = False
) -> Any:
    text = resource_to_text(resource)
    if is_remote_resource(text):
        return text.rstrip("/")
    path = local_path_from_resource(text)
    return path.resolve() if must_be_file else path.resolve()


def tos_to_http_url(tos_uri: str) -> str:
    parsed = urlparse(tos_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"TOS 地址格式不正确：{tos_uri}")

    quoted_key = quote(key, safe="/")
    template = os.environ.get("TOS_HTTP_TEMPLATE") or DEFAULT_TOS_HTTP_TEMPLATE
    return template.format(bucket=bucket, key=quoted_key, path=quoted_key)


def resource_to_request_url(resource: Any) -> str:
    text = resource_to_text(resource)
    if is_tos_resource(text):
        return tos_to_http_url(text)
    return text


def join_resource(base: Any, *parts: str) -> Any:
    text = resource_to_text(base)
    if is_remote_resource(text):
        suffix = "/".join(quote(str(part).strip("/"), safe="") for part in parts)
        return f"{text.rstrip('/')}/{suffix}"
    path = local_path_from_resource(text)
    for part in parts:
        path = path / part
    return path


def open_remote_resource(resource: Any):
    url = resource_to_request_url(resource)
    request = Request(url, headers={"User-Agent": "opencv-jsonl-viewer/1.0"})
    return urlopen(request, timeout=60)


def read_remote_bytes(resource: Any) -> bytes:
    with open_remote_resource(resource) as response:
        return response.read()


def decode_image_bytes(image_bytes: bytes) -> Any | None:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


class ResourceDirectoryDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        initial_value: str,
        initialdir: Path,
    ):
        self.initial_value = initial_value
        self.initialdir = initialdir
        self.entry: ttk.Entry | None = None
        self.result: str | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Widget) -> tk.Widget:
        ttk.Label(
            master,
            text="输入本地目录或 tos/http URL。网络地址直接输入后点 OK。",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        self.entry = ttk.Entry(master, width=86)
        self.entry.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.entry.insert(0, self.initial_value)
        master.columnconfigure(0, weight=1)
        return self.entry

    def buttonbox(self) -> None:
        box = ttk.Frame(self)
        ttk.Button(box, text="OK", width=10, command=self.ok).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(box, text="选择本地", width=12, command=self.choose_local).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(box, text="取消", width=10, command=self.cancel).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def choose_local(self) -> None:
        selected = filedialog.askdirectory(
            title=self.title() or "选择本地目录",
            initialdir=str(self.initialdir),
            parent=self,
        )
        if not selected or self.entry is None:
            return
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(Path(selected).expanduser().resolve()))

    def validate(self) -> bool:
        if self.entry is None:
            return False
        value = self.entry.get().strip()
        if not value:
            messagebox.showerror("输入错误", "目录不能为空", parent=self)
            return False
        return True

    def apply(self) -> None:
        if self.entry is not None:
            self.result = self.entry.get().strip()


class ResourceFileDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        initial_value: str,
        initialdir: Path,
    ):
        self.initial_value = initial_value
        self.initialdir = initialdir
        self.entry: ttk.Entry | None = None
        self.result: str | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Widget) -> tk.Widget:
        ttk.Label(
            master,
            text="输入本地 JSONL 文件路径或 tos/http URL。网络地址直接输入后点 OK。",
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        self.entry = ttk.Entry(master, width=86)
        self.entry.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.entry.insert(0, self.initial_value)
        master.columnconfigure(0, weight=1)
        return self.entry

    def buttonbox(self) -> None:
        box = ttk.Frame(self)
        ttk.Button(box, text="OK", width=10, command=self.ok).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(box, text="选择本地", width=12, command=self.choose_local).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(box, text="取消", width=10, command=self.cancel).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def choose_local(self) -> None:
        selected = filedialog.askopenfilename(
            title=self.title() or "选择 JSONL 文件",
            initialdir=str(self.initialdir),
            filetypes=[
                ("JSONL 文件", "*.jsonl"),
                ("JSON 文件", "*.json"),
                ("所有文件", "*.*"),
            ],
            parent=self,
        )
        if not selected or self.entry is None:
            return
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(Path(selected).expanduser().resolve()))

    def validate(self) -> bool:
        if self.entry is None:
            return False
        value = self.entry.get().strip()
        if not value:
            messagebox.showerror("输入错误", "JSONL 路径不能为空", parent=self)
            return False
        return True

    def apply(self) -> None:
        if self.entry is not None:
            self.result = self.entry.get().strip()


class AnnotationWorkPathDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        initial_value: str,
        initialdir: Path,
    ):
        self.initial_value = initial_value
        self.initialdir = initialdir
        self.entry: ttk.Entry | None = None
        self.result: str | None = None
        super().__init__(parent, title)

    def body(self, master: tk.Widget) -> tk.Widget:
        ttk.Label(
            master,
            text=(
                "输入标注 JSONL 文件路径可继续使用；输入或选择文件夹则会在该文件夹内"
                "自动生成一份标注 JSONL。确认后才会开始写入。"
            ),
            anchor="w",
            wraplength=720,
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        self.entry = ttk.Entry(master, width=92)
        self.entry.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.entry.insert(0, self.initial_value)
        master.columnconfigure(0, weight=1)
        return self.entry

    def buttonbox(self) -> None:
        box = ttk.Frame(self)
        ttk.Button(box, text="OK", width=10, command=self.ok).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(box, text="选择文件", width=12, command=self.choose_file).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(
            box, text="选择文件夹", width=12, command=self.choose_directory
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(box, text="取消", width=10, command=self.cancel).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def choose_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="选择已有标注 JSONL",
            initialdir=str(self.initialdir),
            filetypes=[("JSONL 文件", "*.jsonl"), ("所有文件", "*.*")],
            parent=self,
        )
        if not selected or self.entry is None:
            return
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(Path(selected).expanduser().resolve()))

    def choose_directory(self) -> None:
        selected = filedialog.askdirectory(
            title="选择标注 JSONL 保存文件夹",
            initialdir=str(self.initialdir),
            parent=self,
        )
        if not selected or self.entry is None:
            return
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(Path(selected).expanduser().resolve()))

    def validate(self) -> bool:
        if self.entry is None:
            return False
        value = self.entry.get().strip()
        if not value:
            messagebox.showerror("输入错误", "标注路径不能为空", parent=self)
            return False
        return True

    def apply(self) -> None:
        if self.entry is not None:
            self.result = self.entry.get().strip()


class JsonlLoadCancelled(RuntimeError):
    pass


def require_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if av is None:
        missing.append("av")
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
        type=str,
        default=str(JSONL_PATH),
        help=f"JSONL 文件，默认：{JSONL_PATH}",
    )
    parser.add_argument(
        "--frame-image-dir",
        type=str,
        default=str(FRAME_IMAGE_DIR),
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
    if filename.endswith(".m3u8") or filename.endswith(".ts"):
        stem = filename.rsplit(".", 1)[0]
        if stem.isdigit():
            return int(stem)

    result = item.get("result")
    cam_idx = result.get("cam_idx") if isinstance(result, dict) else None
    if isinstance(cam_idx, int):
        return cam_idx + 101
    return None


def load_annotations(
    jsonl_path: Any,
    progress_callback: Any | None = None,
    cancel_event: threading.Event | None = None,
) -> AnnotationStore:
    jsonl_source = normalize_local_or_remote_resource(jsonl_path, must_be_file=True)
    if not is_remote_resource(jsonl_source) and not Path(jsonl_source).is_file():
        raise FileNotFoundError(f"找不到 JSONL 文件：{jsonl_source}")

    file_size = (
        Path(jsonl_source).stat().st_size if not is_remote_resource(jsonl_source) else 0
    )
    store = AnnotationStore()
    total_lines = 0
    bad_lines = 0
    player_count = 0
    ball_count = 0
    manual_ball_found = False
    camera_line_counts: dict[int, int] = defaultdict(int)
    bytes_read = 0
    last_report = 0

    print(f"正在读取 JSONL：{jsonl_source}")

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise JsonlLoadCancelled("JSONL 加载已取消")

    def handle_line(line: str) -> None:
        nonlocal total_lines, bad_lines, player_count, ball_count
        nonlocal manual_ball_found
        nonlocal bytes_read, last_report

        check_cancelled()
        line = line.strip()
        if not line:
            return

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            return
        if not isinstance(item, dict):
            bad_lines += 1
            return

        camera_id = camera_id_from_item(item)
        pts = item.get("pts")
        if camera_id is None or pts is None:
            bad_lines += 1
            return

        total_lines += 1
        camera_annotation = store.cameras.setdefault(
            camera_id,
            CameraAnnotation(camera_id=camera_id),
        )
        camera_annotation.items.append(item)
        camera_annotation.pts_values.append(int(pts))
        camera_annotation.pts_to_index[int(pts)] = len(camera_annotation.items) - 1
        camera_line_counts[camera_id] += 1

        result, _result_missing = result_dict_from_item(item)
        players, _players_state = detection_list_from_result(result, "players")
        balls, _balls_state = detection_list_from_result(result, "balls")
        player_count += len(players)
        ball_count += len(balls)
        if not manual_ball_found:
            for ball in balls:
                if is_manual_ball(ball):
                    manual_ball_found = True
                    break

        if progress_callback and total_lines - last_report >= 500:
            last_report = total_lines
            pct = min(99, int(bytes_read * 100 / max(1, file_size))) if file_size else 0
            progress_callback(pct, total_lines, len(store.cameras))

    if is_remote_resource(jsonl_source):
        with open_remote_resource(jsonl_source) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                file_size = int(content_length)
            for raw_line in response:
                check_cancelled()
                bytes_read += len(raw_line)
                handle_line(raw_line.decode("utf-8"))
    else:
        with Path(jsonl_source).open("r", encoding="utf-8") as file:
            for line in file:
                check_cancelled()
                bytes_read += len(line.encode("utf-8"))
                handle_line(line)

    check_cancelled()
    if progress_callback:
        progress_callback(99, total_lines, len(store.cameras))

    for camera_annotation in store.cameras.values():
        order = sorted(
            range(len(camera_annotation.items)),
            key=lambda idx: camera_annotation.pts_values[idx],
        )
        camera_annotation.items = [camera_annotation.items[idx] for idx in order]
        camera_annotation.pts_values = [
            camera_annotation.pts_values[idx] for idx in order
        ]
        camera_annotation.pts_to_index = {
            pts: idx for idx, pts in enumerate(camera_annotation.pts_values)
        }

    store.summary = {
        "jsonl_path": str(jsonl_source),
        "total_lines": total_lines,
        "bad_lines": bad_lines,
        "camera_count": len(store.cameras),
        "camera_line_counts": dict(sorted(camera_line_counts.items())),
        "player_count": player_count,
        "ball_count": ball_count,
        "has_manual_balls": manual_ball_found,
    }
    print(f"JSONL 读取完成：{total_lines} 行，{len(store.cameras)} 路相机。")
    if progress_callback:
        progress_callback(100, total_lines, len(store.cameras))
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


def build_video_keyframe_index(
    playlist_path: Path,
    annotation: CameraAnnotation,
) -> VideoKeyframeIndex:
    if av is None:
        raise RuntimeError("缺少 PyAV 依赖，请先安装：pip install av")

    keyframe_indices: list[int] = []
    keyframe_pts_values: list[int] = []
    seen: set[int] = set()

    container = av.open(str(playlist_path))
    try:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = int(frame.pts)
            if pts in seen:
                continue
            seen.add(pts)
            index = nearest_index_by_pts(annotation, pts)
            keyframe_indices.append(index)
            keyframe_pts_values.append(annotation.pts_values[index])
    finally:
        container.close()

    if not keyframe_indices:
        keyframe_indices = [0]
        keyframe_pts_values = [annotation.pts_values[0]]

    order = sorted(range(len(keyframe_indices)), key=lambda idx: keyframe_indices[idx])
    keyframe_indices = [keyframe_indices[idx] for idx in order]
    keyframe_pts_values = [keyframe_pts_values[idx] for idx in order]
    return VideoKeyframeIndex(
        camera_id=annotation.camera_id,
        keyframe_indices=keyframe_indices,
        keyframe_pts_values=keyframe_pts_values,
    )


MISSING_TEXT = "-"


def result_dict_from_item(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    result = item.get("result")
    if isinstance(result, dict):
        return result, False
    return {}, True


def detection_list_from_result(
    result: dict[str, Any],
    field_name: str,
) -> tuple[list[dict[str, Any]], str]:
    if field_name not in result:
        return [], "missing"
    value = result.get(field_name)
    if value is None:
        return [], "missing"
    if not isinstance(value, list):
        return [], "invalid"
    return [item for item in value if isinstance(item, dict)], "ok"


def detection_count_text(result: dict[str, Any], field_name: str, label: str) -> str:
    items, state = detection_list_from_result(result, field_name)
    if state == "missing":
        return f"{label} 缺失"
    if state == "invalid":
        return f"{label} 格式异常"
    return f"{label} {len(items)}"


def field_text(data: dict[str, Any], field_name: str) -> str:
    if field_name not in data:
        return MISSING_TEXT
    value = data.get(field_name)
    if value is None:
        return "null"
    return str(value)


def score_text(data: dict[str, Any], field_name: str) -> str:
    if field_name not in data or data.get(field_name) is None:
        return MISSING_TEXT
    try:
        return f"{float(data[field_name]):.2f}"
    except (TypeError, ValueError):
        return MISSING_TEXT


def scaled_bbox(
    bbox: Any,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        return (
            int(round(float(bbox[0]) * scale_x)),
            int(round(float(bbox[1]) * scale_y)),
            int(round(float(bbox[2]) * scale_x)),
            int(round(float(bbox[3]) * scale_y)),
        )
    except (TypeError, ValueError):
        return None


def player_id_text(player: dict[str, Any]) -> str:
    if "player_id" not in player:
        return MISSING_TEXT
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
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
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
    rect_width, rect_height, baseline_offset = label_metrics(
        text, font_scale, thickness
    )
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
    cv2.putText(
        frame, text, (x + 3, y), font, font_scale, color, thickness, cv2.LINE_AA
    )
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
    return label_rect_from_top_left(
        frame, text, x, y - baseline_offset, font_scale, thickness
    )[2]


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def expanded_rect(
    rect: tuple[int, int, int, int], padding: int
) -> tuple[int, int, int, int]:
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
        label_x, label_y, rect = label_rect_from_top_left(
            frame, text, left, top, font_scale, thickness
        )
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


def is_manual_ball(ball: dict[str, Any]) -> bool:
    if ball.get("source") == MANUAL_BALL_SOURCE:
        return True
    try:
        return float(ball.get("score")) == MANUAL_BALL_SCORE
    except (TypeError, ValueError):
        return False


def ball_signature(ball: dict[str, Any]) -> str:
    return json.dumps(ball, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
            f"{field_text(player, 'reid_team')} "
            f"det:{score_text(player, 'score')} "
            f"team_s:{score_text(player, 'reid_score')} "
            f"id_s:{score_text(player, 'player_id_score')}"
        )

    parts: list[str] = []
    if show_id:
        parts.append(f"id:{player_id}")
    if show_team:
        parts.append(field_text(player, "reid_team"))
    if show_det:
        parts.append(f"det:{score_text(player, 'score')}")
    if show_team_score:
        parts.append(f"team_s:{score_text(player, 'reid_score')}")
    if show_id_score:
        parts.append(f"id_s:{score_text(player, 'player_id_score')}")
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
    deleted_ball_indices: set[int] | None = None,
    show_manual_balls: bool = False,
) -> None:
    result, _result_missing = result_dict_from_item(item)
    box_thickness = max(1, int(round(label_scale * 3)))
    point_radius = max(1, int(round(label_scale * 3)))
    label_thickness = max(1, int(round(label_scale * 3)))
    header_reserved_height = max(34, int(round(56 * label_scale)))
    occupied_labels: list[tuple[int, int, int, int]] = [
        (0, 0, int(frame.shape[1] * 0.72), header_reserved_height)
    ]

    if show_players:
        players, _players_state = detection_list_from_result(result, "players")
        for player in players:
            bbox_rect = scaled_bbox(player.get("bbox"), scale_x, scale_y)
            if bbox_rect is None:
                continue

            x1, y1, x2, y2 = bbox_rect
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
                rect = text_rect(
                    frame, label, label_x, label_y, label_scale, label_thickness
                )
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
                keypoints = player.get("2d_keypoints")
                if not isinstance(keypoints, list):
                    keypoints = []
                for point in keypoints:
                    if not isinstance(point, list) or len(point) < 2:
                        continue
                    px = int(round(float(point[0]) * scale_x))
                    py = int(round(float(point[1]) * scale_y))
                    cv2.circle(frame, (px, py), point_radius, KEYPOINT_COLOR_BGR, -1)

    if show_balls:
        balls, _balls_state = detection_list_from_result(result, "balls")
        deleted_set = deleted_ball_indices or set()
        original_ball_index = -1
        for ball in balls:
            if is_manual_ball(ball):
                continue
            original_ball_index += 1
            if original_ball_index in deleted_set:
                continue
            bbox_rect = scaled_bbox(ball.get("bbox"), scale_x, scale_y)
            if bbox_rect is None:
                continue

            x1, y1, x2, y2 = bbox_rect
            ball_color = (0, 0, 255) if safe_score(ball.get("score")) <= 0 else BALL_COLOR_BGR
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), ball_color, max(2, box_thickness + 1)
            )
            label = f"ball {score_text(ball, 'score')}"
            label_x, label_y = find_label_position(
                frame,
                label,
                (x1, y1, x2, y2),
                occupied_labels,
                label_scale,
                label_thickness,
            )
            rect = text_rect(
                frame, label, label_x, label_y, label_scale, label_thickness
            )
            draw_leader_line(
                frame,
                (x1, y1, x2, y2),
                rect,
                ball_color,
                max(1, label_thickness),
            )
            rect = draw_label(
                frame,
                label,
                label_x,
                label_y,
                ball_color,
                font_scale=label_scale,
                thickness=label_thickness,
            )
            occupied_labels.append(rect)

    if show_manual_balls:
        balls, _balls_state = detection_list_from_result(result, "balls")
        manual_ball_index = 0
        for ball in balls:
            if not is_manual_ball(ball):
                continue
            manual_ball_index += 1
            bbox_rect = scaled_bbox(ball.get("bbox"), scale_x, scale_y)
            if bbox_rect is None:
                continue

            x1, y1, x2, y2 = bbox_rect
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), MANUAL_BALL_COLOR_BGR, max(2, box_thickness + 1)
            )
            label = f"m_ball {manual_ball_index} {score_text(ball, 'score')}"
            label_x, label_y = find_label_position(
                frame,
                label,
                (x1, y1, x2, y2),
                occupied_labels,
                label_scale,
                label_thickness,
            )
            rect = text_rect(
                frame, label, label_x, label_y, label_scale, label_thickness
            )
            draw_leader_line(
                frame,
                (x1, y1, x2, y2),
                rect,
                MANUAL_BALL_COLOR_BGR,
                max(1, label_thickness),
            )
            rect = draw_label(
                frame,
                label,
                label_x,
                label_y,
                MANUAL_BALL_COLOR_BGR,
                font_scale=label_scale,
                thickness=label_thickness,
            )
            occupied_labels.append(rect)


def draw_manual_ball_annotations(
    frame: Any,
    balls: list[dict[str, Any]],
    scale_x: float,
    scale_y: float,
    label_scale: float,
) -> None:
    if not balls:
        return

    box_thickness = max(2, int(round(label_scale * 4)))
    label_thickness = max(1, int(round(label_scale * 3)))
    header_reserved_height = max(34, int(round(56 * label_scale)))
    occupied_labels: list[tuple[int, int, int, int]] = [
        (0, 0, int(frame.shape[1] * 0.72), header_reserved_height)
    ]

    for index, ball in enumerate(balls, start=1):
        bbox_rect = scaled_bbox(ball.get("bbox"), scale_x, scale_y)
        if bbox_rect is None:
            continue

        x1, y1, x2, y2 = bbox_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), MANUAL_BALL_COLOR_BGR, box_thickness)
        label = f"manual ball {index}"
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
            MANUAL_BALL_COLOR_BGR,
            max(1, label_thickness),
        )
        rect = draw_label(
            frame,
            label,
            label_x,
            label_y,
            MANUAL_BALL_COLOR_BGR,
            font_scale=label_scale,
            thickness=label_thickness,
        )
        occupied_labels.append(rect)


class OpenCvJsonlViewer:
    def __init__(
        self,
        root: tk.Tk,
        video_dir: Path,
        frame_image_dir: Any,
        store: AnnotationStore,
        initial_camera: int,
    ):
        self.root = root
        self.video_dir = video_dir
        self.frame_image_dir = frame_image_dir
        self.store = store
        self.current_camera = initial_camera
        self.current_index = 0
        self.current_raw_frame = None
        self.current_frame_image = None
        self.canvas_image_id: int | None = None
        self.camera_box: ttk.Combobox | None = None
        self.choose_media_dir_button: ttk.Button | None = None
        self.cancel_jsonl_load_button: ttk.Button | None = None
        self.json_sidebar_button: ttk.Button | None = None
        self.input_mode_box: ttk.Combobox | None = None
        self.annotation_mode_button: ttk.Button | None = None
        self.annotation_path_button: ttk.Button | None = None
        self.save_annotation_button: ttk.Button | None = None
        self.undo_annotation_button: ttk.Button | None = None
        self.clear_annotation_button: ttk.Button | None = None
        self.restore_original_ball_button: ttk.Button | None = None
        self.annotation_save_mode_box: ttk.Combobox | None = None
        self.annotation_save_mode_label: ttk.Label | None = None
        self.common_row: ttk.Frame | None = None
        self.resource_row: ttk.Frame | None = None
        self.annotation_row: ttk.Frame | None = None
        self.inspect_row: ttk.Frame | None = None
        self.display_row: ttk.Frame | None = None
        self.body_pane: ttk.PanedWindow | None = None
        self.right_panel: ttk.Frame | None = None
        self.json_sidebar_visible = False
        self.json_title_label: ttk.Label | None = None
        self.annotation_json_label: ttk.Label | None = None
        self.annotation_json_separator: tk.Frame | None = None
        self.annotation_json_text: tk.Text | None = None
        self.annotation_json_widgets: list[tk.Widget] = []
        self.current_jsonl_path = resource_to_text(
            self.store.summary.get("jsonl_path", JSONL_PATH)
        )
        self.jsonl_load_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.jsonl_loading = False
        self.jsonl_load_token = 0
        self.jsonl_cancel_event: threading.Event | None = None
        self.video_keyframe_indexes: dict[int, VideoKeyframeIndex] = {}
        self.video_readers: dict[int, PyAvVideoFrameReader] = {}
        self.video_frame_cache: dict[int, Any] = {}
        self.video_cache_start = 0
        self.video_cache_end = -1
        self.video_task_token = 0
        self.video_loading = False
        self.video_loading_camera: int | None = None
        self.video_loading_frame_index: int | None = None
        self.video_prefetch_token = 0
        self.video_prefetching = False
        self.video_prefetch_camera: int | None = None
        self.video_prefetch_start_index: int | None = None
        self.video_decode_lock = threading.Lock()
        self.video_result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.video_cache_max_frames = VIDEO_CACHE_MAX_FRAMES
        self.video_prefetch_after = VIDEO_PREFETCH_AFTER
        self.video_prefetch_target_ahead = VIDEO_PREFETCH_TARGET_AHEAD
        self.image_frame_cache: dict[tuple[int, int], Any] = {}
        self.image_prefetch_pending: set[tuple[int, int]] = set()
        self.image_prefetch_token = 0
        self.image_prefetching = False
        self.image_prefetch_camera: int | None = None
        self.image_prefetch_center_index: int | None = None
        self.image_prefetch_target_index: int | None = None
        self.image_pending_render_index: int | None = None
        self.image_prefetch_before = IMAGE_PREFETCH_BEFORE
        self.image_prefetch_after = IMAGE_PREFETCH_AFTER
        self.image_cache_max_frames = IMAGE_CACHE_MAX_FRAMES
        self.image_prefetch_widgets: list[tk.Widget] = []
        self.manual_ball_annotations: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self.deleted_original_ball_indices: dict[tuple[int, int], set[int]] = {}
        self.deleted_original_ball_history: dict[tuple[int, int], list[tuple[int, float]]] = {}
        self.annotation_output_lines: list[str] = []
        self.annotation_key_to_output_indices: dict[tuple[int, int], list[int]] = {}
        self.annotation_output_entries: list[tuple[int, dict[str, Any]]] = []
        self.annotation_mode_enabled = False
        self.annotation_work_path: Path | None = None
        self.annotation_loaded_path: Path | None = None
        self.annotation_io_token = 0
        self.annotation_io_running = False
        self.annotation_dirty = False
        self.annotation_dirty_version = 0
        self.annotation_last_saved_version = 0
        self.annotation_save_after_id: str | None = None
        self.annotation_saving = False
        self.close_after_annotation_save = False
        self.pending_after_annotation_save: tuple[str, tuple[Any, ...]] | None = None
        self.annotation_write_lock = threading.Lock()
        self.annotation_save_error: str | None = None
        self.annotation_drag_start: tuple[float, float] | None = None
        self.annotation_preview_rect_id: int | None = None
        self.annotation_pan_mode: bool = False
        self.json_offset_frames = 0
        self.is_updating_progress = False
        self.is_playing = False
        self.last_tick_time = time.time()
        self.display_width = DISPLAY_WIDTH
        self.display_height = DISPLAY_HEIGHT

        self.camera_var = tk.StringVar(value=str(initial_camera))
        self.work_mode_var = tk.StringVar(value=WORK_MODE_ANNOTATION)
        self.mode_var = tk.StringVar(value=INPUT_MODE_IMAGE)
        self.pts_var = tk.StringVar(value="")
        self.frame_var = tk.StringVar(value="1")
        self.frame_step_var = tk.StringVar(value=str(DEFAULT_FRAME_STEP))
        self.progress_var = tk.DoubleVar(value=1)
        self.json_offset_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="")
        self.cache_status_var = tk.StringVar(value="缓冲：未启用")
        self.annotation_save_mode_var = tk.StringVar(value="自动保存")
        self.cache_target_var = tk.StringVar(value=str(VIDEO_PREFETCH_TARGET_AHEAD))
        self.image_prefetch_before_var = tk.StringVar(value=str(IMAGE_PREFETCH_BEFORE))
        self.image_prefetch_after_var = tk.StringVar(value=str(IMAGE_PREFETCH_AFTER))
        self.image_cache_max_var = tk.StringVar(value=str(IMAGE_CACHE_MAX_FRAMES))
        self.display_width_var = tk.StringVar(value=str(DISPLAY_WIDTH))
        self.display_height_var = tk.StringVar(value=str(DISPLAY_HEIGHT))
        self.video_scale_var = tk.DoubleVar(value=DEFAULT_VIDEO_SCALE)
        self.video_scale_text_var = tk.StringVar(value=f"{DEFAULT_VIDEO_SCALE:.2f}")
        self.show_players = tk.BooleanVar(value=True)
        self.show_balls = tk.BooleanVar(value=True)
        self.show_keypoints = tk.BooleanVar(value=False)
        self.show_manual_balls = tk.BooleanVar(value=False)
        self.manual_balls_checkbox: ttk.Checkbutton | None = None
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
        self.team_score_threshold_var = tk.StringVar(
            value=f"{ABNORMAL_TEAM_SCORE_THRESHOLD:.2f}"
        )
        self.id_score_threshold_var = tk.StringVar(
            value=f"{ABNORMAL_ID_SCORE_THRESHOLD:.2f}"
        )

        self.root.title("OpenCV JSONL 检测结果查看器")
        self.root.geometry("1500x900")
        self.root.minsize(1100, 700)

        self.build_layout()
        if self.store.cameras:
            self.open_camera(initial_camera)
        else:
            self.status_var.set(
                "尚未加载数据，请点击「加载预设资源」或手动选择 JSONL 文件"
            )
        self.root.after(VIDEO_RESULT_POLL_MS, self.process_video_results)
        self.root.after(100, self.process_jsonl_load_results)

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)

        common_row = ttk.Frame(toolbar)
        common_row.grid(row=0, column=0, sticky="ew")
        resource_row = ttk.Frame(toolbar)
        resource_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        annotation_row = ttk.Frame(toolbar)
        annotation_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        inspect_row = ttk.Frame(toolbar)
        inspect_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        display_row = ttk.Frame(toolbar)
        display_row.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        progress_row = ttk.Frame(toolbar)
        progress_row.grid(row=5, column=0, sticky="ew", pady=(6, 0))
        self.threshold_row = ttk.Frame(toolbar)
        self.threshold_row.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        status_row = ttk.Frame(toolbar)
        status_row.grid(row=7, column=0, sticky="ew", pady=(6, 0))
        cache_row = ttk.Frame(toolbar)
        cache_row.grid(row=8, column=0, sticky="ew", pady=(6, 0))
        self.common_row = common_row
        self.resource_row = resource_row
        self.annotation_row = annotation_row
        self.inspect_row = inspect_row
        self.display_row = display_row
        status_row.columnconfigure(0, weight=1)
        cache_row.columnconfigure(5, weight=1)

        common_row.columnconfigure(16, weight=1)
        resource_row.columnconfigure(9, weight=1)
        annotation_row.columnconfigure(8, weight=1)
        inspect_row.columnconfigure(20, weight=1)
        display_row.columnconfigure(8, weight=1)
        progress_row.columnconfigure(1, weight=1)
        self.threshold_row.columnconfigure(8, weight=1)

        ttk.Label(common_row, text="工作模式").grid(row=0, column=0, padx=(0, 4))
        work_mode_box = ttk.Combobox(
            common_row,
            width=10,
            textvariable=self.work_mode_var,
            values=[WORK_MODE_ANNOTATION, WORK_MODE_INSPECT, WORK_MODE_VIEW],
            state="readonly",
        )
        work_mode_box.grid(row=0, column=1, padx=4)
        work_mode_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.on_work_mode_change()
        )

        ttk.Label(common_row, text="相机").grid(row=0, column=2, padx=(12, 4))
        cameras = [str(camera_id) for camera_id in sorted(self.store.cameras)]
        self.camera_box = ttk.Combobox(
            common_row,
            width=8,
            textvariable=self.camera_var,
            values=cameras,
            state="readonly",
        )
        self.camera_box.grid(row=0, column=3, padx=4)
        self.camera_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.change_camera()
        )

        ttk.Button(common_row, text="播放/暂停", command=self.toggle_play).grid(
            row=0, column=4, padx=(14, 4)
        )
        ttk.Button(common_row, text="后退步长", command=self.prev_frame).grid(
            row=0, column=5, padx=4
        )
        ttk.Button(common_row, text="第一帧", command=self.first_frame).grid(
            row=0, column=6, padx=4
        )
        ttk.Button(common_row, text="前进步长", command=self.next_frame).grid(
            row=0, column=7, padx=4
        )

        ttk.Label(common_row, text="步长").grid(row=0, column=8, padx=(12, 4))
        ttk.Entry(common_row, width=6, textvariable=self.frame_step_var).grid(
            row=0, column=9, padx=4
        )

        ttk.Label(common_row, text="帧序号").grid(row=0, column=10, padx=(14, 4))
        ttk.Entry(common_row, width=10, textvariable=self.frame_var).grid(
            row=0, column=11, padx=4
        )
        ttk.Button(common_row, text="跳帧", command=self.jump_frame).grid(
            row=0, column=12, padx=4
        )

        ttk.Label(common_row, text="PTS").grid(row=0, column=13, padx=(14, 4))
        ttk.Entry(common_row, width=16, textvariable=self.pts_var).grid(
            row=0, column=14, padx=4
        )
        ttk.Button(common_row, text="跳 PTS", command=self.jump_pts).grid(
            row=0, column=15, padx=4
        )

        ttk.Label(resource_row, text="输入源").grid(row=0, column=0, padx=(0, 4))
        self.input_mode_box = ttk.Combobox(
            resource_row,
            width=10,
            textvariable=self.mode_var,
            values=[INPUT_MODE_VIDEO, INPUT_MODE_IMAGE],
            state="readonly",
        )
        self.input_mode_box.grid(row=0, column=1, padx=4)
        self.input_mode_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.change_mode()
        )
        self.choose_media_dir_button = ttk.Button(
            resource_row,
            text="选择图片路径",
            command=self.choose_media_dir,
        )
        self.choose_media_dir_button.grid(row=0, column=2, padx=(12, 4))
        ttk.Button(resource_row, text="选择 JSONL", command=self.choose_jsonl).grid(
            row=0, column=3, padx=4
        )
        ttk.Button(
            resource_row, text="加载预设资源", command=self.load_preset_resources
        ).grid(row=0, column=4, padx=4)
        self.cancel_jsonl_load_button = ttk.Button(
            resource_row,
            text="取消加载",
            command=self.cancel_jsonl_load,
            state=tk.DISABLED,
        )
        self.cancel_jsonl_load_button.grid(row=0, column=5, padx=4)
        self.json_sidebar_button = ttk.Button(
            resource_row,
            text="展开JSON",
            command=self.toggle_json_sidebar,
        )
        self.json_sidebar_button.grid(row=0, column=6, padx=(12, 4))
        self.annotation_mode_button = ttk.Button(
            annotation_row,
            text="开启球标注",
            command=self.toggle_annotation_mode,
        )
        self.annotation_mode_button.grid(row=0, column=0, padx=(0, 4))
        self.annotation_path_button = ttk.Button(
            annotation_row,
            text="选择标注路径",
            command=self.choose_annotation_work_path,
        )
        self.annotation_path_button.grid(row=0, column=1, padx=4)
        self.annotation_save_mode_label = ttk.Label(annotation_row, text="保存模式")
        self.annotation_save_mode_label.grid(row=0, column=2, padx=(12, 4))
        self.annotation_save_mode_box = ttk.Combobox(
            annotation_row,
            width=8,
            textvariable=self.annotation_save_mode_var,
            values=["自动保存", "手动保存"],
            state="readonly",
        )
        self.annotation_save_mode_box.grid(row=0, column=3, padx=4)
        self.save_annotation_button = ttk.Button(
            annotation_row,
            text="保存标注",
            command=self.save_annotation_now,
            state=tk.DISABLED,
        )
        self.save_annotation_button.grid(row=0, column=4, padx=4)
        self.undo_annotation_button = ttk.Button(
            annotation_row,
            text="撤销本帧标注",
            command=self.undo_current_frame_annotation,
            state=tk.DISABLED,
        )
        self.undo_annotation_button.grid(row=0, column=5, padx=4)
        self.clear_annotation_button = ttk.Button(
            annotation_row,
            text="清空本帧标注",
            command=self.clear_current_frame_annotations,
            state=tk.DISABLED,
        )
        self.clear_annotation_button.grid(row=0, column=6, padx=4)
        self.restore_original_ball_button = ttk.Button(
            annotation_row,
            text="撤回原框删除",
            command=self.undo_original_ball_deletion,
            state=tk.DISABLED,
        )
        self.restore_original_ball_button.grid(row=0, column=7, padx=4)

        ttk.Checkbutton(
            inspect_row,
            text="players",
            variable=self.show_players,
            command=self.render_current,
        ).grid(row=0, column=0, padx=(0, 4))
        ttk.Checkbutton(
            inspect_row,
            text="balls",
            variable=self.show_balls,
            command=self.render_current,
        ).grid(row=0, column=1, padx=4)
        ttk.Checkbutton(
            inspect_row,
            text="keypoints",
            variable=self.show_keypoints,
            command=self.render_current,
        ).grid(row=0, column=2, padx=4)
        self.manual_balls_checkbox = ttk.Checkbutton(
            inspect_row,
            text="manual_balls",
            variable=self.show_manual_balls,
            command=self.render_current,
            state=tk.DISABLED,
        )
        self.manual_balls_checkbox.grid(row=0, column=3, padx=4)

        ttk.Label(display_row, text="视频大小").grid(row=0, column=0, padx=(0, 4))
        video_scale = ttk.Scale(
            display_row,
            from_=0.25,
            to=3.5,
            variable=self.video_scale_var,
            orient=tk.HORIZONTAL,
            command=self.on_video_scale_change,
            length=160,
        )
        video_scale.grid(row=0, column=1, padx=4, sticky="w")
        ttk.Label(display_row, textvariable=self.video_scale_text_var, width=5).grid(
            row=0, column=2, padx=(0, 4)
        )

        ttk.Label(display_row, text="显示宽").grid(row=0, column=3, padx=(10, 4))
        ttk.Entry(display_row, width=8, textvariable=self.display_width_var).grid(
            row=0, column=4, padx=4
        )
        ttk.Label(display_row, text="显示高").grid(row=0, column=5, padx=4)
        ttk.Entry(display_row, width=8, textvariable=self.display_height_var).grid(
            row=0, column=6, padx=4
        )
        ttk.Button(
            display_row, text="应用显示", command=self.apply_display_settings
        ).grid(row=0, column=7, padx=4)

        ttk.Label(inspect_row, text="文字").grid(row=0, column=4, padx=(8, 2))
        label_scale = ttk.Scale(
            inspect_row,
            from_=0.25,
            to=1.2,
            variable=self.label_scale_var,
            orient=tk.HORIZONTAL,
            command=self.on_label_scale_change,
            length=80,
        )
        label_scale.grid(row=0, column=5, padx=2)
        ttk.Label(inspect_row, textvariable=self.label_scale_text_var, width=5).grid(
            row=0, column=6, padx=(0, 2)
        )

        ttk.Label(inspect_row, text="标签").grid(row=0, column=7, padx=(8, 2))
        label_mode_box = ttk.Combobox(
            inspect_row,
            width=8,
            textvariable=self.label_mode_var,
            values=["排错模式", "详细模式"],
            state="readonly",
        )
        label_mode_box.grid(row=0, column=8, padx=2)
        label_mode_box.bind(
            "<<ComboboxSelected>>", lambda _event: self.on_label_mode_change()
        )

        self.label_option_widgets: list[ttk.Checkbutton] = []
        option_specs = [
            ("id", self.show_label_id),
            ("team", self.show_label_team),
            ("det", self.show_label_det),
            ("team_s", self.show_label_team_score),
            ("id_s", self.show_label_id_score),
        ]
        for offset, (text, variable) in enumerate(option_specs, start=9):
            checkbox = ttk.Checkbutton(
                inspect_row,
                text=text,
                variable=variable,
                command=self.render_current,
            )
            checkbox.grid(row=0, column=offset, padx=1)
            self.label_option_widgets.append(checkbox)

        ttk.Label(inspect_row, text="偏移").grid(row=0, column=14, padx=(8, 2))
        ttk.Button(
            inspect_row, text="←", width=3, command=lambda: self.step_json_offset(-1)
        ).grid(row=0, column=15, padx=1)
        json_offset_entry = ttk.Entry(
            inspect_row, width=5, textvariable=self.json_offset_var
        )
        json_offset_entry.grid(row=0, column=16, padx=1)
        json_offset_entry.bind("<Return>", lambda _event: self.apply_json_offset())
        ttk.Button(
            inspect_row, text="→", width=3, command=lambda: self.step_json_offset(1)
        ).grid(row=0, column=17, padx=1)
        ttk.Button(inspect_row, text="应用", command=self.apply_json_offset).grid(
            row=0, column=18, padx=(4, 1)
        )
        ttk.Button(inspect_row, text="归零", command=self.reset_json_offset).grid(
            row=0, column=19, padx=1
        )

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
        self.progress_scale.bind(
            "<ButtonRelease-1>", lambda _event: self.apply_progress_seek()
        )
        self.progress_scale.bind("<Return>", lambda _event: self.apply_progress_seek())

        ttk.Label(self.threshold_row, text="异常阈值").grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Label(self.threshold_row, text="det").grid(row=0, column=1, padx=(0, 4))
        det_entry = ttk.Entry(
            self.threshold_row, width=7, textvariable=self.det_threshold_var
        )
        det_entry.grid(row=0, column=2, padx=(0, 10))
        ttk.Label(self.threshold_row, text="team_s").grid(row=0, column=3, padx=(0, 4))
        team_score_entry = ttk.Entry(
            self.threshold_row, width=7, textvariable=self.team_score_threshold_var
        )
        team_score_entry.grid(row=0, column=4, padx=(0, 10))
        ttk.Label(self.threshold_row, text="id_s").grid(row=0, column=5, padx=(0, 4))
        id_score_entry = ttk.Entry(
            self.threshold_row, width=7, textvariable=self.id_score_threshold_var
        )
        id_score_entry.grid(row=0, column=6, padx=(0, 10))
        ttk.Button(
            self.threshold_row, text="应用阈值", command=self.apply_threshold_settings
        ).grid(row=0, column=7, padx=4)
        for entry in [det_entry, team_score_entry, id_score_entry]:
            entry.bind("<Return>", lambda _event: self.apply_threshold_settings())

        ttk.Label(status_row, textvariable=self.status_var, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Label(cache_row, text="缓冲目标").grid(row=0, column=0, padx=(0, 4))
        cache_target_entry = ttk.Entry(
            cache_row, width=6, textvariable=self.cache_target_var
        )
        cache_target_entry.grid(row=0, column=1, padx=(0, 4))
        cache_target_entry.bind("<Return>", lambda _event: self.apply_cache_target())
        ttk.Button(cache_row, text="应用", command=self.apply_cache_target).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Label(cache_row, textvariable=self.cache_status_var, anchor="w").grid(
            row=0, column=3, sticky="w"
        )
        image_separator = ttk.Separator(cache_row, orient=tk.VERTICAL)
        image_separator.grid(row=0, column=4, sticky="ns", padx=8)
        image_before_label = ttk.Label(cache_row, text="图片预载前")
        image_before_label.grid(row=0, column=5, padx=(0, 4), sticky="e")
        image_before_entry = ttk.Entry(
            cache_row, width=5, textvariable=self.image_prefetch_before_var
        )
        image_before_entry.grid(row=0, column=6, padx=(0, 4))
        image_after_label = ttk.Label(cache_row, text="后")
        image_after_label.grid(row=0, column=7, padx=(4, 4))
        image_after_entry = ttk.Entry(
            cache_row, width=5, textvariable=self.image_prefetch_after_var
        )
        image_after_entry.grid(row=0, column=8, padx=(0, 4))
        image_max_label = ttk.Label(cache_row, text="上限")
        image_max_label.grid(row=0, column=9, padx=(4, 4))
        image_max_entry = ttk.Entry(
            cache_row, width=6, textvariable=self.image_cache_max_var
        )
        image_max_entry.grid(row=0, column=10, padx=(0, 4))
        image_apply_button = ttk.Button(
            cache_row, text="应用图片预载", command=self.apply_image_prefetch_range
        )
        image_apply_button.grid(row=0, column=11, padx=(0, 8))
        for entry in [image_before_entry, image_after_entry, image_max_entry]:
            entry.bind("<Return>", lambda _event: self.apply_image_prefetch_range())
        self.image_prefetch_widgets = [
            image_separator,
            image_before_label,
            image_before_entry,
            image_after_label,
            image_after_entry,
            image_max_label,
            image_max_entry,
            image_apply_button,
        ]
        self.update_image_prefetch_controls()
        self.update_label_option_state()

        self.body_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.body_pane.grid(row=1, column=0, sticky="nsew")

        left = ttk.Frame(self.body_pane, padding=8)
        right = ttk.Frame(self.body_pane, padding=8)
        self.right_panel = right
        self.body_pane.add(left, weight=3)

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
        canvas_yscroll = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.video_canvas.yview
        )
        canvas_xscroll = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.video_canvas.xview
        )
        canvas_yscroll.grid(row=0, column=1, sticky="ns")
        canvas_xscroll.grid(row=1, column=0, sticky="ew")
        self.video_canvas.configure(
            xscrollcommand=canvas_xscroll.set,
            yscrollcommand=canvas_yscroll.set,
        )
        self.video_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.video_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.video_canvas.bind("<ButtonRelease-3>", self.on_canvas_right_click)
        self.video_canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        self.video_canvas.bind("<MouseWheel>", self.on_canvas_mousewheel)

        hint = ttk.Label(
            left,
            text=(
                "快捷键：空格播放/暂停，← 后退步长，→ 前进步长，q 退出；"
                "观看/检查模式左键拖动画面，开启球标注后左键拖框、右键删除球框、"
                "双击左键切换平移模式，滚轮上下移动，Shift+滚轮左右移动"
            ),
            anchor="center",
        )
        hint.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        right.rowconfigure(1, weight=1)
        right.rowconfigure(5, weight=1)
        right.columnconfigure(0, weight=1)
        self.json_title_label = ttk.Label(
            right,
            text="当前叠加用 JSON 原数据",
            font=("", 11, "bold"),
        )
        self.json_title_label.grid(row=0, column=0, sticky="w")

        self.json_text = tk.Text(right, wrap=tk.NONE)
        self.json_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        yscroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.json_text.yview)
        xscroll = ttk.Scrollbar(
            right, orient=tk.HORIZONTAL, command=self.json_text.xview
        )
        yscroll.grid(row=1, column=1, sticky="ns", pady=(6, 0))
        xscroll.grid(row=2, column=0, sticky="ew")
        self.json_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.annotation_json_separator = tk.Frame(right, height=3, background="#000000")
        self.annotation_json_separator.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 6)
        )
        self.annotation_json_label = ttk.Label(
            right,
            text="当前标注 JSON",
            font=("", 11, "bold"),
        )
        self.annotation_json_label.grid(row=4, column=0, sticky="w")
        self.annotation_json_text = tk.Text(right, wrap=tk.NONE)
        self.annotation_json_text.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        annotation_yscroll = ttk.Scrollbar(
            right, orient=tk.VERTICAL, command=self.annotation_json_text.yview
        )
        annotation_xscroll = ttk.Scrollbar(
            right, orient=tk.HORIZONTAL, command=self.annotation_json_text.xview
        )
        annotation_yscroll.grid(row=5, column=1, sticky="ns", pady=(6, 0))
        annotation_xscroll.grid(row=6, column=0, sticky="ew")
        self.annotation_json_text.configure(
            yscrollcommand=annotation_yscroll.set,
            xscrollcommand=annotation_xscroll.set,
        )
        self.annotation_json_widgets = [
            self.annotation_json_separator,
            self.annotation_json_label,
            self.annotation_json_text,
            annotation_yscroll,
            annotation_xscroll,
        ]

        self.root.bind("<space>", lambda _event: self.toggle_play())
        self.root.bind("<Left>", lambda _event: self.prev_frame())
        self.root.bind("<Right>", lambda _event: self.next_frame())
        self.root.bind("q", lambda _event: self.root.destroy())
        self.update_choose_dir_button_label()
        self.apply_work_mode_visibility(render=False)

    def toggle_json_sidebar(self) -> None:
        if self.body_pane is None or self.right_panel is None:
            return
        if self.is_view_work_mode():
            return
        if self.json_sidebar_visible:
            self.body_pane.forget(self.right_panel)
            self.json_sidebar_visible = False
            if self.json_sidebar_button is not None:
                self.json_sidebar_button.configure(text="展开JSON")
        else:
            self.body_pane.add(self.right_panel, weight=2)
            self.json_sidebar_visible = True
            if self.json_sidebar_button is not None:
                self.json_sidebar_button.configure(text="收起JSON")

    def is_annotation_work_mode(self) -> bool:
        return self.work_mode_var.get() == WORK_MODE_ANNOTATION

    def is_inspect_work_mode(self) -> bool:
        return self.work_mode_var.get() == WORK_MODE_INSPECT

    def is_view_work_mode(self) -> bool:
        return self.work_mode_var.get() == WORK_MODE_VIEW

    def on_work_mode_change(self) -> None:
        target_mode = self.work_mode_var.get()
        if target_mode != WORK_MODE_ANNOTATION and self.annotation_mode_enabled:
            self.toggle_annotation_mode()
            if self.annotation_mode_enabled:
                self.work_mode_var.set(WORK_MODE_ANNOTATION)
                self.apply_work_mode_visibility()
                return

        if target_mode == WORK_MODE_ANNOTATION and self.is_video_mode():
            self.switch_to_empty_annotation_image_mode()

        self.apply_work_mode_visibility()

    def switch_to_empty_annotation_image_mode(self) -> None:
        self.is_playing = False
        if self.jsonl_cancel_event is not None:
            self.jsonl_cancel_event.set()
        self.jsonl_load_token += 1
        self.jsonl_loading = False
        self.jsonl_cancel_event = None
        self.update_jsonl_load_button_state()
        self.cancel_video_task()
        self.cancel_image_prefetch_task()
        self.clear_video_cache()
        self.clear_image_cache()
        self.close_video_readers()
        self.video_keyframe_indexes.clear()
        self.reset_annotation_state()
        self.mode_var.set(INPUT_MODE_IMAGE)
        self.store = AnnotationStore()
        self._update_manual_balls_checkbox_state()
        self.current_jsonl_path = ""
        self.frame_image_dir = ""
        self.current_camera = DEFAULT_CAMERA
        self.camera_var.set(str(DEFAULT_CAMERA))
        self.current_index = 0
        self.current_raw_frame = None
        self.current_frame_image = None
        self.canvas_image_id = None
        if hasattr(self, "video_canvas"):
            self.video_canvas.delete("all")
        if hasattr(self, "progress_scale"):
            self.progress_scale.configure(from_=1, to=1)
        self.progress_var.set(1)
        self.frame_var.set("1")
        self.pts_var.set("")
        self.refresh_camera_choices()
        self.show_json({})
        self.status_var.set("已进入标注模式：请重新选择图片路径和 JSONL")

    def collapse_json_sidebar(self) -> None:
        if self.body_pane is None or self.right_panel is None:
            return
        if not self.json_sidebar_visible:
            return
        self.body_pane.forget(self.right_panel)
        self.json_sidebar_visible = False
        if self.json_sidebar_button is not None:
            self.json_sidebar_button.configure(text="展开JSON")

    def apply_work_mode_visibility(self, render: bool = True) -> None:
        if self.is_annotation_work_mode():
            if self.input_mode_box is not None:
                self.input_mode_box.configure(
                    values=[INPUT_MODE_IMAGE],
                    state="readonly",
                )
            self.mode_var.set(INPUT_MODE_IMAGE)
        elif self.input_mode_box is not None:
            self.input_mode_box.configure(
                values=[INPUT_MODE_VIDEO, INPUT_MODE_IMAGE],
                state="readonly",
            )

        if self.annotation_row is not None:
            if self.is_annotation_work_mode():
                self.annotation_row.grid()
            else:
                self.annotation_row.grid_remove()
        if self.inspect_row is not None:
            if self.is_inspect_work_mode():
                self.inspect_row.grid()
            else:
                self.inspect_row.grid_remove()

        if self.json_sidebar_button is not None:
            if self.is_view_work_mode():
                self.json_sidebar_button.grid_remove()
                self.collapse_json_sidebar()
            else:
                self.json_sidebar_button.grid()
        self.update_choose_dir_button_label()
        self.update_annotation_buttons()
        self.update_label_option_state()
        self.update_json_panel_mode()
        if render and self.store.cameras:
            self.render_current()

    def update_json_panel_mode(self) -> None:
        show_annotation_json = self.is_annotation_work_mode()
        for widget in getattr(self, "annotation_json_widgets", []):
            if show_annotation_json:
                widget.grid()
            else:
                widget.grid_remove()
        if self.json_title_label is not None:
            title = (
                "当前原始 JSON" if show_annotation_json else "当前叠加用 JSON 原数据"
            )
            self.json_title_label.configure(text=title)

    def update_choose_dir_button_label(self) -> None:
        if self.choose_media_dir_button is None:
            return
        label = "选择视频路径" if self.is_video_mode() else "选择图片路径"
        self.choose_media_dir_button.configure(text=label)

    def update_annotation_buttons(self) -> None:
        mode_text = "关闭球标注" if self.annotation_mode_enabled else "开启球标注"
        edit_state = tk.NORMAL if self.annotation_mode_enabled else tk.DISABLED
        path_state = tk.DISABLED if self.annotation_io_running else tk.NORMAL
        if self.annotation_mode_button is not None:
            self.annotation_mode_button.configure(
                text=mode_text,
                state=tk.DISABLED if self.annotation_io_running else tk.NORMAL,
            )
        if self.annotation_path_button is not None:
            self.annotation_path_button.configure(state=path_state)
        if self.save_annotation_button is not None:
            save_state = (
                tk.NORMAL
                if self.annotation_work_path is not None
                and not self.annotation_io_running
                and not self.annotation_saving
                else tk.DISABLED
            )
            self.save_annotation_button.configure(state=save_state)
        if self.undo_annotation_button is not None:
            self.undo_annotation_button.configure(state=edit_state)
        if self.clear_annotation_button is not None:
            self.clear_annotation_button.configure(state=edit_state)
        if self.restore_original_ball_button is not None:
            self.restore_original_ball_button.configure(state=edit_state)
        if hasattr(self, "video_canvas"):
            cursor = "crosshair" if self.annotation_mode_enabled else "fleur"
            self.video_canvas.configure(cursor=cursor)

    def reset_annotation_state(self) -> None:
        self.save_annotation_work_if_dirty()
        if self.annotation_save_after_id is not None:
            self.root.after_cancel(self.annotation_save_after_id)
            self.annotation_save_after_id = None
        self.annotation_mode_enabled = False
        self.annotation_pan_mode = False
        self.manual_ball_annotations.clear()
        self.deleted_original_ball_indices.clear()
        self.deleted_original_ball_history.clear()
        self.annotation_output_lines.clear()
        self.annotation_key_to_output_indices.clear()
        self.annotation_output_entries.clear()
        self.annotation_work_path = None
        self.annotation_loaded_path = None
        self.annotation_io_token += 1
        self.annotation_io_running = False
        self.annotation_dirty = False
        self.annotation_dirty_version = 0
        self.annotation_last_saved_version = 0
        self.close_after_annotation_save = False
        self.pending_after_annotation_save = None
        self.annotation_save_error = None
        self.cancel_annotation_drag()
        self.update_annotation_buttons()

    def current_annotation_key(self) -> tuple[int, int] | None:
        pts = self.current_display_pts()
        if pts is None:
            return None
        return (self.current_camera, pts)

    def annotation_key_from_item(
        self,
        item: dict[str, Any],
        fallback_camera_id: int,
    ) -> tuple[int, int] | None:
        camera_id = camera_id_from_item(item) or fallback_camera_id
        pts = item.get("pts")
        try:
            return (int(camera_id), int(pts))
        except (TypeError, ValueError):
            return None

    def current_frame_manual_balls(self) -> list[dict[str, Any]]:
        key = self.current_annotation_key()
        if key is None:
            return []
        return self.manual_ball_annotations.get(key, [])

    def current_frame_deleted_original_balls(self) -> set[int]:
        key = self.current_annotation_key()
        if key is None:
            return set()
        return self.deleted_original_ball_indices.get(key, set())

    def default_annotation_work_path(self) -> Path:
        jsonl_text = resource_to_text(self.current_jsonl_path)
        if jsonl_text and not is_remote_resource(jsonl_text):
            source_path = local_path_from_resource(jsonl_text)
            stem = source_path.stem or "annotations"
            return source_path.with_name(f"{stem}{ANNOTATION_WORK_SUFFIX}.jsonl")

        name = display_resource_name(jsonl_text) if jsonl_text else "annotations.jsonl"
        stem = Path(name).stem or "annotations"
        return SCRIPT_DIR / f"{stem}{ANNOTATION_WORK_SUFFIX}.jsonl"

    def annotation_work_path_in_directory(self, directory: Path) -> Path:
        return directory / self.default_annotation_work_path().name

    def resolve_annotation_work_path(self, selected_text: str) -> tuple[Path, bool]:
        if is_remote_resource(selected_text):
            raise ValueError("标注工作 JSONL 必须保存到本地路径，暂不支持远程地址")

        selected_path = Path(selected_text).expanduser().resolve()
        if selected_path.exists():
            if selected_path.is_dir():
                return self.annotation_work_path_in_directory(selected_path), True
            return selected_path, False

        if selected_path.suffix.lower() == ".jsonl":
            return selected_path, False
        return self.annotation_work_path_in_directory(selected_path), True

    def choose_annotation_work_path(self, enable_after_done: bool = False) -> str:
        if not self.store.cameras:
            messagebox.showerror("无法选择标注路径", "请先加载 JSONL 数据")
            return "cancelled"
        if self.annotation_io_running:
            messagebox.showinfo("标注任务进行中", "标注 JSONL 正在载入或创建，请稍候")
            return "pending"

        initial_path = self.annotation_work_path or self.default_annotation_work_path()
        dialog = AnnotationWorkPathDialog(
            self.root,
            "选择标注路径",
            str(initial_path),
            initial_path.parent if initial_path.parent.is_dir() else PROJECT_DIR,
        )
        if dialog.result is None:
            return "cancelled"
        try:
            selected_path, selected_directory = self.resolve_annotation_work_path(
                dialog.result
            )
        except Exception as exc:
            messagebox.showerror("标注路径不可用", str(exc))
            return "cancelled"

        self.annotation_work_path = selected_path
        file_existed_before = selected_path.is_file()
        already_loaded = (
            file_existed_before
            and self.annotation_loaded_path is not None
            and self.annotation_loaded_path == selected_path
            and bool(self.annotation_output_lines)
        )
        if already_loaded:
            self.status_var.set(f"已确认当前标注文件：{self.annotation_work_path}")
            self.render_current()
            return "ready"

        self.start_annotation_work_io_task(
            selected_path,
            file_existed_before=file_existed_before,
            selected_directory=selected_directory,
            enable_after_done=enable_after_done,
        )
        return "pending"

    def start_annotation_work_io_task(
        self,
        selected_path: Path,
        *,
        file_existed_before: bool,
        selected_directory: bool,
        enable_after_done: bool,
    ) -> None:
        self.annotation_io_token += 1
        token = self.annotation_io_token
        self.annotation_io_running = True
        self.update_annotation_buttons()

        action = "load" if file_existed_before else "create"
        if action == "load":
            self.status_var.set(f"正在载入标注 JSONL：{selected_path} | 0%")
        elif selected_directory:
            self.status_var.set(
                f"正在复制标注 JSONL 到文件夹：{selected_path.parent} | 0%"
            )
        else:
            self.status_var.set(f"正在复制标注 JSONL：{selected_path} | 0%")

        manual_snapshot = deepcopy(self.manual_ball_annotations)
        deleted_snapshot = {
            key: set(indices)
            for key, indices in self.deleted_original_ball_indices.items()
        }

        def progress_cb(pct: int, detail: str) -> None:
            self.video_result_queue.put(
                {
                    "kind": "annotation_work_progress",
                    "token": token,
                    "action": action,
                    "path": selected_path,
                    "pct": pct,
                    "detail": detail,
                }
            )

        def worker() -> None:
            error = None
            manual_annotations = None
            deleted_original_ball_indices = None
            output_lines = None
            key_to_output_indices = None
            output_entries = None
            try:
                if action == "load":
                    (
                        manual_annotations,
                        deleted_original_ball_indices,
                        deleted_ball_history,
                    ) = self.read_annotation_work_file_data(selected_path, progress_cb)
                else:
                    manual_annotations = manual_snapshot
                    deleted_original_ball_indices = deleted_snapshot
                    deleted_ball_history = None  # create 模式不需要 history
                (
                    output_lines,
                    key_to_output_indices,
                    output_entries,
                ) = self.build_annotation_output_cache(
                    manual_annotations or {},
                    deleted_original_ball_indices or {},
                    progress_callback=progress_cb,
                )
                if action == "create":
                    self.write_annotation_output_lines(
                        selected_path,
                        output_lines,
                        progress_callback=progress_cb,
                    )
            except Exception as exc:
                error = str(exc)

            self.video_result_queue.put(
                {
                    "kind": "annotation_work_done",
                    "token": token,
                    "action": action,
                    "path": selected_path,
                    "error": error,
                    "manual_annotations": manual_annotations,
                    "deleted_original_ball_indices": deleted_original_ball_indices,
                    "deleted_ball_history": deleted_ball_history,
                    "output_lines": output_lines,
                    "key_to_output_indices": key_to_output_indices,
                    "output_entries": output_entries,
                    "enable_after_done": enable_after_done,
                }
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def read_annotation_work_file_data(
        self,
        path: Path,
        progress_callback: Any | None = None,
    ) -> tuple[
        dict[tuple[int, int], list[dict[str, Any]]],
        dict[tuple[int, int], set[int]],
        dict[tuple[int, int], list[tuple[int, float]]],
    ]:
        file_size = path.stat().st_size if path.is_file() else 0
        bytes_read = 0
        total_lines = 0
        loaded: dict[tuple[int, int], list[dict[str, Any]]] = {}
        kept_original_signatures: dict[tuple[int, int], Counter[str]] = {}
        work_keys: set[tuple[int, int]] = set()

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                bytes_read += len(line.encode("utf-8"))
                total_lines += 1
                if progress_callback and total_lines % 500 == 0:
                    pct = min(95, int(bytes_read * 95 / max(1, file_size)))
                    progress_callback(pct, f"{total_lines} 行")

                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                result, _missing = result_dict_from_item(item)
                camera_id = camera_id_from_item(item)
                pts = item.get("pts")
                try:
                    key = (int(camera_id), int(pts))
                except (TypeError, ValueError):
                    continue

                balls, _state = detection_list_from_result(result, "balls")
                work_keys.add(key)
                manual_balls = [
                    deepcopy(ball) for ball in balls if is_manual_ball(ball)
                ]
                for ball in manual_balls:
                    ball.pop("source", None)
                    ball["score"] = MANUAL_BALL_SCORE
                if manual_balls:
                    loaded[key] = manual_balls
                kept_original_signatures[key] = Counter(
                    ball_signature(ball) for ball in balls if not is_manual_ball(ball)
                )

        if progress_callback:
            progress_callback(96, "分析被删除的原始球框")
        deleted, deleted_history = self.infer_deleted_original_balls(
            kept_original_signatures, work_keys
        )
        if progress_callback:
            progress_callback(100, f"{total_lines} 行")
        return loaded, deleted, deleted_history

    def sorted_annotation_items(self) -> list[tuple[int, int, int, dict[str, Any]]]:
        items: list[tuple[int, int, int, dict[str, Any]]] = []
        for camera_id, annotation in self.store.cameras.items():
            for item in annotation.items:
                result, _missing = result_dict_from_item(item)
                try:
                    pts = int(item.get("pts"))
                except (TypeError, ValueError):
                    pts = 0
                try:
                    sort_camera = int(result.get("cam_idx", camera_id))
                except (TypeError, ValueError):
                    sort_camera = camera_id
                items.append((pts, sort_camera, camera_id, item))
        items.sort(key=lambda entry: (entry[0], entry[1]))
        return items

    def build_annotation_output_cache(
        self,
        manual_annotations: dict[tuple[int, int], list[dict[str, Any]]],
        deleted_original_ball_indices: dict[tuple[int, int], set[int]],
        progress_callback: Any | None = None,
    ) -> tuple[
        list[str],
        dict[tuple[int, int], list[int]],
        list[tuple[int, dict[str, Any]]],
    ]:
        items = self.sorted_annotation_items()
        total_items = max(1, len(items))
        output_lines: list[str] = []
        key_to_output_indices: dict[tuple[int, int], list[int]] = {}
        output_entries: list[tuple[int, dict[str, Any]]] = []

        for line_index, (_pts, _sort_camera, camera_id, item) in enumerate(items):
            output_item = self.build_annotation_work_item(
                item,
                camera_id,
                manual_annotations,
                deleted_original_ball_indices,
            )
            output_lines.append(json.dumps(output_item, ensure_ascii=False))
            output_entries.append((camera_id, item))
            key = self.annotation_key_from_item(item, camera_id)
            if key is not None:
                key_to_output_indices.setdefault(key, []).append(line_index)

            if progress_callback and (line_index + 1) % 500 == 0:
                pct = min(95, int((line_index + 1) * 95 / total_items))
                progress_callback(
                    pct, f"构建输出缓存 {line_index + 1}/{total_items} 行"
                )

        if progress_callback:
            progress_callback(96, "输出缓存构建完成")
        return output_lines, key_to_output_indices, output_entries

    def rebuild_annotation_output_lines_for_key(
        self,
        key: tuple[int, int],
    ) -> None:
        indices = self.annotation_key_to_output_indices.get(key)
        if not indices:
            return
        for line_index in indices:
            if line_index >= len(self.annotation_output_entries):
                continue
            fallback_camera_id, item = self.annotation_output_entries[line_index]
            output_item = self.build_annotation_work_item(
                item,
                fallback_camera_id,
                self.manual_ball_annotations,
                self.deleted_original_ball_indices,
            )
            self.annotation_output_lines[line_index] = json.dumps(
                output_item,
                ensure_ascii=False,
            )

    def write_annotation_output_lines(
        self,
        target_path: Path,
        output_lines: list[str],
        progress_callback: Any | None = None,
    ) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        total_lines = max(1, len(output_lines))
        with self.annotation_write_lock:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
                for line_index, line in enumerate(output_lines, start=1):
                    fh.write(line)
                    fh.write("\n")
                    if progress_callback and line_index % 500 == 0:
                        pct = min(99, int(line_index * 100 / total_lines))
                        progress_callback(pct, f"写入 {line_index}/{total_lines} 行")
            if progress_callback:
                progress_callback(
                    100, f"写入 {len(output_lines)}/{len(output_lines)} 行"
                )
            tmp_path.replace(target_path)

    def handle_annotation_work_progress(self, result: dict[str, Any]) -> None:
        if result.get("token") != self.annotation_io_token:
            return
        action = result.get("action")
        pct = result.get("pct")
        path = result.get("path")
        detail = result.get("detail") or ""
        verb = "载入" if action == "load" else "复制"
        self.status_var.set(f"正在{verb}标注 JSONL：{path} | {pct}% | {detail}")

    def handle_annotation_work_done(self, result: dict[str, Any]) -> None:
        if result.get("token") != self.annotation_io_token:
            return
        self.annotation_io_running = False
        self.update_annotation_buttons()

        error = result.get("error")
        path = result.get("path")
        if error is not None:
            messagebox.showerror("标注 JSONL 处理失败", str(error))
            self.status_var.set(f"标注 JSONL 处理失败：{error}")
            return

        if result.get("action") == "load":
            self.manual_ball_annotations = result.get("manual_annotations") or {}
            self.deleted_original_ball_indices = (
                result.get("deleted_original_ball_indices") or {}
            )
            loaded_history = result.get("deleted_ball_history")
            if loaded_history is not None:
                self.deleted_original_ball_history = loaded_history
            else:
                self.deleted_original_ball_history = {
                    key: [(idx, 0.0) for idx in sorted(indices)]
                    for key, indices in self.deleted_original_ball_indices.items()
                }
            action_text = "已载入标注 JSONL"
        else:
            action_text = "已创建标注工作 JSONL"

        self.annotation_output_lines = result.get("output_lines") or []
        self.annotation_key_to_output_indices = (
            result.get("key_to_output_indices") or {}
        )
        self.annotation_output_entries = result.get("output_entries") or []
        self.annotation_work_path = path
        self.annotation_loaded_path = path
        self.annotation_dirty = False
        self.annotation_save_error = None

        if result.get("enable_after_done"):
            self.annotation_mode_enabled = True
            self.update_annotation_buttons()
            self.status_var.set(f"{action_text}，球标注已开启：{path}")
        else:
            self.status_var.set(f"{action_text}：{path}")
        self.render_current()

    def toggle_annotation_mode(self) -> None:
        if self.annotation_mode_enabled:
            if self.annotation_dirty:
                answer = messagebox.askyesnocancel(
                    "标注尚未保存",
                    "当前标注 JSONL 有未保存改动，关闭球标注前是否保存？\n\n"
                    "选择「是」：立即后台保存。\n"
                    "选择「否」：暂不保存，保留当前内存改动。\n"
                    "选择「取消」：继续标注。",
                )
                if answer is None:
                    return
                if answer:
                    if self.annotation_save_after_id is not None:
                        self.root.after_cancel(self.annotation_save_after_id)
                        self.annotation_save_after_id = None
                    self.start_annotation_save_worker()
                    self.status_var.set(
                        f"正在后台保存标注 JSONL：{self.annotation_work_path}"
                    )
            self.annotation_mode_enabled = False
            self.annotation_pan_mode = False
            self.cancel_annotation_drag()
            self.update_annotation_buttons()
            self.render_current()
            return

        if not self.store.cameras:
            messagebox.showerror("无法标注", "请先加载 JSONL 数据")
            return
        result = self.choose_annotation_work_path(enable_after_done=True)
        if result == "cancelled":
            self.status_var.set("已取消开启球标注")
            return
        if result == "pending":
            return
        try:
            self.ensure_annotation_work_file()
        except Exception as exc:
            messagebox.showerror("标注文件创建失败", str(exc))
            self.status_var.set(f"标注文件创建失败：{exc}")
            return

        self.annotation_mode_enabled = True
        self.update_annotation_buttons()
        self.status_var.set(
            f"球标注已开启：{self.annotation_work_path} | 保存模式 {self.annotation_save_mode_var.get()}"
        )
        self.render_current()

    def ensure_annotation_work_file(self) -> None:
        if self.annotation_work_path is None:
            self.annotation_work_path = self.default_annotation_work_path()
        self.annotation_work_path.parent.mkdir(parents=True, exist_ok=True)
        if self.annotation_work_path.is_file():
            if self.annotation_loaded_path == self.annotation_work_path:
                return
            self.load_manual_annotations_from_work_file()
            (
                self.annotation_output_lines,
                self.annotation_key_to_output_indices,
                self.annotation_output_entries,
            ) = self.build_annotation_output_cache(
                self.manual_ball_annotations,
                self.deleted_original_ball_indices,
            )
            self.annotation_dirty = False
            self.annotation_loaded_path = self.annotation_work_path
            return
        (
            self.annotation_output_lines,
            self.annotation_key_to_output_indices,
            self.annotation_output_entries,
        ) = self.build_annotation_output_cache(
            self.manual_ball_annotations,
            self.deleted_original_ball_indices,
        )
        self.write_annotation_output_lines(
            self.annotation_work_path,
            self.annotation_output_lines,
        )
        self.annotation_loaded_path = self.annotation_work_path

    def load_manual_annotations_from_work_file(self) -> None:
        if self.annotation_work_path is None or not self.annotation_work_path.is_file():
            return

        loaded, deleted, history = self.read_annotation_work_file_data(
            self.annotation_work_path
        )
        self.manual_ball_annotations = loaded
        self.deleted_original_ball_indices = deleted
        if history is not None:
            self.deleted_original_ball_history = history
        else:
            self.deleted_original_ball_history = {
                key: [(idx, 0.0) for idx in sorted(indices)]
                for key, indices in self.deleted_original_ball_indices.items()
            }

    def infer_deleted_original_balls(
        self,
        kept_original_signatures: dict[tuple[int, int], Counter[str]],
        work_keys: set[tuple[int, int]],
    ) -> tuple[dict[tuple[int, int], set[int]], dict[tuple[int, int], list[tuple[int, float]]]]:
        """从已加载的标注 JSONL 推断被删除的原始球框。

        对比原始 JSONL 和标注 JSONL 中非人工球签名的差异。
        缺失的球被视为已删除，记录其索引和原始 score 以便撤回。
        返回 (deleted_indices, deleted_history)。
        """
        deleted: dict[tuple[int, int], set[int]] = {}
        history: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for camera_id, annotation in self.store.cameras.items():
            for item in annotation.items:
                key = self.annotation_key_from_item(item, camera_id)
                if key is None or key not in work_keys:
                    continue
                result, _missing = result_dict_from_item(item)
                original_balls, _state = detection_list_from_result(result, "balls")
                kept = kept_original_signatures.get(key, Counter()).copy()
                for index, ball in enumerate(
                    ball for ball in original_balls if not is_manual_ball(ball)
                ):
                    signature = ball_signature(ball)
                    if kept[signature] > 0:
                        kept[signature] -= 1
                        continue
                    deleted.setdefault(key, set()).add(index)
                    history.setdefault(key, []).append(
                        (index, safe_score(ball.get("score")))
                    )
        return deleted, history

    def save_annotation_work_if_dirty(self) -> None:
        if not self.annotation_dirty:
            return
        if self.annotation_save_after_id is not None:
            self.root.after_cancel(self.annotation_save_after_id)
            self.annotation_save_after_id = None
        try:
            if self.annotation_work_path is None:
                self.annotation_work_path = self.default_annotation_work_path()
            self.write_annotation_output_lines(
                self.annotation_work_path,
                self.annotation_output_lines,
            )
            self.annotation_last_saved_version = self.annotation_dirty_version
            self.annotation_dirty = False
        except Exception as exc:
            self.annotation_save_error = str(exc)
            messagebox.showerror("标注自动保存失败", str(exc))
            self.status_var.set(f"标注自动保存失败：{exc}")

    def mark_annotation_dirty_and_save(
        self,
        changed_key: tuple[int, int] | None = None,
    ) -> bool:
        if changed_key is not None:
            self.rebuild_annotation_output_lines_for_key(changed_key)
        self.annotation_dirty = True
        self.annotation_dirty_version += 1
        return True

    def schedule_annotation_save(self) -> None:
        if self.annotation_save_after_id is not None:
            self.root.after_cancel(self.annotation_save_after_id)
        self.annotation_save_after_id = self.root.after(
            ANNOTATION_SAVE_DEBOUNCE_MS,
            self.start_annotation_save_worker,
        )

    def is_annotation_auto_save_mode(self) -> bool:
        return self.annotation_save_mode_var.get() == "自动保存"

    def set_pending_after_annotation_save(
        self,
        action: str,
        *args: Any,
    ) -> None:
        self.pending_after_annotation_save = (action, args)

    def run_pending_after_annotation_save(self) -> None:
        pending = self.pending_after_annotation_save
        self.pending_after_annotation_save = None
        if pending is None:
            return
        action, args = pending
        if action == "seek":
            self.seek_and_render(int(args[0]))
        elif action == "open_camera":
            camera_id = int(args[0])
            target_pts = args[1]
            self.open_camera(camera_id, target_pts=target_pts)
        elif action == "change_mode":
            self.change_mode()
        elif action == "close":
            self.close_without_annotation_prompt()

    def start_annotation_save_for_pending_action(
        self,
        action: str,
        *args: Any,
    ) -> bool:
        if not self.annotation_dirty:
            return True
        self.set_pending_after_annotation_save(action, *args)
        if self.annotation_save_after_id is not None:
            self.root.after_cancel(self.annotation_save_after_id)
            self.annotation_save_after_id = None
        if self.annotation_saving:
            self.status_var.set("标注 JSONL 正在保存中，保存完成后继续操作")
            return False
        self.start_annotation_save_worker()
        self.status_var.set("正在保存标注 JSONL，保存完成后继续操作")
        return False

    def confirm_unsaved_annotation_navigation(
        self,
        action_name: str,
        pending_action: str,
        *pending_args: Any,
    ) -> bool:
        if not self.annotation_dirty:
            return True
        if self.is_annotation_auto_save_mode():
            return self.start_annotation_save_for_pending_action(
                pending_action,
                *pending_args,
            )

        answer = messagebox.askyesnocancel(
            "标注尚未保存",
            f"当前标注 JSONL 有未保存改动，是否在{action_name}前保存？\n\n"
            "选择「是」：保存完成后继续操作。\n"
            "选择「否」：不保存并继续操作。\n"
            "选择「取消」：留在当前画面。",
        )
        if answer is None:
            return False
        if answer:
            return self.start_annotation_save_for_pending_action(
                pending_action,
                *pending_args,
            )
        return True

    def start_annotation_save_worker(self) -> None:
        self.annotation_save_after_id = None
        if not self.annotation_dirty:
            return
        if self.annotation_saving:
            self.annotation_save_after_id = self.root.after(
                ANNOTATION_SAVE_DEBOUNCE_MS,
                self.start_annotation_save_worker,
            )
            return
        try:
            if self.annotation_work_path is None:
                self.annotation_work_path = self.default_annotation_work_path()
            self.annotation_work_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.annotation_save_error = str(exc)
            self.status_var.set(f"标注自动保存失败：{exc}")
            return

        save_version = self.annotation_dirty_version
        save_path = self.annotation_work_path
        output_lines_snapshot = list(self.annotation_output_lines)
        self.annotation_saving = True
        self.update_annotation_buttons()
        self.status_var.set(f"正在保存标注 JSONL：{save_path} | 0%")

        def progress_cb(pct: int, detail: str) -> None:
            self.video_result_queue.put(
                {
                    "kind": "annotation_save_progress",
                    "version": save_version,
                    "path": save_path,
                    "pct": pct,
                    "detail": detail,
                }
            )

        def worker() -> None:
            error = None
            try:
                self.write_annotation_output_lines(
                    save_path,
                    output_lines_snapshot,
                    progress_callback=progress_cb,
                )
            except Exception as exc:
                error = str(exc)
            self.video_result_queue.put(
                {
                    "kind": "annotation_save",
                    "version": save_version,
                    "path": save_path,
                    "error": error,
                }
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def save_annotation_now(self) -> None:
        if self.annotation_work_path is None:
            messagebox.showinfo("无法保存", "请先选择或创建标注 JSONL 路径")
            return
        if self.annotation_io_running:
            messagebox.showinfo("标注任务进行中", "标注 JSONL 正在载入或创建，请稍候")
            return
        if self.annotation_saving:
            self.status_var.set(f"标注 JSONL 正在保存中：{self.annotation_work_path}")
            return
        if self.annotation_save_after_id is not None:
            self.root.after_cancel(self.annotation_save_after_id)
            self.annotation_save_after_id = None
        if not self.annotation_dirty:
            self.status_var.set(f"标注 JSONL 无需保存：{self.annotation_work_path}")
            return
        self.start_annotation_save_worker()

    def handle_annotation_save_progress(self, result: dict[str, Any]) -> None:
        if int(result.get("version", -1)) < self.annotation_last_saved_version:
            return
        pct = result.get("pct")
        detail = result.get("detail") or ""
        path = result.get("path")
        self.status_var.set(f"正在保存标注 JSONL：{path} | {pct}% | {detail}")

    def handle_annotation_save_result(self, result: dict[str, Any]) -> None:
        version = int(result["version"])
        error = result.get("error")
        self.annotation_saving = False
        self.update_annotation_buttons()
        if error is not None:
            self.annotation_save_error = str(error)
            messagebox.showerror("标注自动保存失败", str(error))
            self.status_var.set(f"标注自动保存失败：{error}")
            return
        self.annotation_save_error = None
        path = result.get("path")
        if path is not None:
            self.annotation_loaded_path = path
        if version >= self.annotation_last_saved_version:
            self.annotation_last_saved_version = version
        if version == self.annotation_dirty_version:
            self.annotation_dirty = False
            self.status_var.set(f"标注 JSONL 保存完成：{path}")
            if self.close_after_annotation_save:
                self.close_after_annotation_save = False
                self.close_without_annotation_prompt()
                return
            self.run_pending_after_annotation_save()
        elif self.annotation_dirty:
            self.schedule_annotation_save()

    def write_annotation_work_jsonl(
        self,
        manual_annotations: dict[tuple[int, int], list[dict[str, Any]]] | None = None,
        deleted_original_ball_indices: dict[tuple[int, int], set[int]] | None = None,
        mark_clean: bool = True,
        output_path: Path | None = None,
        progress_callback: Any | None = None,
    ) -> None:
        if output_path is None and self.annotation_work_path is None:
            self.annotation_work_path = self.default_annotation_work_path()
        target_path = output_path or self.annotation_work_path
        if target_path is None:
            raise ValueError("标注 JSONL 路径未设置")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        if manual_annotations is None:
            manual_annotations = deepcopy(self.manual_ball_annotations)
        if deleted_original_ball_indices is None:
            deleted_original_ball_indices = {
                key: set(indices)
                for key, indices in self.deleted_original_ball_indices.items()
            }

        items: list[tuple[int, int, int, dict[str, Any]]] = []
        for camera_id, annotation in self.store.cameras.items():
            for item in annotation.items:
                result, _missing = result_dict_from_item(item)
                try:
                    pts = int(item.get("pts"))
                except (TypeError, ValueError):
                    pts = 0
                try:
                    sort_camera = int(result.get("cam_idx", camera_id))
                except (TypeError, ValueError):
                    sort_camera = camera_id
                items.append((pts, sort_camera, camera_id, item))
        items.sort(key=lambda entry: (entry[0], entry[1]))
        total_items = max(1, len(items))

        with self.annotation_write_lock:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
                for line_index, (_pts, _sort_camera, camera_id, item) in enumerate(
                    items,
                    start=1,
                ):
                    output_item = self.build_annotation_work_item(
                        item,
                        camera_id,
                        manual_annotations,
                        deleted_original_ball_indices,
                    )
                    fh.write(json.dumps(output_item, ensure_ascii=False))
                    fh.write("\n")
                    if progress_callback and line_index % 500 == 0:
                        pct = min(99, int(line_index * 100 / total_items))
                        progress_callback(pct, f"{line_index}/{total_items} 行")
            if progress_callback:
                progress_callback(100, f"{len(items)}/{len(items)} 行")
            tmp_path.replace(target_path)
        if output_path is None:
            self.annotation_loaded_path = self.annotation_work_path
        if mark_clean:
            self.annotation_dirty = False

    def build_annotation_work_item(
        self,
        item: dict[str, Any],
        fallback_camera_id: int,
        manual_annotations: dict[tuple[int, int], list[dict[str, Any]]],
        deleted_original_ball_indices: dict[tuple[int, int], set[int]] | None = None,
    ) -> dict[str, Any]:
        output_item = deepcopy(item)
        result, _missing = result_dict_from_item(output_item)
        if "result" not in output_item or not isinstance(
            output_item.get("result"), dict
        ):
            output_item["result"] = result

        manual_key = self.annotation_key_from_item(output_item, fallback_camera_id)

        original_balls, _state = detection_list_from_result(result, "balls")
        deleted_indices = (
            deleted_original_ball_indices.get(manual_key, set())
            if manual_key is not None and deleted_original_ball_indices is not None
            else set()
        )
        copied_original_balls: list[dict[str, Any]] = []
        original_idx = -1
        for ball in original_balls:
            if is_manual_ball(ball):
                continue
            original_idx += 1
            copied_ball = deepcopy(ball)
            if original_idx in deleted_indices:
                copied_ball["score"] = 0
            copied_original_balls.append(copied_ball)
        manual_balls = (
            [deepcopy(ball) for ball in manual_annotations.get(manual_key, [])]
            if manual_key is not None
            else []
        )
        for ball in manual_balls:
            ball.pop("source", None)
            ball["score"] = MANUAL_BALL_SCORE

        result["balls"] = copied_original_balls + manual_balls
        return output_item

    def undo_current_frame_annotation(self) -> None:
        if not self.annotation_mode_enabled:
            return
        key = self.current_annotation_key()
        if key is None:
            return
        balls = self.manual_ball_annotations.get(key)
        if not balls:
            self.status_var.set("当前帧没有可撤销的人工球框")
            return
        balls.pop()
        if balls:
            self.manual_ball_annotations[key] = balls
        else:
            self.manual_ball_annotations.pop(key, None)
        if self.mark_annotation_dirty_and_save(key):
            self.status_var.set(
                f"已撤销当前帧最后一个球框，标注未保存：{self.annotation_work_path}"
            )
        self.render_current()

    def clear_current_frame_annotations(self) -> None:
        if not self.annotation_mode_enabled:
            return
        key = self.current_annotation_key()
        if key is None or key not in self.manual_ball_annotations:
            self.status_var.set("当前帧没有人工球框可清空")
            return
        self.manual_ball_annotations.pop(key, None)
        if self.mark_annotation_dirty_and_save(key):
            self.status_var.set(
                f"已清空当前帧人工球框，标注未保存：{self.annotation_work_path}"
            )
        self.render_current()

    def undo_original_ball_deletion(self) -> None:
        if not self.annotation_mode_enabled:
            return
        key = self.current_annotation_key()
        if key is None:
            return
        history = self.deleted_original_ball_history.get(key)
        if not history:
            self.status_var.set("当前帧没有可撤回的原始球框删除")
            return
        index, _original_score = history.pop()
        deleted_indices = self.deleted_original_ball_indices.get(key)
        if deleted_indices is not None:
            deleted_indices.discard(index)
            if not deleted_indices:
                self.deleted_original_ball_indices.pop(key, None)
        if not history:
            self.deleted_original_ball_history.pop(key, None)

        if self.mark_annotation_dirty_and_save(key):
            self.status_var.set(
                f"已撤回原始球框删除，标注未保存：{self.annotation_work_path}"
            )
        self.render_current()

    def is_canvas_point_in_bbox(
        self,
        canvas_x: float,
        canvas_y: float,
        bbox_rect: tuple[int, int, int, int],
    ) -> bool:
        x1, y1, x2, y2 = bbox_rect
        return (
            x1 - ANNOTATION_HIT_TOLERANCE <= canvas_x <= x2 + ANNOTATION_HIT_TOLERANCE
            and y1 - ANNOTATION_HIT_TOLERANCE
            <= canvas_y
            <= y2 + ANNOTATION_HIT_TOLERANCE
        )

    def collect_ball_delete_candidates(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> list[tuple[float, str, int, tuple[int, int] | None]]:
        if self.current_raw_frame is None:
            return []

        frame_h, frame_w = self.current_raw_frame.shape[:2]
        scale_x = self.display_width / frame_w
        scale_y = self.display_height / frame_h
        candidates: list[tuple[float, str, int, tuple[int, int] | None]] = []

        manual_key = self.current_annotation_key()
        if manual_key is not None:
            for index, ball in enumerate(
                self.manual_ball_annotations.get(manual_key, [])
            ):
                bbox_rect = scaled_bbox(ball.get("bbox"), scale_x, scale_y)
                if bbox_rect is None:
                    continue
                x1, y1, x2, y2 = bbox_rect
                if self.is_canvas_point_in_bbox(canvas_x, canvas_y, bbox_rect):
                    area = max(1, (x2 - x1) * (y2 - y1))
                    candidates.append((area, "manual", index, manual_key))

        annotation = self.get_annotation()
        json_index = max(
            0,
            min(
                self.current_index + self.json_offset_frames,
                len(annotation.items) - 1,
            ),
        )
        item = annotation.items[json_index]
        original_key = self.annotation_key_from_item(item, self.current_camera)
        if original_key is None:
            return candidates

        result, _missing = result_dict_from_item(item)
        balls, _state = detection_list_from_result(result, "balls")
        deleted_indices = self.deleted_original_ball_indices.get(original_key, set())
        original_index = -1
        for ball in balls:
            if is_manual_ball(ball):
                continue
            original_index += 1
            if original_index in deleted_indices:
                continue
            bbox_rect = scaled_bbox(ball.get("bbox"), scale_x, scale_y)
            if bbox_rect is None:
                continue
            x1, y1, x2, y2 = bbox_rect
            if self.is_canvas_point_in_bbox(canvas_x, canvas_y, bbox_rect):
                area = max(1, (x2 - x1) * (y2 - y1))
                candidates.append((area, "original", original_index, original_key))

        return candidates

    def delete_ball_at_canvas_point(self, canvas_x: float, canvas_y: float) -> None:
        candidates = self.collect_ball_delete_candidates(canvas_x, canvas_y)
        if not candidates:
            self.status_var.set("右键位置没有可删除的球框")
            return

        _area, ball_type, index, key = min(candidates, key=lambda item: item[0])
        if key is None:
            return

        if ball_type == "manual":
            manual_balls = self.manual_ball_annotations.get(key)
            if not manual_balls or index >= len(manual_balls):
                return
            manual_balls.pop(index)
            if manual_balls:
                self.manual_ball_annotations[key] = manual_balls
            else:
                self.manual_ball_annotations.pop(key, None)
            if self.mark_annotation_dirty_and_save(key):
                self.status_var.set(
                    f"已删除人工球框 {index + 1}，标注未保存：{self.annotation_work_path}"
                )
            self.render_current()
            return

        # 不修改原始 item；只记录追踪信息
        self.deleted_original_ball_indices.setdefault(key, set()).add(index)
        self.deleted_original_ball_history.setdefault(key, []).append((index, 0.0))
        if self.mark_annotation_dirty_and_save(key):
            self.status_var.set(
                f"已删除原始球框 {index + 1}（score→0），标注未保存：{self.annotation_work_path}"
            )
        self.render_current()

    def delete_original_ball_at_canvas_point(
        self, canvas_x: float, canvas_y: float
    ) -> None:
        if self.current_raw_frame is None:
            return
        annotation = self.get_annotation()
        json_index = max(
            0,
            min(
                self.current_index + self.json_offset_frames,
                len(annotation.items) - 1,
            ),
        )
        item = annotation.items[json_index]
        key = self.annotation_key_from_item(item, self.current_camera)
        if key is None:
            return

        result, _missing = result_dict_from_item(item)
        balls, _state = detection_list_from_result(result, "balls")
        frame_h, frame_w = self.current_raw_frame.shape[:2]
        scale_x = self.display_width / frame_w
        scale_y = self.display_height / frame_h
        deleted_indices = self.deleted_original_ball_indices.get(key, set())
        candidates: list[tuple[float, int, float]] = []
        original_index = -1
        for ball in balls:
            if is_manual_ball(ball):
                continue
            original_index += 1
            if original_index in deleted_indices:
                continue
            bbox_rect = scaled_bbox(ball.get("bbox"), scale_x, scale_y)
            if bbox_rect is None:
                continue
            x1, y1, x2, y2 = bbox_rect
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                area = max(1, (x2 - x1) * (y2 - y1))
                candidates.append((area, original_index, safe_score(ball.get("score"))))

        if not candidates:
            self.status_var.set("右键位置没有可删除的原始球框")
            return

        _area, original_index, original_score = min(
            candidates, key=lambda item: item[0]
        )
        # 不修改原始 item；只记录追踪信息，在 build_annotation_work_item 中设 score=0
        self.deleted_original_ball_indices.setdefault(key, set()).add(original_index)
        self.deleted_original_ball_history.setdefault(key, []).append(
            (original_index, original_score)
        )

        if self.mark_annotation_dirty_and_save(key):
            self.status_var.set(
                f"已删除原始球框 {original_index + 1}（score→0），标注未保存：{self.annotation_work_path}"
            )
        self.render_current()

    def delete_manual_ball_at_canvas_point(
        self, canvas_x: float, canvas_y: float
    ) -> bool:
        if self.current_raw_frame is None:
            return False
        key = self.current_annotation_key()
        if key is None:
            return False
        manual_balls = self.manual_ball_annotations.get(key)
        if not manual_balls:
            return False

        frame_h, frame_w = self.current_raw_frame.shape[:2]
        scale_x = self.display_width / frame_w
        scale_y = self.display_height / frame_h
        candidates: list[tuple[float, int]] = []
        for index, ball in enumerate(manual_balls):
            bbox_rect = scaled_bbox(ball.get("bbox"), scale_x, scale_y)
            if bbox_rect is None:
                continue
            x1, y1, x2, y2 = bbox_rect
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                area = max(1, (x2 - x1) * (y2 - y1))
                candidates.append((area, index))

        if not candidates:
            return False

        _area, manual_index = min(candidates, key=lambda item: item[0])
        manual_balls.pop(manual_index)
        if manual_balls:
            self.manual_ball_annotations[key] = manual_balls
        else:
            self.manual_ball_annotations.pop(key, None)
        if self.mark_annotation_dirty_and_save(key):
            self.status_var.set(
                f"已删除人工球框 {manual_index + 1}，标注未保存：{self.annotation_work_path}"
            )
        self.render_current()
        return True

    def cancel_annotation_drag(self) -> None:
        self.annotation_drag_start = None
        if self.annotation_preview_rect_id is not None and hasattr(
            self, "video_canvas"
        ):
            self.video_canvas.delete(self.annotation_preview_rect_id)
        self.annotation_preview_rect_id = None

    def get_annotation(self) -> CameraAnnotation:
        annotation = self.store.cameras.get(self.current_camera)
        if annotation is None:
            raise KeyError(f"JSONL 中没有相机 {self.current_camera} 的数据")
        return annotation

    def refresh_camera_choices(self) -> None:
        cameras = [str(camera_id) for camera_id in sorted(self.store.cameras)]
        if self.camera_box is not None:
            self.camera_box.configure(values=cameras)

    def choose_media_dir(self) -> None:
        if self.is_video_mode():
            initial_value = resource_to_text(self.video_dir)
            initialdir = self.video_dir if self.video_dir.is_dir() else PROJECT_DIR
            dialog_title = "选择视频路径"
        else:
            initial_value = resource_to_text(self.frame_image_dir)
            local_image_dir = (
                local_path_from_resource(self.frame_image_dir)
                if not is_remote_resource(self.frame_image_dir)
                else PROJECT_DIR
            )
            initialdir = local_image_dir if local_image_dir.is_dir() else PROJECT_DIR
            dialog_title = "选择图片路径"

        dialog = ResourceDirectoryDialog(
            self.root,
            dialog_title,
            initial_value,
            initialdir,
        )
        if dialog.result is None:
            return
        selected_resource = normalize_local_or_remote_resource(dialog.result)

        self.is_playing = False
        current_pts = self.current_display_pts()

        if self.is_video_mode():
            if is_remote_resource(selected_resource):
                messagebox.showerror("暂不支持", "视频模式目前只支持本地视频目录")
                return
            self.cancel_video_task()
            self.clear_video_cache()
            self.close_video_readers()
            self.video_keyframe_indexes.clear()
            self.video_dir = local_path_from_resource(selected_resource).resolve()
            self.status_var.set(f"已切换视频目录：{self.video_dir}")
        else:
            self.frame_image_dir = selected_resource
            self.clear_image_cache()
            self.status_var.set(f"已切换图片目录：{self.frame_image_dir}")
        self.update_image_prefetch_controls()

        if not self.store.cameras:
            if messagebox.askyesno(
                "选择 JSONL", "尚未加载 JSONL 数据，是否现在选择 JSONL 文件？"
            ):
                self.choose_jsonl()
            return

        self.open_camera(self.current_camera, target_pts=current_pts)

    def choose_jsonl(self) -> None:
        if self.jsonl_loading:
            messagebox.showinfo("正在加载", "JSONL 正在加载中，请稍候")
            return

        if is_remote_resource(self.current_jsonl_path):
            initialdir = PROJECT_DIR
        else:
            current_path = local_path_from_resource(self.current_jsonl_path)
            initialdir = (
                current_path.parent if current_path.parent.is_dir() else PROJECT_DIR
            )

        dialog = ResourceFileDialog(
            self.root,
            "选择 JSONL",
            self.current_jsonl_path,
            initialdir,
        )
        if dialog.result is None:
            return
        selected_resource = normalize_local_or_remote_resource(
            dialog.result, must_be_file=True
        )

        if (
            not is_remote_resource(selected_resource)
            and not Path(selected_resource).is_file()
        ):
            messagebox.showerror("文件不存在", f"找不到文件：{selected_resource}")
            return
        self.start_jsonl_load(selected_resource)

    def load_preset_resources(self) -> None:
        if self.jsonl_loading:
            messagebox.showinfo("正在加载", "JSONL 正在加载中，请稍候")
            return

        PRESETS = [
            {
                "label": "数据集 A（20260526_164647）",
                "image": "https://sense-omni.tos-cn-shanghai.volces.com/hocky/lym/data_saved/0526/label/top100/jpeg/20260526_164647/",
                "jsonl": "https://sense-omni.tos-cn-shanghai.volces.com/hocky/lym/data_saved/0526/label/top100/jsonl/20260526_164647/output_no_players_top100.jsonl",
            },
            {
                "label": "数据集 B（20260526_165340）",
                "image": "https://sense-omni.tos-cn-shanghai.volces.com/hocky/lym/data_saved/0526/label/top100/jpeg/20260526_165340/",
                "jsonl": "https://sense-omni.tos-cn-shanghai.volces.com/hocky/lym/data_saved/0526/label/top100/jsonl/20260526_165340/output_no_players_top100.jsonl",
            },
        ]

        choice = simpledialog.askinteger(
            "选择预设资源",
            "请选择要加载的数据集（输入 1 或 2）：\n\n"
            "1 — 数据集 A（20260526_164647）\n"
            "2 — 数据集 B（20260526_165340）",
            minvalue=1,
            maxvalue=len(PRESETS),
            parent=self.root,
        )
        if choice is None:
            return

        preset = PRESETS[choice - 1]
        self.frame_image_dir = preset["image"]
        self.start_jsonl_load(preset["jsonl"], preset_camera=DEFAULT_CAMERA)

    def update_jsonl_load_button_state(self) -> None:
        if self.cancel_jsonl_load_button is None:
            return
        state = tk.NORMAL if self.jsonl_loading else tk.DISABLED
        self.cancel_jsonl_load_button.configure(state=state)

    def cancel_jsonl_load(self) -> None:
        if not self.jsonl_loading:
            return
        if self.jsonl_cancel_event is not None:
            self.jsonl_cancel_event.set()
        self.jsonl_load_token += 1
        self.jsonl_loading = False
        self.update_jsonl_load_button_state()
        self.status_var.set("JSONL 加载已取消")

    def start_jsonl_load(
        self, jsonl_path: Any, preset_camera: int | None = None
    ) -> None:
        self.is_playing = False
        if self.jsonl_cancel_event is not None:
            self.jsonl_cancel_event.set()
        self.jsonl_load_token += 1
        token = self.jsonl_load_token
        cancel_event = threading.Event()
        self.jsonl_cancel_event = cancel_event
        self.jsonl_loading = True
        jsonl_source = normalize_local_or_remote_resource(jsonl_path, must_be_file=True)
        self.status_var.set(
            f"正在加载 JSONL：{display_resource_name(jsonl_source)} ..."
        )
        self.update_jsonl_load_button_state()

        def progress_cb(pct: int, lines: int, cameras: int) -> None:
            self.jsonl_load_queue.put(
                {
                    "kind": "progress",
                    "token": token,
                    "pct": pct,
                    "lines": lines,
                    "cameras": cameras,
                    "path": str(jsonl_source),
                }
            )

        def worker() -> None:
            try:
                new_store = load_annotations(
                    jsonl_source,
                    progress_callback=progress_cb,
                    cancel_event=cancel_event,
                )
                self.jsonl_load_queue.put(
                    {
                        "kind": "done",
                        "token": token,
                        "store": new_store,
                        "path": jsonl_source,
                        "preset_camera": preset_camera,
                        "error": None,
                        "cancelled": False,
                    }
                )
            except JsonlLoadCancelled:
                self.jsonl_load_queue.put(
                    {
                        "kind": "done",
                        "token": token,
                        "store": None,
                        "path": jsonl_source,
                        "preset_camera": preset_camera,
                        "error": None,
                        "cancelled": True,
                    }
                )
            except Exception as exc:
                self.jsonl_load_queue.put(
                    {
                        "kind": "done",
                        "token": token,
                        "store": None,
                        "path": jsonl_source,
                        "preset_camera": preset_camera,
                        "error": str(exc),
                        "cancelled": False,
                    }
                )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def process_jsonl_load_results(self) -> None:
        try:
            while True:
                result = self.jsonl_load_queue.get_nowait()
                if result.get("token") != self.jsonl_load_token:
                    continue
                kind = result.get("kind")
                if kind == "progress":
                    pct = result["pct"]
                    lines = result["lines"]
                    cameras = result["cameras"]
                    path = result["path"]
                    name = display_resource_name(path)
                    self.status_var.set(
                        f"正在加载 JSONL：{name} | {pct}% | {lines} 行 | {cameras} 路相机"
                    )
                elif kind == "done":
                    self.handle_jsonl_load_done(result)
        except queue.Empty:
            pass

        if self.root.winfo_exists():
            self.root.after(100, self.process_jsonl_load_results)

    def handle_jsonl_load_done(self, result: dict[str, Any]) -> None:
        if result.get("token") != self.jsonl_load_token:
            return
        self.jsonl_loading = False
        self.jsonl_cancel_event = None
        self.update_jsonl_load_button_state()
        if result.get("cancelled"):
            self.status_var.set("JSONL 加载已取消")
            return
        error = result.get("error")
        new_store = result.get("store")
        jsonl_path = result.get("path")
        preset_camera = result.get("preset_camera")

        if error is not None:
            messagebox.showerror("读取 JSONL 失败", str(error))
            self.status_var.set(f"JSONL 加载失败：{error}")
            return
        if new_store is None or not new_store.cameras:
            messagebox.showerror("读取 JSONL 失败", "JSONL 里没有可用相机数据")
            self.status_var.set("JSONL 加载失败：没有可用相机数据")
            return

        self.cancel_video_task()
        self.clear_video_cache()
        self.clear_image_cache()
        self.close_video_readers()
        self.video_keyframe_indexes.clear()
        self.reset_annotation_state()
        self.store = new_store
        self.current_jsonl_path = resource_to_text(jsonl_path)
        self.refresh_camera_choices()

        current_camera = self.current_camera
        if preset_camera is not None and preset_camera in self.store.cameras:
            camera_id = preset_camera
        elif current_camera in self.store.cameras:
            camera_id = current_camera
        else:
            camera_id = sorted(self.store.cameras)[0]
        self.current_camera = camera_id
        self.camera_var.set(str(camera_id))
        self.current_index = 0
        self.current_raw_frame = None

        summary = new_store.summary
        self.status_var.set(
            f"JSONL 加载完成：{display_resource_name(jsonl_path)} | "
            f"{summary.get('total_lines', 0)} 行 | "
            f"{summary.get('camera_count', 0)} 路相机"
        )
        self._update_manual_balls_checkbox_state()
        self.open_camera(camera_id)
        self.schedule_image_prefetch()

    def clear_video_cache(self) -> None:
        self.video_frame_cache.clear()
        self.video_cache_start = 0
        self.video_cache_end = -1
        self.update_cache_status()

    def update_video_cache_bounds(self) -> None:
        if self.video_frame_cache:
            self.video_cache_start = min(self.video_frame_cache)
            self.video_cache_end = max(self.video_frame_cache)
        else:
            self.video_cache_start = 0
            self.video_cache_end = -1

    def trim_video_cache(self, keep_from_index: int | None = None) -> None:
        if keep_from_index is None:
            keep_from_index = max(0, self.current_index - VIDEO_CACHE_KEEP_BEHIND)

        for frame_index in list(self.video_frame_cache):
            if frame_index < keep_from_index:
                del self.video_frame_cache[frame_index]

        max_frames = max(8, self.video_cache_max_frames)
        while len(self.video_frame_cache) > max_frames:
            oldest_index = min(self.video_frame_cache)
            if oldest_index >= self.current_index:
                break
            del self.video_frame_cache[oldest_index]

        self.update_video_cache_bounds()
        self.update_cache_status()

    def merge_video_cache(
        self, cache: dict[int, Any], keep_from_index: int | None = None
    ) -> None:
        self.video_frame_cache.update(cache)
        self.trim_video_cache(keep_from_index)

    def update_cache_status(self) -> None:
        if not hasattr(self, "cache_status_var"):
            return
        if self.is_network_image_mode():
            step = self.cache_frame_step()
            cached_count = sum(
                1
                for camera_id, _idx in self.image_frame_cache
                if camera_id == self.current_camera
            )
            pending_count = sum(
                1
                for camera_id, _idx in self.image_prefetch_pending
                if camera_id == self.current_camera
            )
            total_target = self.image_prefetch_before + self.image_prefetch_after + 1
            target_index = (
                self.image_prefetch_target_index
                if self.image_prefetch_target_index is not None
                else self.current_index
            )
            window_cached = 0
            ahead_cached = 0
            try:
                annotation = self.get_annotation()
            except KeyError:
                annotation = None
            if annotation is not None and annotation.items:
                target_index = max(0, min(target_index, len(annotation.items) - 1))
                target_indices = self.step_aligned_prefetch_indices(
                    target_index,
                    len(annotation.items),
                    self.image_prefetch_before,
                    self.image_prefetch_after,
                )
                window_cached = sum(
                    1
                    for idx in target_indices
                    if (self.current_camera, idx) in self.image_frame_cache
                )
                total_target = len(target_indices)
                probe = self.current_index + step
                while (
                    probe < len(annotation.items)
                    and (self.current_camera, probe) in self.image_frame_cache
                ):
                    ahead_cached += 1
                    probe += step
            parts = [
                f"网络图片已缓存 {cached_count}/{self.image_cache_max_frames} 帧",
                f"当前窗口 {window_cached}/{total_target}",
                f"前向步长连续 {ahead_cached}",
                f"待下载 {pending_count}",
                f"步长 {step}",
                f"范围 前{self.image_prefetch_before}/后{self.image_prefetch_after}",
            ]
            if self.image_prefetching:
                parts.append(
                    f"预取中(中心帧 {self.image_prefetch_center_index + 1 if self.image_prefetch_center_index is not None else '?'})"
                )
            else:
                parts.append("空闲")
            if self.image_prefetch_target_index is not None:
                parts.append(f"目标帧 {self.image_prefetch_target_index + 1}")
            parts.append(f"目标 {total_target} 帧")
            self.cache_status_var.set("缓冲：" + " | ".join(parts))
            return
        if not self.is_video_mode():
            self.cache_status_var.set("缓冲：图片模式不启用")
            return
        step = self.cache_frame_step()
        cached_count = len(self.video_frame_cache)
        ahead = 0
        probe = self.current_index + step
        while probe in self.video_frame_cache:
            ahead += 1
            probe += step
        target = self.video_prefetch_target_ahead
        max_frames = self.video_cache_max_frames
        parts = [
            f"已缓冲 {cached_count}/{max_frames} 帧",
            f"前向步长连续 {ahead}",
            f"目标距离 {target}",
            f"步长 {step}",
        ]
        if self.video_loading:
            parts.append(
                f"主解码中(帧 {self.video_loading_frame_index + 1 if self.video_loading_frame_index is not None else '?'})"
            )
        if self.video_prefetching:
            parts.append(
                f"预取中(起 {self.video_prefetch_start_index + 1 if self.video_prefetch_start_index is not None else '?'})"
            )
        if not self.video_loading and not self.video_prefetching:
            parts.append("空闲")
        self.cache_status_var.set("缓冲：" + " | ".join(parts))

    def apply_cache_target(self) -> None:
        try:
            target = int(self.cache_target_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "缓冲目标帧数必须是整数")
            return
        if target < 8:
            messagebox.showerror("输入错误", "缓冲目标至少 8 帧")
            return
        if target > 4096:
            messagebox.showerror("输入错误", "缓冲目标过大，建议不超过 4096")
            return
        self.video_prefetch_target_ahead = target
        self.video_cache_max_frames = max(
            target + VIDEO_CACHE_KEEP_BEHIND + 16, target * 2
        )
        self.video_prefetch_after = max(VIDEO_PREFETCH_AFTER, target // 2)
        self.status_var.set(
            f"已设置缓冲目标：前向 {target} 帧，缓存上限 {self.video_cache_max_frames} 帧"
        )
        self.update_cache_status()
        self.maybe_prefetch_video()

    def is_network_image_mode(self) -> bool:
        return (not self.is_video_mode()) and is_remote_resource(self.frame_image_dir)

    def cache_frame_step(self) -> int:
        try:
            step = int(self.frame_step_var.get())
        except ValueError:
            return DEFAULT_FRAME_STEP
        return max(1, min(step, 10000))

    def step_aligned_prefetch_indices(
        self,
        center_index: int,
        total_items: int,
        before: int,
        after: int,
    ) -> list[int]:
        if total_items <= 0:
            return []
        step = self.cache_frame_step()
        center_index = max(0, min(center_index, total_items - 1))
        start_index = max(0, center_index - before)
        end_index = min(total_items - 1, center_index + after)

        indices = [center_index]
        indices.extend(range(center_index + step, end_index + 1, step))
        indices.extend(range(center_index - step, start_index - 1, -step))
        return indices

    def next_step_aligned_index_after(self, after_index: int, total_items: int) -> int:
        step = self.cache_frame_step()
        if total_items <= 0:
            return 0
        if after_index < self.current_index:
            return self.current_index
        remainder = (after_index - self.current_index) % step
        delta = step if remainder == 0 else step - remainder
        return after_index + delta

    def update_image_prefetch_controls(self) -> None:
        visible = self.is_network_image_mode()
        for widget in self.image_prefetch_widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()
        self.update_cache_status()
        if visible:
            self.schedule_image_prefetch()

    def schedule_image_prefetch(self) -> None:
        if self.is_network_image_mode() and self.store.cameras:
            self.root.after(
                0, lambda: self.maybe_prefetch_image_frames(self.current_index)
            )

    def apply_image_prefetch_range(self) -> None:
        try:
            before = int(self.image_prefetch_before_var.get())
            after = int(self.image_prefetch_after_var.get())
            max_frames = int(self.image_cache_max_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "图片预载范围和缓存上限必须是整数")
            return

        if before < 0 or after < 0:
            messagebox.showerror("输入错误", "图片预载范围不能小于 0")
            return
        if max_frames < 1:
            messagebox.showerror("输入错误", "图片缓存上限至少 1 帧")
            return
        if before > 1000 or after > 1000:
            messagebox.showerror("输入错误", "图片预载范围过大，建议不超过 1000")
            return
        if max_frames > 4096:
            messagebox.showerror("输入错误", "图片缓存上限过大，建议不超过 4096")
            return

        self.image_prefetch_before = before
        self.image_prefetch_after = after
        self.image_cache_max_frames = max(max_frames, before + after + 1)
        self.image_prefetch_before_var.set(str(before))
        self.image_prefetch_after_var.set(str(after))
        self.image_cache_max_var.set(str(self.image_cache_max_frames))
        self.trim_image_cache_to_limit()
        self.update_cache_status()
        self.maybe_prefetch_image_frames(self.current_index)

    def clear_image_cache(self) -> None:
        self.image_frame_cache.clear()
        self.image_prefetch_pending.clear()
        self.image_pending_render_index = None
        self.cancel_image_prefetch_task()
        self.update_cache_status()

    def clear_current_camera_image_cache(self) -> None:
        for key in list(self.image_frame_cache):
            if key[0] == self.current_camera:
                del self.image_frame_cache[key]
        for key in list(self.image_prefetch_pending):
            if key[0] == self.current_camera:
                self.image_prefetch_pending.remove(key)
        self.update_cache_status()

    def trim_image_cache_to_limit(self, center_index: int | None = None) -> None:
        if center_index is None:
            center_index = self.current_index
        max_frames = max(1, self.image_cache_max_frames)
        if len(self.image_frame_cache) <= max_frames:
            return

        def eviction_score(key: tuple[int, int]) -> tuple[int, int]:
            camera_id, frame_index = key
            different_camera = 1 if camera_id != self.current_camera else 0
            return different_camera, abs(frame_index - center_index)

        keys_by_distance = sorted(
            self.image_frame_cache,
            key=eviction_score,
            reverse=True,
        )
        for key in keys_by_distance:
            if len(self.image_frame_cache) <= max_frames:
                break
            del self.image_frame_cache[key]

    def cancel_image_prefetch_task(self) -> None:
        self.image_prefetch_token += 1
        self.image_prefetching = False
        self.image_prefetch_camera = None
        self.image_prefetch_center_index = None
        self.image_prefetch_target_index = None
        self.image_prefetch_pending.clear()

    def start_image_prefetch_task(self, camera_id: int, center_index: int) -> int:
        self.image_prefetch_token += 1
        self.image_prefetching = True
        self.image_prefetch_camera = camera_id
        self.image_prefetch_center_index = center_index
        self.update_cache_status()
        return self.image_prefetch_token

    def finish_image_prefetch_task(self, token: int) -> bool:
        if token != self.image_prefetch_token:
            return False
        self.image_prefetching = False
        self.image_prefetch_camera = None
        self.image_prefetch_center_index = None
        self.update_cache_status()
        return True

    def is_image_prefetch_current(self, token: int, camera_id: int) -> bool:
        return token == self.image_prefetch_token and camera_id == self.current_camera

    def image_resource_for_frame(self, camera_id: int, pts: int) -> Any:
        return join_resource(self.frame_image_dir, str(camera_id), f"{pts}.jpg")

    def fetch_remote_image_frame(self, image_resource: Any) -> Any | None:
        image_bytes = read_remote_bytes(image_resource)
        return decode_image_bytes(image_bytes)

    def maybe_prefetch_image_frames(self, center_index: int) -> None:
        if not self.is_network_image_mode():
            return
        try:
            annotation = self.get_annotation()
        except KeyError:
            return
        if not annotation.items:
            return

        center_index = max(0, min(center_index, len(annotation.items) - 1))
        target_indices = self.step_aligned_prefetch_indices(
            center_index,
            len(annotation.items),
            self.image_prefetch_before,
            self.image_prefetch_after,
        )
        camera_id = self.current_camera
        cached_indices = [
            idx
            for cached_camera, idx in self.image_frame_cache
            if cached_camera == camera_id
        ]
        target_index_set = set(target_indices)
        has_window_overlap = any(idx in target_index_set for idx in cached_indices)
        if cached_indices and not has_window_overlap:
            self.cancel_image_prefetch_task()
            self.clear_current_camera_image_cache()
        self.image_prefetch_target_index = center_index
        missing_indices = [
            idx
            for idx in target_indices
            if (camera_id, idx) not in self.image_frame_cache
            and (camera_id, idx) not in self.image_prefetch_pending
        ]
        if not missing_indices:
            self.trim_image_cache_to_limit(center_index)
            self.update_cache_status()
            return
        if self.image_prefetching:
            self.update_cache_status()
            return

        frame_image_dir = self.frame_image_dir
        pts_values = annotation.pts_values
        token = self.start_image_prefetch_task(camera_id, center_index)
        pending_keys = {(camera_id, idx) for idx in missing_indices}
        self.image_prefetch_pending.update(pending_keys)
        self.update_cache_status()

        def worker() -> None:
            errors = 0

            def download_one(idx: int) -> dict[str, Any]:
                """下载单帧图片，返回结果字典。stale 表示 token 已过期。"""
                if not self.is_image_prefetch_current(token, camera_id):
                    return {"idx": idx, "frame": None, "error": True, "stale": True}
                pts = pts_values[idx]
                image_resource = join_resource(
                    frame_image_dir, str(camera_id), f"{pts}.jpg"
                )
                try:
                    frame = self.fetch_remote_image_frame(image_resource)
                except Exception:
                    return {"idx": idx, "frame": None, "error": True, "stale": False}
                if frame is not None:
                    return {"idx": idx, "frame": frame, "error": False, "stale": False}
                else:
                    return {"idx": idx, "frame": None, "error": True, "stale": False}

            max_workers = min(IMAGE_CONCURRENT_DOWNLOADS, len(missing_indices))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(download_one, idx) for idx in missing_indices
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result.get("stale"):
                        continue
                    idx = result["idx"]
                    if result.get("error"):
                        errors += 1
                    self.video_result_queue.put(
                        {
                            "kind": "image_frame",
                            "token": token,
                            "camera_id": camera_id,
                            "frame_index": idx,
                            "center_index": center_index,
                            "frame": result.get("frame"),
                            "error": result.get("error", False),
                        }
                    )

            self.video_result_queue.put(
                {
                    "kind": "image_prefetch",
                    "token": token,
                    "camera_id": camera_id,
                    "center_index": center_index,
                    "pending_keys": pending_keys,
                    "errors": errors,
                    "cancelled": False,
                }
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _ensure_image_cache_ahead(self, current_index: int) -> None:
        """播放时主动检查前方缓存余量，不足则触发预取向前扩展窗口。

        与 maybe_prefetch_image_frames 的区别：
        - maybe_prefetch 以 center 为中心检查其前后窗口是否有缺失帧，
          若窗口内缓存已满则直接返回（不做事）。
        - 本方法专门处理「当前窗口已满但前方即将耗尽」的情况，
          计算步长对齐的连续前向缓存帧数，低于阈值时以第一个未缓存的
          位置为中心触发 maybe_prefetch_image_frames。
        """
        if not self.is_network_image_mode():
            return
        if self.image_prefetching:
            return
        try:
            annotation = self.get_annotation()
        except KeyError:
            return
        if not annotation.items:
            return

        step = self.cache_frame_step()
        total = len(annotation.items)

        # 统计从 current_index 起步长对齐的连续已缓存帧数
        ahead = 0
        probe = current_index + step
        while probe < total and (self.current_camera, probe) in self.image_frame_cache:
            ahead += 1
            probe += step

        # 前方连续缓存低于半个预载后窗时，从第一个缺口位置向前补充
        min_ahead = max(1, self.image_prefetch_after // 2)
        if ahead < min_ahead and probe < total:
            self.maybe_prefetch_image_frames(probe)

    def handle_image_frame_result(self, result: dict[str, Any]) -> None:
        token = int(result["token"])
        camera_id = int(result["camera_id"])
        frame_index = int(result["frame_index"])
        if not self.is_image_prefetch_current(token, camera_id):
            return

        key = (camera_id, frame_index)
        self.image_prefetch_pending.discard(key)
        frame = result.get("frame")
        if frame is not None:
            self.image_frame_cache[key] = frame
            self.trim_image_cache_to_limit(
                self.image_prefetch_target_index
                if self.image_prefetch_target_index is not None
                else frame_index
            )
            if (
                self.image_pending_render_index == frame_index
                and camera_id == self.current_camera
            ):
                self.current_index = frame_index
                self.current_raw_frame = frame
                self.image_pending_render_index = None
                self.render_frame(frame)
                self._ensure_image_cache_ahead(frame_index)
        self.update_cache_status()

    def handle_image_prefetch_result(self, result: dict[str, Any]) -> None:
        token = int(result["token"])
        camera_id = int(result["camera_id"])
        center_index = int(result["center_index"])
        if not self.is_image_prefetch_current(token, camera_id):
            return
        self.finish_image_prefetch_task(token)
        pending_keys = result.get("pending_keys") or set()
        self.image_prefetch_pending.difference_update(pending_keys)
        self.trim_image_cache_to_limit(center_index)
        self.update_cache_status()
        next_target = (
            self.image_prefetch_target_index
            if self.image_prefetch_target_index is not None
            else self.current_index
        )
        self.maybe_prefetch_image_frames(next_target)
        self._ensure_image_cache_ahead(self.current_index)

    def close_video_readers(self) -> None:
        for reader in self.video_readers.values():
            reader.close()
        self.video_readers.clear()

    def close_video_reader(self, camera_id: int) -> None:
        reader = self.video_readers.pop(camera_id, None)
        if reader is not None:
            reader.close()

    def resolve_video_path(self, camera_id: int) -> Path | None:
        m3u8_path = self.video_dir / f"{camera_id}.m3u8"
        if m3u8_path.is_file():
            return m3u8_path
        ts_path = self.video_dir / f"{camera_id}.ts"
        if ts_path.is_file():
            return ts_path
        return None

    def start_video_task(self, camera_id: int, frame_index: int) -> int:
        self.video_task_token += 1
        self.video_loading = True
        self.video_loading_camera = camera_id
        self.video_loading_frame_index = frame_index
        self.status_var.set(
            f"视频模式 | 相机 {camera_id} | 正在后台解码第 {frame_index + 1} 帧..."
        )
        return self.video_task_token

    def cancel_video_task(self) -> None:
        self.video_task_token += 1
        self.video_loading = False
        self.video_loading_camera = None
        self.video_loading_frame_index = None
        self.cancel_video_prefetch_task()

    def finish_video_task(self, token: int) -> bool:
        if token != self.video_task_token:
            return False
        self.video_loading = False
        self.video_loading_camera = None
        self.video_loading_frame_index = None
        return True

    def is_video_task_current(self, token: int, camera_id: int) -> bool:
        return token == self.video_task_token and camera_id == self.current_camera

    def start_video_prefetch_task(self, camera_id: int, start_index: int) -> int:
        self.video_prefetch_token += 1
        self.video_prefetching = True
        self.video_prefetch_camera = camera_id
        self.video_prefetch_start_index = start_index
        return self.video_prefetch_token

    def cancel_video_prefetch_task(self) -> None:
        self.video_prefetch_token += 1
        self.video_prefetching = False
        self.video_prefetch_camera = None
        self.video_prefetch_start_index = None

    def finish_video_prefetch_task(self, token: int) -> bool:
        if token != self.video_prefetch_token:
            return False
        self.video_prefetching = False
        self.video_prefetch_camera = None
        self.video_prefetch_start_index = None
        return True

    def is_video_prefetch_current(self, token: int, camera_id: int) -> bool:
        return token == self.video_prefetch_token and camera_id == self.current_camera

    def process_video_results(self) -> None:
        try:
            while True:
                result = self.video_result_queue.get_nowait()
                kind = result.get("kind")
                if kind == "main":
                    self.handle_main_video_result(result)
                elif kind == "prefetch":
                    self.handle_prefetch_video_result(result)
                elif kind == "image_frame":
                    self.handle_image_frame_result(result)
                elif kind == "image_prefetch":
                    self.handle_image_prefetch_result(result)
                elif kind == "annotation_save":
                    self.handle_annotation_save_result(result)
                elif kind == "annotation_save_progress":
                    self.handle_annotation_save_progress(result)
                elif kind == "annotation_work_progress":
                    self.handle_annotation_work_progress(result)
                elif kind == "annotation_work_done":
                    self.handle_annotation_work_done(result)
        except queue.Empty:
            pass

        if self.root.winfo_exists():
            self.root.after(VIDEO_RESULT_POLL_MS, self.process_video_results)

    def handle_main_video_result(self, result: dict[str, Any]) -> None:
        token = int(result["token"])
        camera_id = int(result["camera_id"])
        frame_index = int(result["frame_index"])
        reader = result.get("reader")
        key_index = result.get("key_index")
        cache = result.get("cache") or {}
        frame = result.get("frame")
        error = result.get("error")

        if not self.is_video_task_current(token, camera_id):
            if reader is not None and self.video_readers.get(camera_id) is not reader:
                reader.close()
            return

        self.finish_video_task(token)

        if error is not None:
            self.status_var.set(f"视频模式解码失败：{error}")
            return

        if key_index is not None:
            self.video_keyframe_indexes[camera_id] = key_index
        if reader is not None:
            self.video_readers[camera_id] = reader
        self.merge_video_cache(
            cache, keep_from_index=max(0, frame_index - VIDEO_CACHE_KEEP_BEHIND)
        )

        if frame is None:
            self.status_var.set(
                f"读取视频帧失败：camera={camera_id}, frame={frame_index + 1}"
            )
            return

        self.current_index = frame_index
        self.current_raw_frame = frame
        self.render_frame(frame)
        self.maybe_prefetch_image_frames(frame_index)
        self.maybe_prefetch_video()

    def handle_prefetch_video_result(self, result: dict[str, Any]) -> None:
        token = int(result["token"])
        camera_id = int(result["camera_id"])
        reader = result.get("reader")
        key_index = result.get("key_index")
        cache = result.get("cache") or {}
        error = result.get("error")

        if not self.is_video_prefetch_current(token, camera_id):
            if reader is not None and self.video_readers.get(camera_id) is not reader:
                reader.close()
            return

        self.finish_video_prefetch_task(token)

        if error is not None:
            return

        if key_index is not None:
            self.video_keyframe_indexes[camera_id] = key_index
        if reader is not None:
            self.video_readers[camera_id] = reader
        if cache:
            self.merge_video_cache(cache)
        self.maybe_prefetch_video()

    def get_video_keyframe_index(self) -> VideoKeyframeIndex:
        key_index = self.video_keyframe_indexes.get(self.current_camera)
        if key_index is None:
            raise RuntimeError(f"相机 {self.current_camera} 的关键帧索引尚未建立")
        return key_index

    def get_video_reader(self) -> PyAvVideoFrameReader:
        reader = self.video_readers.get(self.current_camera)
        if reader is not None:
            return reader

        playlist_path = self.resolve_video_path(self.current_camera)
        if playlist_path is None:
            raise FileNotFoundError(f"找不到相机 {self.current_camera} 的视频文件")
        annotation = self.get_annotation()
        key_index = self.get_video_keyframe_index()
        reader = PyAvVideoFrameReader(playlist_path, annotation, key_index)
        self.video_readers[self.current_camera] = reader
        return reader

    def decode_video_cache_window(self, target_frame_index: int) -> Any | None:
        self.clear_video_cache()
        try:
            reader = self.get_video_reader()
            step = self.cache_frame_step()
            self.video_frame_cache = reader.decode_cache_window(
                target_frame_index,
                before=VIDEO_CACHE_BEFORE,
                after=VIDEO_CACHE_AFTER,
                max_frames=self.video_cache_max_frames,
                step=step,
            )
        except Exception as exc:
            self.status_var.set(f"PyAV 解码失败：{exc}")
            return None

        if not self.video_frame_cache:
            return None

        self.video_cache_start = min(self.video_frame_cache)
        self.video_cache_end = max(self.video_frame_cache)
        self.update_cache_status()
        return self.video_frame_cache.get(target_frame_index)

    def seek_and_render_video_async(self, frame_index: int) -> None:
        annotation = self.get_annotation()
        frame_index = max(0, min(frame_index, len(annotation.items) - 1))
        camera_id = self.current_camera
        playlist_path = self.resolve_video_path(camera_id)
        if playlist_path is None:
            messagebox.showerror(
                "视频不存在", f"找不到相机 {camera_id} 的视频文件（.m3u8 或 .ts）"
            )
            return

        cached_frame = self.video_frame_cache.get(frame_index)
        if cached_frame is not None:
            self.current_index = frame_index
            self.current_raw_frame = cached_frame
            self.render_frame(cached_frame)
            self.maybe_prefetch_video()
            return

        max_frames = self.video_cache_max_frames
        cache_step = self.cache_frame_step()
        token = self.start_video_task(camera_id, frame_index)
        self.update_cache_status()

        def worker() -> None:
            try:
                with self.video_decode_lock:
                    if not self.is_video_task_current(token, camera_id):
                        return

                    if camera_id not in self.video_keyframe_indexes:
                        annotation_local = self.store.cameras[camera_id]
                        key_index = build_video_keyframe_index(
                            playlist_path, annotation_local
                        )
                    else:
                        key_index = self.video_keyframe_indexes[camera_id]

                    if not self.is_video_task_current(token, camera_id):
                        return

                    reader = self.video_readers.get(camera_id)
                    if reader is None:
                        annotation_local = self.store.cameras[camera_id]
                        reader = PyAvVideoFrameReader(
                            playlist_path, annotation_local, key_index
                        )
                    else:
                        reader.keyframe_index = key_index
                        reader.annotation = self.store.cameras[camera_id]

                    cache = reader.decode_cache_window(
                        frame_index,
                        before=VIDEO_CACHE_BEFORE,
                        after=VIDEO_CACHE_AFTER,
                        max_frames=max_frames,
                        step=cache_step,
                    )
                frame = cache.get(frame_index)
                error = None if frame is not None else "解码结果里没有目标帧"
            except Exception as exc:
                key_index = None
                reader = None
                cache = {}
                frame = None
                error = str(exc)

            self.video_result_queue.put(
                {
                    "kind": "main",
                    "token": token,
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "reader": reader,
                    "key_index": key_index,
                    "cache": cache,
                    "frame": frame,
                    "error": error,
                }
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def maybe_prefetch_video(self) -> None:
        if not self.is_video_mode():
            return
        if self.video_loading or self.video_prefetching:
            return

        annotation = self.get_annotation()
        if not annotation.items:
            return

        if self.video_cache_end < self.current_index:
            return

        cached_ahead = self.video_cache_end - self.current_index
        if cached_ahead >= self.video_prefetch_target_ahead:
            self.update_cache_status()
            return

        start_index = self.next_step_aligned_index_after(
            self.video_cache_end,
            len(annotation.items),
        )
        if start_index >= len(annotation.items):
            return

        camera_id = self.current_camera
        playlist_path = self.resolve_video_path(camera_id)
        if playlist_path is None:
            return

        prefetch_after = self.video_prefetch_after
        max_frames = self.video_cache_max_frames
        cache_step = self.cache_frame_step()

        token = self.start_video_prefetch_task(camera_id, start_index)
        self.update_cache_status()

        def worker() -> None:
            try:
                with self.video_decode_lock:
                    if not self.is_video_prefetch_current(token, camera_id):
                        return

                    if camera_id not in self.video_keyframe_indexes:
                        annotation_local = self.store.cameras[camera_id]
                        key_index = build_video_keyframe_index(
                            playlist_path, annotation_local
                        )
                    else:
                        key_index = self.video_keyframe_indexes[camera_id]

                    if not self.is_video_prefetch_current(token, camera_id):
                        return

                    reader = self.video_readers.get(camera_id)
                    if reader is None:
                        annotation_local = self.store.cameras[camera_id]
                        reader = PyAvVideoFrameReader(
                            playlist_path, annotation_local, key_index
                        )
                    else:
                        reader.keyframe_index = key_index
                        reader.annotation = self.store.cameras[camera_id]

                    cache = reader.decode_cache_window(
                        start_index,
                        before=0,
                        after=prefetch_after,
                        max_frames=max_frames,
                        step=cache_step,
                    )
                error = None
            except Exception as exc:
                key_index = None
                reader = None
                cache = {}
                error = str(exc)

            self.video_result_queue.put(
                {
                    "kind": "prefetch",
                    "token": token,
                    "camera_id": camera_id,
                    "reader": reader,
                    "key_index": key_index,
                    "cache": cache,
                    "error": error,
                }
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

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
        if not self.confirm_unsaved_annotation_navigation(
            "切帧",
            "seek",
            frame_number - 1,
        ):
            return
        self.seek_and_render(frame_number - 1)

    def open_camera(self, camera_id: int, target_pts: int | None = None) -> None:
        if (
            camera_id != self.current_camera
            and not self.confirm_unsaved_annotation_navigation(
                "切相机",
                "open_camera",
                camera_id,
                target_pts,
            )
        ):
            return
        self.is_playing = False
        self.cancel_video_task()
        if not self.store.cameras:
            self.status_var.set("尚未加载 JSONL 数据")
            return
        if camera_id not in self.store.cameras:
            messagebox.showerror("相机不存在", f"JSONL 中没有相机 {camera_id} 的数据")
            return

        if self.is_video_mode():
            playlist_path = self.resolve_video_path(camera_id)
            if playlist_path is None:
                messagebox.showerror(
                    "视频不存在", f"找不到相机 {camera_id} 的视频文件（.m3u8 或 .ts）"
                )
                return

        if camera_id != self.current_camera:
            self.close_video_reader(self.current_camera)

        self.current_camera = camera_id
        self.camera_var.set(str(camera_id))
        self.current_raw_frame = None
        self.clear_image_cache()
        self.clear_video_cache()
        self.update_image_prefetch_controls()

        if not self.is_video_mode():
            self.close_video_reader(camera_id)

        annotation = self.get_annotation()
        self.update_progress_range()
        if target_pts is None:
            frame_index = max(0, min(self.current_index, len(annotation.items) - 1))
        else:
            frame_index = nearest_index_by_pts(annotation, target_pts)
        self.seek_and_render(frame_index)
        self.schedule_image_prefetch()

    def is_video_mode(self) -> bool:
        return self.mode_var.get() == INPUT_MODE_VIDEO

    def change_mode(self) -> None:
        if self.is_annotation_work_mode() and self.mode_var.get() != INPUT_MODE_IMAGE:
            self.mode_var.set(INPUT_MODE_IMAGE)
            messagebox.showinfo("标注模式", "标注模式仅支持图片模式")
            return
        if not self.confirm_unsaved_annotation_navigation(
            "切换模式",
            "change_mode",
        ):
            return
        self.is_playing = False
        self.current_raw_frame = None
        self.clear_video_cache()
        self.clear_image_cache()
        if not self.is_video_mode():
            self.cancel_video_task()
            self.close_video_readers()
        self.update_choose_dir_button_label()
        self.update_image_prefetch_controls()
        self.apply_work_mode_visibility(render=False)
        current_pts = self.current_display_pts()
        self.open_camera(self.current_camera, target_pts=current_pts)
        self.schedule_image_prefetch()

    def change_camera(self) -> None:
        try:
            camera_id = int(self.camera_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "相机编号必须是整数")
            return
        current_pts = self.current_display_pts()
        self.open_camera(camera_id, target_pts=current_pts)

    def get_frame_step(self) -> int:
        try:
            step = int(self.frame_step_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "步长必须是正整数")
            self.frame_step_var.set(str(DEFAULT_FRAME_STEP))
            return DEFAULT_FRAME_STEP
        if step < 1:
            messagebox.showerror("输入错误", "步长必须大于等于 1")
            self.frame_step_var.set(str(DEFAULT_FRAME_STEP))
            return DEFAULT_FRAME_STEP
        if step > 10000:
            messagebox.showerror("输入错误", "步长过大，建议不超过 10000")
            step = 10000
        self.frame_step_var.set(str(step))
        return step

    def toggle_play(self) -> None:
        if not self.is_video_mode():
            # 图片模式下也允许自动翻页，只是本质上是连续切图。
            pass
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.last_tick_time = time.time()
            self.play_tick()
        else:
            self.maybe_prefetch_image_frames(self.current_index)

    def play_tick(self) -> None:
        if not self.is_playing:
            return

        next_index = self.current_index + 1
        if self.is_network_image_mode():
            annotation = self.get_annotation()
            next_index = max(0, min(next_index, len(annotation.items) - 1))
            if (self.current_camera, next_index) not in self.image_frame_cache:
                self.maybe_prefetch_image_frames(next_index)
                self.root.after(PLAY_INTERVAL_MS, self.play_tick)
                return
        if self.video_loading and (
            not self.is_video_mode() or next_index not in self.video_frame_cache
        ):
            self.root.after(PLAY_INTERVAL_MS, self.play_tick)
            return

        self.seek_and_render(next_index)
        self._ensure_image_cache_ahead(self.current_index)
        self.root.after(PLAY_INTERVAL_MS, self.play_tick)

    def prev_frame(self) -> None:
        self.is_playing = False
        target_index = self.current_index - self.get_frame_step()
        if not self.confirm_unsaved_annotation_navigation("切帧", "seek", target_index):
            return
        self.seek_and_render(target_index)

    def first_frame(self) -> None:
        self.is_playing = False
        if not self.confirm_unsaved_annotation_navigation("切帧", "seek", 0):
            return
        self.seek_and_render(0)

    def next_frame(self) -> None:
        self.is_playing = False
        target_index = self.current_index + self.get_frame_step()
        if not self.confirm_unsaved_annotation_navigation("切帧", "seek", target_index):
            return
        self.seek_and_render(target_index)

    def jump_frame(self) -> None:
        self.is_playing = False
        try:
            frame_number = int(self.frame_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "帧序号必须是整数")
            return
        target_index = frame_number - 1
        if not self.confirm_unsaved_annotation_navigation("切帧", "seek", target_index):
            return
        self.seek_and_render(target_index)

    def jump_pts(self) -> None:
        self.is_playing = False
        try:
            pts = int(self.pts_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "PTS 必须是整数")
            return
        annotation = self.get_annotation()
        target_index = nearest_index_by_pts(annotation, pts)
        if not self.confirm_unsaved_annotation_navigation("切帧", "seek", target_index):
            return
        self.seek_and_render(target_index)

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
        scale = min(max((width_scale + height_scale) / 2, 0.25), 2.0)
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
        if is_detail_mode or not self.is_inspect_work_mode():
            self.threshold_row.grid_remove()
        else:
            self.threshold_row.grid()

    def _update_manual_balls_checkbox_state(self) -> None:
        """根据当前 JSONL 是否包含 score=2 的球来启用/禁用 manual_balls 复选框。"""
        if self.manual_balls_checkbox is None:
            return
        has = self.store.summary.get("has_manual_balls", False)
        self.manual_balls_checkbox.configure(state=tk.NORMAL if has else tk.DISABLED)
        if not has:
            self.show_manual_balls.set(False)

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
        target_index = self.current_index + step
        if self.video_loading and (
            not self.is_video_mode() or target_index not in self.video_frame_cache
        ):
            return
        self.seek_and_render(target_index)

    def seek_and_render(self, frame_index: int) -> None:
        if (
            self.annotation_mode_enabled
            and self.annotation_dirty
            and self.annotation_save_after_id is None
            and not self.annotation_saving
        ):
            self.start_annotation_save_worker()
        annotation = self.get_annotation()
        if not annotation.items:
            return

        frame_index = max(0, min(frame_index, len(annotation.items) - 1))

        if self.is_video_mode():
            self.seek_and_render_video_async(frame_index)
            return

        if (
            self.is_network_image_mode()
            and (
                self.current_camera,
                frame_index,
            )
            not in self.image_frame_cache
        ):
            self.image_pending_render_index = frame_index
            self.maybe_prefetch_image_frames(frame_index)
            pts = annotation.pts_values[frame_index]
            self.status_var.set(
                f"等待网络图片缓存：camera={self.current_camera}, frame={frame_index + 1}, pts={pts}"
            )
            self.update_cache_status()
            return

        frame = self.read_frame_data(frame_index)
        if frame is None:
            self.status_var.set(
                f"读取图片帧失败：camera={self.current_camera}, frame={frame_index + 1}"
            )
            return

        self.current_index = frame_index
        self.current_raw_frame = frame
        self.render_frame(frame)
        self._ensure_image_cache_ahead(frame_index)

    def render_current(self) -> None:
        if self.video_loading and self.current_raw_frame is None:
            return
        if self.current_raw_frame is not None:
            self.render_frame(self.current_raw_frame)
        else:
            self.seek_and_render(self.current_index)

    def read_video_frame(self, frame_index: int) -> Any | None:
        cached_frame = self.video_frame_cache.get(frame_index)
        if cached_frame is not None:
            return cached_frame
        return self.decode_video_cache_window(frame_index)

    def read_image_frame(self, frame_index: int) -> Any | None:
        annotation = self.get_annotation()
        pts = annotation.pts_values[frame_index]
        image_resource = join_resource(
            self.frame_image_dir,
            str(self.current_camera),
            f"{pts}.jpg",
        )

        if is_remote_resource(image_resource):
            cached_frame = self.image_frame_cache.get(
                (self.current_camera, frame_index)
            )
            if cached_frame is not None:
                return cached_frame
            try:
                frame = self.fetch_remote_image_frame(image_resource)
            except Exception as exc:
                self.show_json(
                    {
                        "error": "读取网络图片帧失败",
                        "camera": self.current_camera,
                        "frame_index": frame_index,
                        "pts": pts,
                        "url": resource_to_text(image_resource),
                        "request_url": resource_to_request_url(image_resource),
                        "exception": str(exc),
                    }
                )
                return None
            if frame is None:
                self.show_json(
                    {
                        "error": "网络图片解码失败",
                        "camera": self.current_camera,
                        "frame_index": frame_index,
                        "pts": pts,
                        "url": resource_to_text(image_resource),
                    }
                )
                return None
            self.image_frame_cache[(self.current_camera, frame_index)] = frame
            self.trim_image_cache_to_limit(frame_index)
            return frame

        image_path = Path(image_resource)
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
            return self.read_video_frame(frame_index)
        return self.read_image_frame(frame_index)

    def render_frame(self, frame: Any) -> None:
        annotation = self.get_annotation()
        json_index = max(
            0,
            min(
                self.current_index + self.json_offset_frames, len(annotation.items) - 1
            ),
        )
        item = annotation.items[json_index]
        item_annotation_key = self.annotation_key_from_item(item, self.current_camera)
        deleted_ball_indices = (
            self.deleted_original_ball_indices.get(item_annotation_key, set())
            if item_annotation_key is not None
            else set()
        )
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

        if self.is_view_work_mode():
            draw_players = False
            draw_balls = False
            draw_keypoints = False
            draw_manual_balls = False
        elif self.is_annotation_work_mode():
            draw_players = False
            draw_balls = True
            draw_keypoints = False
            draw_manual_balls = False
        else:
            draw_players = self.show_players.get()
            draw_balls = self.show_balls.get()
            draw_keypoints = self.show_keypoints.get()
            draw_manual_balls = self.show_manual_balls.get()

        draw_detections(
            display_frame,
            item,
            show_players=draw_players,
            show_balls=draw_balls,
            show_keypoints=draw_keypoints,
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
            deleted_ball_indices=deleted_ball_indices,
            show_manual_balls=draw_manual_balls,
        )
        if not self.is_view_work_mode() and (
            draw_balls or self.annotation_mode_enabled or self.is_annotation_work_mode()
        ):
            draw_manual_ball_annotations(
                display_frame,
                self.current_frame_manual_balls(),
                scale_x,
                scale_y,
                label_scale,
            )
        if self.is_inspect_work_mode():
            self.draw_overlay_header(
                display_frame, item, pts, json_index, json_pts, label_scale
            )
        self.show_frame(display_frame)
        annotation_json_item = (
            self.build_annotation_work_item(
                item,
                self.current_camera,
                self.manual_ball_annotations,
                self.deleted_original_ball_indices,
            )
            if self.is_annotation_work_mode()
            else None
        )
        # item 来自原始 JSONL，从未被标注删除操作修改；
        # build_annotation_work_item 在 deepcopy 上设 score=0，不影响 item
        self.show_json(item, annotation_json_item)

        result, result_missing = result_dict_from_item(item)
        players_text = detection_count_text(result, "players", "players")
        balls_text = detection_count_text(result, "balls", "balls")
        schema_status = "result 缺失 | " if result_missing else ""
        manual_count = len(self.current_frame_manual_balls())
        deleted_indices = self.deleted_original_ball_indices.get(
            self.current_annotation_key(), set()
        )
        deleted_count = len(deleted_indices)
        annotation_status = ""
        if self.annotation_mode_enabled:
            work_name = (
                display_resource_name(self.annotation_work_path)
                if self.annotation_work_path is not None
                else "未创建"
            )
            save_state = "保存中" if self.annotation_saving else "已保存"
            if self.annotation_dirty:
                save_state = "待保存" if not self.annotation_saving else "保存中"
            annotation_status = (
                f"人工球框 {manual_count} | 删除原框 {deleted_count} | "
                f"标注{save_state} | 标注文件 {work_name} | "
            )
        threshold_status = ""
        if self.label_mode_var.get() == "排错模式":
            threshold_status = (
                f"阈值 det={self.current_det_threshold:.2f}, "
                f"team_s={self.current_team_score_threshold:.2f}, "
                f"id_s={self.current_id_score_threshold:.2f} | "
            )
        video_cache_status = ""
        if self.is_video_mode():
            prefetch_status = "预读中 | " if self.video_prefetching else ""
            video_cache_status = f"视频缓存 {self.video_cache_start + 1}-{self.video_cache_end + 1} | {prefetch_status}"
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
            f"{schema_status}{players_text} | {balls_text} | "
            f"{annotation_status}"
            f"原始尺寸 {frame_w}x{frame_h} -> 显示尺寸 {self.display_width}x{self.display_height} | "
            f"标签 {self.label_mode_var.get()} | "
            f"{threshold_status}"
            f"{video_cache_status}"
            f"文字 {label_scale:.2f}"
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
        result, result_missing = result_dict_from_item(item)
        players_text = detection_count_text(result, "players", "players")
        balls_text = detection_count_text(result, "balls", "balls")
        result_text = "result missing  " if result_missing else ""
        text = (
            f"camera {self.current_camera}  "
            f"cam_idx {field_text(result, 'cam_idx')}  "
            f"frame {self.current_index + 1}  "
            f"pts {pts}  "
            f"json_offset {self.json_offset_frames:+d}  "
            f"json_frame {json_index + 1}  "
            f"json_pts {json_pts}  "
            f"{result_text}"
            f"{players_text}  "
            f"{balls_text}"
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
        xview = (
            self.video_canvas.xview() if hasattr(self, "video_canvas") else (0.0, 1.0)
        )
        yview = (
            self.video_canvas.yview() if hasattr(self, "video_canvas") else (0.0, 1.0)
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.current_frame_image = photo
        if self.canvas_image_id is None:
            self.canvas_image_id = self.video_canvas.create_image(
                0, 0, image=photo, anchor="nw"
            )
        else:
            self.video_canvas.itemconfigure(self.canvas_image_id, image=photo)
        self.video_canvas.configure(scrollregion=(0, 0, frame.shape[1], frame.shape[0]))
        self.video_canvas.xview_moveto(xview[0])
        self.video_canvas.yview_moveto(yview[0])

    def on_canvas_double_click(self, event: tk.Event) -> None:
        """标注模式下双击左键进入/退出画面平移模式。"""
        if not self.annotation_mode_enabled:
            return
        if self.annotation_pan_mode:
            self.annotation_pan_mode = False
            self.cancel_annotation_drag()
            self.video_canvas.configure(cursor="crosshair")
            self.status_var.set("已退出平移模式，恢复标注")
        else:
            self.annotation_pan_mode = True
            self.cancel_annotation_drag()
            self.video_canvas.scan_mark(event.x, event.y)
            self.video_canvas.configure(cursor="fleur")
            self.status_var.set("双击平移模式：拖动左键移动画面，双击恢复标注")

    def on_canvas_press(self, event: tk.Event) -> None:
        if self.annotation_mode_enabled and self.annotation_pan_mode:
            self.video_canvas.scan_mark(event.x, event.y)
            return
        if self.annotation_mode_enabled:
            if self.current_raw_frame is None:
                return
            x = self.video_canvas.canvasx(event.x)
            y = self.video_canvas.canvasy(event.y)
            x = max(0.0, min(float(self.display_width), float(x)))
            y = max(0.0, min(float(self.display_height), float(y)))
            self.annotation_drag_start = (x, y)
            if self.annotation_preview_rect_id is not None:
                self.video_canvas.delete(self.annotation_preview_rect_id)
            self.annotation_preview_rect_id = self.video_canvas.create_rectangle(
                x,
                y,
                x,
                y,
                outline="#ff00ff",
                width=2,
            )
            return
        self.video_canvas.scan_mark(event.x, event.y)

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.annotation_mode_enabled and self.annotation_pan_mode:
            self.video_canvas.scan_dragto(event.x, event.y, gain=1)
            return
        if self.annotation_mode_enabled:
            if (
                self.annotation_drag_start is None
                or self.annotation_preview_rect_id is None
            ):
                return
            x = self.video_canvas.canvasx(event.x)
            y = self.video_canvas.canvasy(event.y)
            x = max(0.0, min(float(self.display_width), float(x)))
            y = max(0.0, min(float(self.display_height), float(y)))
            start_x, start_y = self.annotation_drag_start
            self.video_canvas.coords(
                self.annotation_preview_rect_id,
                start_x,
                start_y,
                x,
                y,
            )
            return
        self.video_canvas.scan_dragto(event.x, event.y, gain=1)

    def on_canvas_release(self, event: tk.Event) -> None:
        if not self.annotation_mode_enabled:
            return
        if self.annotation_pan_mode:
            return
        if self.annotation_drag_start is None or self.current_raw_frame is None:
            self.cancel_annotation_drag()
            return

        start_x, start_y = self.annotation_drag_start
        end_x = self.video_canvas.canvasx(event.x)
        end_y = self.video_canvas.canvasy(event.y)
        end_x = max(0.0, min(float(self.display_width), float(end_x)))
        end_y = max(0.0, min(float(self.display_height), float(end_y)))
        self.cancel_annotation_drag()

        display_x1, display_x2 = sorted((start_x, end_x))
        display_y1, display_y2 = sorted((start_y, end_y))
        if (
            display_x2 - display_x1 < MIN_ANNOTATION_BOX_SIZE
            or display_y2 - display_y1 < MIN_ANNOTATION_BOX_SIZE
        ):
            self.status_var.set("标注框太小，已忽略")
            return

        frame_h, frame_w = self.current_raw_frame.shape[:2]
        scale_x = self.display_width / frame_w
        scale_y = self.display_height / frame_h
        bbox = [
            int(round(display_x1 / scale_x)),
            int(round(display_y1 / scale_y)),
            int(round(display_x2 / scale_x)),
            int(round(display_y2 / scale_y)),
        ]
        bbox[0] = max(0, min(frame_w - 1, bbox[0]))
        bbox[1] = max(0, min(frame_h - 1, bbox[1]))
        bbox[2] = max(0, min(frame_w - 1, bbox[2]))
        bbox[3] = max(0, min(frame_h - 1, bbox[3]))
        if (
            bbox[2] - bbox[0] < MIN_ANNOTATION_BOX_SIZE
            or bbox[3] - bbox[1] < MIN_ANNOTATION_BOX_SIZE
        ):
            self.status_var.set("标注框太小，已忽略")
            return

        key = self.current_annotation_key()
        if key is None:
            return
        ball = {
            "bbox": bbox,
            "score": MANUAL_BALL_SCORE,
        }
        self.manual_ball_annotations.setdefault(key, []).append(ball)
        if self.mark_annotation_dirty_and_save(key):
            self.status_var.set(
                f"已新增人工球框，标注未保存：{self.annotation_work_path}"
            )
        self.render_current()

    def on_canvas_right_click(self, event: tk.Event) -> None:
        if not self.annotation_mode_enabled:
            return
        x = self.video_canvas.canvasx(event.x)
        y = self.video_canvas.canvasy(event.y)
        x = max(0.0, min(float(self.display_width), float(x)))
        y = max(0.0, min(float(self.display_height), float(y)))
        self.delete_ball_at_canvas_point(x, y)

    def on_canvas_mousewheel(self, event: tk.Event) -> None:
        units = int(-1 * (event.delta / 120)) if event.delta else 0
        if units == 0:
            units = -1 if event.delta > 0 else 1
        if event.state & 0x0001:
            self.video_canvas.xview_scroll(units, "units")
        else:
            self.video_canvas.yview_scroll(units, "units")

    def show_json(
        self,
        item: dict[str, Any],
        annotation_item: dict[str, Any] | None = None,
    ) -> None:
        self.json_text.configure(state=tk.NORMAL)
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", json.dumps(item, ensure_ascii=False, indent=2))
        self.json_text.configure(state=tk.DISABLED)
        if self.annotation_json_text is not None:
            self.annotation_json_text.configure(state=tk.NORMAL)
            self.annotation_json_text.delete("1.0", tk.END)
            if annotation_item is not None:
                self.annotation_json_text.insert(
                    "1.0",
                    json.dumps(annotation_item, ensure_ascii=False, indent=2),
                )
            self.annotation_json_text.configure(state=tk.DISABLED)

    def close(self) -> None:
        if self.annotation_dirty:
            answer = messagebox.askyesnocancel(
                "标注尚未保存",
                "当前标注 JSONL 有未保存改动，关闭页面前是否保存？\n\n"
                "选择「是」：保存完成后关闭。\n"
                "选择「否」：不保存并直接关闭。\n"
                "选择「取消」：返回页面。",
            )
            if answer is None:
                return
            if answer:
                self.close_after_annotation_save = True
                if self.annotation_save_after_id is not None:
                    self.root.after_cancel(self.annotation_save_after_id)
                    self.annotation_save_after_id = None
                if self.annotation_saving:
                    self.status_var.set(
                        "标注 JSONL 正在保存中，保存完成后会自动关闭页面"
                    )
                else:
                    self.start_annotation_save_worker()
                return
        self.close_without_annotation_prompt()

    def close_without_annotation_prompt(self) -> None:
        if self.jsonl_cancel_event is not None:
            self.jsonl_cancel_event.set()
        self.cancel_image_prefetch_task()
        self.cancel_video_task()
        self.close_video_readers()
        self.root.destroy()


def pause_before_exit(code: int = 0) -> None:
    if sys.platform == "win32" and sys.stdin.isatty():
        try:
            input("\n按 Enter 键退出...")
        except EOFError:
            pass
    raise SystemExit(code)


def main() -> None:
    try:
        args = parse_args()
        require_dependencies()

        video_dir = args.video_dir.expanduser().resolve()
        frame_image_dir = normalize_local_or_remote_resource(args.frame_image_dir)

        store = AnnotationStore()

        root = tk.Tk()
        app = OpenCvJsonlViewer(root, video_dir, frame_image_dir, store, args.camera)
        root.protocol("WM_DELETE_WINDOW", app.close)
        root.mainloop()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        pause_before_exit(1)
    except tk.TclError as exc:
        print(f"无法创建图形界面：{exc}", file=sys.stderr)
        pause_before_exit(1)
    except Exception:
        import traceback

        traceback.print_exc()
        pause_before_exit(1)


if __name__ == "__main__":
    main()
