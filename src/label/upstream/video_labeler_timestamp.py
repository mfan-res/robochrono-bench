#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频时间戳标注工具 - 支持20个类别（1-9, q-p），输出为 *_segments.json
"""

import cv2
import sys
import os
import re
import json
import argparse
from pathlib import Path
import numpy as np


# 支持的类别键映射：1-9, q-p 共20个
CLASS_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
              'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']
KEY_TO_ID = {k: i for i, k in enumerate(CLASS_KEYS)}  # {'1':0, '2':1, ..., 'q':9, ...}
ACTION_VERBS = {
    "approach", "close", "drop", "grasp", "hold", "insert", "lift", "move",
    "open", "pick", "place", "pour", "press", "pull", "push", "put",
    "release", "remove", "rotate", "take",
}


def get_class_file_from_video(video_path):
    """根据视频路径自动推断类别文件路径"""
    video_path = Path(video_path)
    video_dir = video_path.parent
    video_name = video_path.stem

    class_name = re.sub(r'\d+', '', video_name)
    class_name = re.sub(r'_+', '_', class_name)
    class_name = class_name.strip('_')

    class_file = video_dir / f"{class_name}.txt"
    return str(class_file)


def get_trailing_number(path):
    """获取文件名末尾的编号，用于按视频序号排序/定位。"""
    match = re.search(r'(\d+)(?:_h264)?$', Path(path).stem)
    return int(match.group(1)) if match else None


def video_sort_key(path):
    """按末尾编号排序；没有编号的文件回退到普通文件名排序。"""
    path = Path(path)
    trailing_number = get_trailing_number(path)
    if trailing_number is None:
        return (1, path.name.lower())
    prefix = re.sub(r'\d+(?:_h264)?$', '', path.stem)
    return (0, prefix.lower(), trailing_number, path.suffix.lower())


def apply_start_video_filter(video_files, start_video):
    """从指定编号、文件名或 stem 开始处理视频列表。"""
    if start_video is None:
        return video_files

    start_text = str(start_video).strip()
    if not start_text:
        return video_files

    # 纯数字表示从末尾编号 >= 该数字的视频开始。
    if start_text.isdigit():
        start_number = int(start_text)
        filtered = [
            video_path for video_path in video_files
            if get_trailing_number(video_path) is not None and get_trailing_number(video_path) >= start_number
        ]
        if not filtered:
            print(f"警告: 没有找到末尾编号 >= {start_number} 的视频，将不处理任何视频")
        else:
            print(f"\n从末尾编号 >= {start_number} 的视频开始处理")
        return filtered

    # 非数字按文件名或 stem 精确匹配，找到后从该文件在排序列表中的位置开始。
    start_path = Path(start_text)
    target_name = start_path.name
    target_stem = start_path.stem
    for index, video_path in enumerate(video_files):
        if video_path.name == target_name or video_path.stem == target_stem:
            print(f"\n从指定视频开始处理: {video_path.name}")
            return video_files[index:]

    print(f"警告: 没有找到起始视频 {start_text}，将不处理任何视频")
    return []


def get_default_label_path(video_path, output_dir=None):
    """获取视频对应的标注输出 json 路径。"""
    video_path = Path(video_path)
    video_stem = video_path.stem
    base_name = re.sub(r'_h264$', '', video_stem)
    label_dir = Path(output_dir) if output_dir is not None else video_path.parent
    return label_dir / f"{base_name}_segments.json"


def format_timestamp(seconds):
    """把秒数格式化为 HH:MM:SS.mmm。"""
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def normalize_class_text(class_text):
    """清理类别编号前缀，如 C1:pick red cube、M1:pick red cube。"""
    class_text = re.sub(r'^\s*[A-Za-z]*\d+\s*[:：.\-]\s*', '', class_text)
    return class_text.strip()


def build_segment_description(class_text):
    """把类别文本转换为示例 JSON 中的 objects/main_verbs/narration 字段。"""
    cleaned = normalize_class_text(class_text)
    if not cleaned:
        return [], [], ""

    words = cleaned.rstrip(".").split()
    verb = words[0].lower() if words else ""
    objects = []
    main_verbs = []

    if verb in ACTION_VERBS:
        main_verbs = [verb]
        object_words = words[1:]
        if object_words and object_words[0].lower() in {"a", "an", "the"}:
            object_words = object_words[1:]
        if object_words:
            objects = [" ".join(object_words)]

    narration_body = cleaned.rstrip(".")
    if main_verbs and objects and len(words) > 1 and words[1].lower() not in {"a", "an", "the"}:
        narration_body = f"{words[0]} the {' '.join(words[1:])}"
    if narration_body:
        narration_body = narration_body[0].upper() + narration_body[1:]
    narration = f" {narration_body}." if narration_body else ""
    return objects, main_verbs, narration


class VideoLabeler:
    def __init__(self, video_path, class_file_path=None, output_path=None):
        self.video_path = video_path

        if class_file_path is None:
            class_file_path = get_class_file_from_video(video_path)
            print(f"[自动推断] 类别文件: {class_file_path}")

        self.class_file_path = class_file_path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        if not os.path.exists(class_file_path):
            raise FileNotFoundError(f"类别文件不存在: {class_file_path}")

        # 读取类别
        self.classes = self._load_classes()
        self.num_classes = len(self.classes)
        if self.num_classes == 0:
            raise ValueError("类别文件为空")
        if self.num_classes > 20:
            raise ValueError(f"类别数量不能超过20个（当前{self.num_classes}个），支持 0-9, a-j")

        print(f"加载了 {self.num_classes} 个类别:")
        for i, cls in enumerate(self.classes):
            key = CLASS_KEYS[i]
            print(f"  [{key}] {cls}")

        # 打开视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n视频信息:")
        print(f"  总帧数: {self.total_frames}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  分辨率: {self.width}x{self.height}")
        print(f"  时长: {self.total_frames/self.fps:.2f}秒")

        self.segments = []  # [(start_frame, end_frame, class_id), ...]
        self.current_segment_start = None
        self.state = 'playing'
        self.pending_end_frame = None
        self.current_frame = 0
        self.window_name = "Video Labeler"

        if output_path is None:
            self.output_path = get_default_label_path(video_path)
        else:
            self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_classes(self):
        """加载类别文件"""
        classes = []
        with open(self.class_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    classes.append(line)
        return classes

    def _on_trackbar(self, pos):
        """滑动条回调函数"""
        if self.state != 'waiting_for_class':
            self.current_frame = pos
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def _get_current_class(self):
        """获取当前帧的类别"""
        for start, end, cls_id in reversed(self.segments):
            if start <= self.current_frame <= end:
                return cls_id
        return None

    def _frame_to_time(self, frame_index):
        """将帧号转换为秒。end_frame 保存时会用后一帧时间作为区间结束。"""
        if self.fps and self.fps > 0:
            return frame_index / self.fps
        return float(frame_index)

    def _format_segment_range(self, start_frame, end_frame):
        start_time = self._frame_to_time(start_frame)
        end_time = self._frame_to_time(end_frame + 1)
        return f"frames {start_frame}-{end_frame} | {start_time:.3f}-{end_time:.3f}s"

    def _draw_info(self, frame):
        """在帧上绘制信息"""
        h, w = frame.shape[:2]

        # 创建顶部信息面板
        overlay = frame.copy()
        panel_height = 200 if self.state == 'waiting_for_class' else 160
        # cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        # cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 分两列显示类别列表（0-9 左列，a-j 右列）
        y_offset = 20
        cv2.putText(frame, "Available Classes:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 左列：0-9
        left_x = 10
        for i in range(min(10, self.num_classes)):
            y_offset = 40 + i * 16
            key = CLASS_KEYS[i]
            if self.state == 'waiting_for_class':
                color = (0, 255, 255)
                prefix = ">>> " if KEY_TO_ID.get(key) == self.num_classes - 1 else "    "
            else:
                color = (150, 150, 150)
                prefix = "    "
            text = f"{prefix}[{key}] {self.classes[i]}"
            cv2.putText(frame, text, (left_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 右列：a-j（如果有）
        if self.num_classes > 10:
            right_x = w // 2
            for i in range(10, self.num_classes):
                y_offset = 40 + (i - 10) * 16
                key = CLASS_KEYS[i]
                if self.state == 'waiting_for_class':
                    color = (0, 255, 255)
                    prefix = ">>> "
                else:
                    color = (150, 150, 150)
                    prefix = "    "
                text = f"{prefix}[{key}] {self.classes[i]}"
                cv2.putText(frame, text, (right_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 显示当前状态和信息
        info_y = panel_height - 40

        if self.state == 'waiting_for_class':
            range_text = self._format_segment_range(self.current_segment_start, self.pending_end_frame)
            cv2.putText(frame, f"SELECT CLASS for {range_text}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Press key 1-9 or q-p to select class", 
                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            progress = (self.current_frame / self.total_frames) * 100 if self.total_frames > 0 else 0
            cv2.putText(frame, f"Frame: {self.current_frame}/{self.total_frames} ({progress:.1f}%)", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            current_time = self.current_frame / self.fps if self.fps > 0 else 0
            cv2.putText(frame, f"Time: {current_time:.2f}s", 
                       (200, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            current_cls = self._get_current_class()
            if current_cls is not None:
                cls_key = CLASS_KEYS[current_cls]
                cls_text = f"Current: [{cls_key}] {self.classes[current_cls]}"
                cv2.putText(frame, cls_text, (400, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 显示已完成的段
        seg_y = panel_height + 30
        cv2.putText(frame, "Completed Segments:", (10, seg_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for i, (start, end, cls_id) in enumerate(self.segments[-5:]):  # 只显示最近5段
            seg_y += 18
            cls_key = CLASS_KEYS[cls_id]
            start_time = self._frame_to_time(start)
            end_time = self._frame_to_time(end + 1)
            text = f"  [{i}] {start_time:.2f}-{end_time:.2f}s: [{cls_key}] {self.classes[cls_id]}"
            cv2.putText(frame, text, (10, seg_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)

        if len(self.segments) > 5:
            seg_y += 18
            cv2.putText(frame, f"  ... and {len(self.segments)-5} more", (10, seg_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # 显示当前正在编辑的段起点
        if self.current_segment_start is not None:
            seg_y += 25
            start_time = self._frame_to_time(self.current_segment_start)
            next_info = f"Segment start: frame {self.current_segment_start} ({start_time:.3f}s)"
            cv2.putText(frame, next_info, (10, seg_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        # 底部操作提示
        hint_y = h - 60
        hints_bg = frame.copy()
        # cv2.rectangle(hints_bg, (0, h-80), (w, h), (0, 0, 0), -1)
        # cv2.addWeighted(hints_bg, 0.8, frame, 0.2, 0, frame)

        if self.state == 'waiting_for_class':
            hint_text = "Press 1-9/q-p to select class | Backspace to cancel"
        else:
            hint_text = "'k':Start/End segment | 'f':End at video end | 's':Save | 'z':Undo | 'q':Quit | Space:Play/Pause | Arrows:Seek"

        cv2.putText(frame, hint_text, (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 进度条标记
        if self.total_frames > 0:
            bar_width = w - 20
            for start, end, cls_id in self.segments:
                x_pos = int(10 + (end / self.total_frames) * bar_width)
                color = (0, 255, 0) if cls_id % 2 == 0 else (255, 0, 0)
                cv2.line(frame, (x_pos, h-50), (x_pos, h-20), color, 2)
                cv2.putText(frame, str(end), (x_pos-15, h-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return frame
    def _add_segment(self, end_frame, class_id):
        """添加一个段"""
        if class_id < 0 or class_id >= self.num_classes:
            cls_key = CLASS_KEYS[class_id] if 0 <= class_id < 20 else str(class_id)
            print(f"错误: 无效的类别 [{cls_key}]")
            return False

        if self.current_segment_start is None:
            print("错误: 尚未标记起始帧，请先按 'k' 标记起始帧")
            return False

        if end_frame < self.current_segment_start:
            print(f"错误: 结束帧 {end_frame} 小于起始帧 {self.current_segment_start}")
            return False

        for start, end, _ in self.segments:
            if end_frame < start or self.current_segment_start > end:
                continue

            overlap_start = max(self.current_segment_start, start)
            overlap_end = min(end_frame, end)
            shared_boundary = (
                overlap_start == overlap_end
                and (self.current_segment_start == end or end_frame == start)
            )
            if not shared_boundary:
                print(f"错误: 当前段 {self.current_segment_start}-{end_frame} 与已有段 {start}-{end} 重叠")
                return False

        self.segments.append((self.current_segment_start, end_frame, class_id))
        self.segments.sort(key=lambda item: item[0])
        cls_key = CLASS_KEYS[class_id]
        print(f"添加时间段 [{len(self.segments)-1}]: {self._format_segment_range(self.current_segment_start, end_frame)} -> [{cls_key}] {self.classes[class_id]}")

        self.current_segment_start = None
        return True

    def _generate_labels(self):
        """生成 egodata 风格的时间戳区间 JSON。"""
        segments = []
        video_stem = Path(self.video_path).stem
        segment_id_prefix = re.sub(r'_h264$', '', video_stem)

        for index, (start, end, cls_id) in enumerate(
            sorted(self.segments, key=lambda item: item[0]),
            start=1,
        ):
            start_time = self._frame_to_time(start)
            end_time = self._frame_to_time(end + 1)
            objects, main_verbs, narration = build_segment_description(self.classes[cls_id])
            segments.append({
                "id": f"{segment_id_prefix}-{index}",
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "start_time": format_timestamp(start_time),
                "end_time": format_timestamp(end_time),
                "start_frame": start,
                "end_frame": end,
                "objects": objects,
                "main_verbs": main_verbs,
                "narration": narration,
            })
        return {"segments": segments}

    def save_labels(self):
        """保存标签文件"""
        labels = self._generate_labels()

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"\n保存完成!")
        print(f"输出文件: {self.output_path}")
        print("输出格式: *_segments.json，顶层包含 segments 列表")
        print(f"已标注段数: {len(self.segments)}")

        print("\n分段统计:")
        for i, (start, end, cls_id) in enumerate(self.segments):
            cls_key = CLASS_KEYS[cls_id]
            frame_count = end - start + 1
            start_time = self._frame_to_time(start)
            end_time = self._frame_to_time(end + 1)
            print(f"  [{i}] [{cls_key}] {self.classes[cls_id]}: {start_time:.3f}-{end_time:.3f}s, 帧 {start}-{end} ({frame_count}帧)")

    def _is_valid_class_key(self, key_char):
        """检查按键是否是有效的类别键"""
        return key_char in KEY_TO_ID and KEY_TO_ID[key_char] < self.num_classes

    def run(self):
        """运行标注工具"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1400, 900)

        cv2.createTrackbar("Progress", self.window_name, 0, max(0, self.total_frames-1), self._on_trackbar)

        paused = True  # 默认暂停，不自动播放

        print("\n" + "="*60)
        print("标注工具已启动 (支持 20 个类别: 1-9, q-p)")
        print("="*60)
        print("操作说明:")
        print("  1. 拖动滑动条定位到时间段起点，按 'k' 标记起始帧")
        print("  2. 拖动滑动条定位到时间段终点，再按 'k' 标记结束帧")
        print("  3. 按数字键 1-9 或字母 q-p 选择该时间段的类别")
        print("  4. 重复步骤1-3，可跳过任意不需要标注的帧")
        print("  5. 按 'f' 将当前段结束设置为视频结尾")
        print("  6. 按 'z' 撤销上一个标注段")
        print("  7. 按 's' 保存标签文件")
        print("  输出格式: *_segments.json，允许未标注帧不出现在任何 segment 中")
        print("="*60 + "\n")

        while True:
            # 总是读取当前帧进行显示
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if not ret:
                # 定位到最后一帧再试一次（避免自动退出到下一个视频）
                self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret:
                    # 无法读取帧时，不再直接 break，而是显示一个空白/提示帧并暂停在末帧
                    h = self.height if self.height > 0 else 480
                    w = self.width if self.width > 0 else 640
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(frame, "End of video - paused at final frame. Press 's' to save and continue.",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    paused = True

            cv2.setTrackbarPos("Progress", self.window_name, self.current_frame)
            display_frame = self._draw_info(frame.copy())
            cv2.imshow(self.window_name, display_frame)

            wait_time = 0 if self.state == 'waiting_for_class' else 30
            key = cv2.waitKey(wait_time) & 0xFF

            if key == 255:
                # 只在未暂停且不在等待类别状态时自动前进
                if not paused and self.state != 'waiting_for_class':
                    self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                    
                    # 检查是否到达视频末尾
                    if self.current_frame >= self.total_frames - 1:
                        paused = True
                        print("\n" + "⚠️ " * 20)
                        print("✓ 视频已播放到末尾！")
                        print("="*60)
                        print("请选择:")
                        print("  's' - 保存标签文件并进入下一个视频")
                        print("  'f' - 将当前段结束设置为视频结尾")
                        print("  'z' - 撤销上一段重新标注")
                        print("  'q' - 退出（不保存）")
                        print("="*60 + "\n")
                continue

            # 将按键转换为字符
            key_char = None
            if ord('1') <= key <= ord('9'):
                key_char = chr(key)
            elif chr(key).lower() in CLASS_KEYS:
                key_char = chr(key).lower()

            # 处理类别选择键 (1-9, q-p)
            if key_char and key_char in KEY_TO_ID:
                if self.state == 'waiting_for_class':
                    class_id = KEY_TO_ID[key_char]
                    if class_id < self.num_classes:
                        if self._add_segment(self.pending_end_frame, class_id):
                            self.state = 'playing'
                            self.pending_end_frame = None
                            print(f"类别 [{key_char}] 已选择，可移动到下一个起点继续标注\n")
                    else:
                        print(f"错误: 类别 [{key_char}] 不存在，有效范围: {''.join(CLASS_KEYS[:self.num_classes])}")
                else:
                    print(f"提示: 先按 'k' 标记起始帧和结束帧，再选择类别")
                continue

            # 其他按键处理
            if key == ord('q') or key == 27:
                if len(self.segments) > 0:
                    save = input("\n是否保存标签? (y/n): ").strip().lower()
                    if save == 'y':
                        self.save_labels()
                break

            elif key == ord('k'):
                if self.state == 'waiting_for_class':
                    print("提示: 请先为当前段选择类别（按1-9/q-p），或按Backspace取消")
                else:
                    if self.current_frame >= self.total_frames:
                        print("错误: 已经到达视频末尾")
                        continue

                    if self.current_segment_start is None:
                        self.current_segment_start = self.current_frame
                        start_time = self._frame_to_time(self.current_segment_start)
                        print(f"\n标记起始帧: {self.current_segment_start} ({start_time:.3f}s)")
                        print("请移动到该时间段的结束帧，再按 'k'")
                        continue

                    if self.current_frame < self.current_segment_start:
                        print(f"错误: 结束帧 {self.current_frame} 小于起始帧 {self.current_segment_start}")
                        continue

                    self.pending_end_frame = self.current_frame
                    self.state = 'waiting_for_class'
                    print(f"\n标记结束帧: {self.current_frame}")
                    print(f"时间段: {self._format_segment_range(self.current_segment_start, self.current_frame)}")
                    print("请按 1-9 或 q-p 选择类别:")
                    for i, cls in enumerate(self.classes):
                        key_str = CLASS_KEYS[i]
                        marker = " <<<" if i == self.num_classes - 1 else ""
                        print(f"  [{key_str}] {cls}{marker}")

            elif key == ord('f'):
                if self.state == 'waiting_for_class':
                    print("提示: 请先为当前段选择类别")
                else:
                    end_frame = self.total_frames - 1
                    if self.current_segment_start is None:
                        self.current_segment_start = min(self.current_frame, end_frame)

                    if self.current_segment_start > end_frame:
                        print("错误: 起始帧已超过视频末尾，所有段已标注完成")
                        continue

                    self.pending_end_frame = end_frame
                    self.state = 'waiting_for_class'
                    print(f"\n标记到视频结尾: {self._format_segment_range(self.current_segment_start, end_frame)}")
                    print("请按 1-9 或 q-p 选择该时间段的类别:")
                    for i, cls in enumerate(self.classes):
                        key_str = CLASS_KEYS[i]
                        print(f"  [{key_str}] {cls}")

            elif key == 8 or key == 127:
                if self.state == 'waiting_for_class':
                    self.state = 'playing'
                    self.pending_end_frame = None
                    print("取消当前结束帧，请重新选择结束帧")
                elif self.current_segment_start is not None:
                    print(f"取消起始帧: {self.current_segment_start}")
                    self.current_segment_start = None
                elif len(self.segments) > 0:
                    removed = self.segments.pop()
                    self.current_segment_start = removed[0]
                    removed_key = CLASS_KEYS[removed[2]]
                    print(f"删除最后一段: 帧 {removed[0]}-{removed[1]} -> [{removed_key}] {self.classes[removed[2]]}")

            elif key == ord('z'):
                if self.state == 'waiting_for_class':
                    self.state = 'playing'
                    self.pending_end_frame = None
                    print("取消当前结束帧，请重新选择结束帧")
                elif self.current_segment_start is not None:
                    print(f"取消起始帧: {self.current_segment_start}")
                    self.current_segment_start = None
                elif len(self.segments) > 0:
                    removed = self.segments.pop()
                    self.current_segment_start = removed[0]
                    removed_key = CLASS_KEYS[removed[2]]
                    print(f"撤销上一段: 帧 {removed[0]}-{removed[1]} -> [{removed_key}] {self.classes[removed[2]]}")
                else:
                    print("提示: 没有可以撤销的段")

            elif key == ord('s'):
                if self.state == 'waiting_for_class':
                    print("警告: 当前时间段已选择结束帧但还没有类别")
                    confirm = input("是否只保存已完成的时间段? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                    self.state = 'playing'
                    self.pending_end_frame = None
                    self.current_segment_start = None
                elif self.current_segment_start is not None:
                    print(f"警告: 当前起始帧 {self.current_segment_start} 还没有结束帧和类别")
                    confirm = input("是否只保存已完成的时间段? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                    self.current_segment_start = None

                if len(self.segments) == 0:
                    print("警告: 还没有标注任何段")
                    confirm = input("是否继续保存? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue

                self.save_labels()
                print(f"\n标签已保存到: {self.output_path}")
                print("即将进入下一个视频...\n")
                break   # ⭐ 保存后才能进入下一个视频

            elif key == ord(' '):
                if self.state != 'waiting_for_class':
                    paused = not paused
                    status = "暂停" if paused else "播放"
                    print(f"{status} (按空格键切换)")

            # 方向键控制
            elif key == 81 or key == 2:  # 左方向键
                if self.state != 'waiting_for_class':
                    self.current_frame = max(0, self.current_frame - 1)

            elif key == 83 or key == 3:  # 右方向键
                if self.state != 'waiting_for_class':
                    self.current_frame = min(self.total_frames - 1, self.current_frame + 1)

            elif key == 82 or key == 0:  # 上方向键
                if self.state != 'waiting_for_class':
                    self.current_frame = max(0, self.current_frame - 10)

            elif key == 84 or key == 1:  # 下方向键
                if self.state != 'waiting_for_class':
                    self.current_frame = min(self.total_frames - 1, self.current_frame + 10)

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n标注工具已关闭")


def process_video_directory(directory_path, class_file_path=None, start_video=None, output_path=None):
    """遍历目录下所有视频进行标注"""
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"错误: 目录不存在: {directory_path}")
        return
    
    if not directory_path.is_dir():
        print(f"错误: 不是目录: {directory_path}")
        return
    
    output_dir = None
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix:
            print("错误: 目录模式下 --output 不能指定单个标签文件，请指定输出目录")
            return
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有视频文件
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(directory_path.glob(f"*{ext}"))
    
    if not video_files:
        print(f"错误: 目录下没有找到视频文件: {directory_path}")
        return
    
    video_files = sorted(video_files, key=video_sort_key)
    video_files = apply_start_video_filter(video_files, start_video)
    if not video_files:
        print(f"错误: 没有需要从指定起点处理的视频: {directory_path}")
        return

    print(f"\n找到 {len(video_files)} 个视频文件")
    print("="*60)
    
    # 过滤出需要标注的视频（只在对应 *_segments.json 已存在时跳过）
    videos_to_process = []
    skipped_videos = []
    
    for video_path in video_files:
        label_file = get_default_label_path(video_path, output_dir)
        
        if label_file.exists():
            skipped_videos.append((video_path.name, label_file.name))
        else:
            videos_to_process.append(video_path)
    
    if skipped_videos:
        print(f"\n已跳过 {len(skipped_videos)} 个视频（已有 *_segments.json 标签文件）:")
        for video_name, label_name in skipped_videos:
            print(f"  [跳过] {video_name} -> {label_name}")
    
    if not videos_to_process:
        print("\n没有需要标注的视频，全部已完成！")
        return
    
    print(f"\n需要标注 {len(videos_to_process)} 个视频:")
    for i, video_path in enumerate(videos_to_process, 1):
        print(f"  [{i}] {video_path.name}")
    
    print("\n"+"="*60)
    
    # 逐个处理视频
    for idx, video_path in enumerate(videos_to_process, 1):
        print(f"\n\n{'='*60}")
        print(f"处理第 {idx}/{len(videos_to_process)} 个视频")
        print(f"{'='*60}")
        print(f"视频: {video_path.name}\n")
        
        try:
            labeler = VideoLabeler(
                str(video_path),
                class_file_path,
                output_path=get_default_label_path(video_path, output_dir)
            )
            labeler.run()
            print(f"\n✓ 完成: {video_path.name}")
        except FileNotFoundError as e:
            print(f"\n✗ 跳过: {video_path.name}")
            print(f"  原因: {e}")
            continue
        except Exception as e:
            print(f"\n✗ 错误: {video_path.name}")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n\n{'='*60}")
    print("全部视频处理完成！")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="视频标注工具：支持单视频或目录批量标注"
    )
    parser.add_argument("path", nargs="?", help="视频文件或目录路径")
    parser.add_argument("class_file", nargs="?", help="类别文件路径")
    parser.add_argument(
        "start_video_positional",
        nargs="?",
        help="起始视频编号/文件名（兼容旧式位置参数，推荐使用 --start）"
    )
    parser.add_argument(
        "--start",
        "-s",
        dest="start_video",
        help="目录模式下从指定末尾编号、文件名或文件 stem 开始，例如: --start 23 或 --start action_0023.mp4"
    )
    parser.add_argument(
        "--txt",
        "--class-file",
        dest="class_file_option",
        default="/home/kewei/NAS/lerobot_datasets-26-06-17-17-25-20/videos/observation.images.left_eye/chunk-000/cube.txt",
        help="类别 txt 文件路径；指定后优先于位置参数中的类别文件路径"
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        help="标注结果 *_segments.json 输出路径；单视频可指定 json 文件，目录模式请指定输出目录"
    )
    args = parser.parse_args()

    start_video = args.start_video if args.start_video is not None else args.start_video_positional

    if args.path:
        path = args.path
        class_file_path = args.class_file_option if args.class_file_option is not None else args.class_file
        
        # 判断是文件还是目录
        path_obj = Path(path)
        if path_obj.is_file():
            # 处理单个视频
            try:
                labeler = VideoLabeler(path, class_file_path, args.output_path)
                labeler.run()
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        elif path_obj.is_dir():
            # 处理目录下的所有视频
            process_video_directory(path, class_file_path, start_video, args.output_path)
        else:
            print(f"错误: 路径不存在: {path}")
            sys.exit(1)
    else:
        # 默认处理指定目录
        directory_path = "/home/kewei/YWC/cola-pour/test_data/standard"
        
        print("使用默认路径:")
        print(f"  目录: {directory_path}")
        print("\n或者通过命令行指定:")
        print(f"  python {sys.argv[0]} <视频文件或目录路径> [类别文件路径]")
        print(f"  python {sys.argv[0]} <视频文件或目录路径> --txt <类别文件路径>")
        print(f"  python {sys.argv[0]} <目录路径> [类别文件路径] --start <起始编号或视频名>")
        print(f"  python {sys.argv[0]} <视频文件> [类别文件路径] --output <标注结果_segments.json路径>")
        print("\n开始处理目录下的视频...\n")
        
        process_video_directory(directory_path)


if __name__ == "__main__":
    main()
