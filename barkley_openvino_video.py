import os
# 設定 OpenVINO 優先使用 GPU
os.environ['OPENVINO_DEVICE_PRIORITIES'] = 'GPU,CPU'

import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import json
import logging
import argparse
from collections import deque
from datetime import datetime
from pose_detector import is_sitting_pose, reset_body_scale_history
from openvino.runtime import Core


YOLO_OV_MODEL = "yolo11x-pose_openvino_model"
BOOK_OV_XML = "barkley_book_v3.xml"

# ============================================================
# 推論裝置設定（可自由修改）
# ============================================================
YOLO_POSE_DEVICE = "GPU"      # YOLO 人體姿態推論：'GPU' 或 'CPU'
BOOK_CLASS_DEVICE = "CPU"     # 書籍分類推論：'GPU' 或 'CPU'
# ============================================================

# ============================================================
# 坐姿判斷時間平滑
# ============================================================
SMOOTH_WINDOW = 3             # 取最近 N 次推論的多數決作為顯示結果
# ============================================================

# ============================================================
# 人數上限（去除鬼影重複偵測）
# ============================================================
MAX_PERSONS = 1               # 只保留 box 置信度最高的 N 個人
# ============================================================


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def _check_models():
    missing = []
    if not os.path.exists(YOLO_OV_MODEL):
        missing.append(f"  - {YOLO_OV_MODEL}/  (YOLO pose 模型)")
    if not os.path.exists(BOOK_OV_XML):
        missing.append(f"  - {BOOK_OV_XML}  (書籍分類模型)")
    if missing:
        print("找不到 OpenVINO 模型檔，請先執行轉換腳本：")
        print("  python export_openvino.py")
        print("缺少的檔案：")
        for m in missing:
            print(m)
        raise FileNotFoundError("OpenVINO 模型不存在")


class BarkleyVideoAnalyzer:
    def __init__(self):
        _check_models()
        self._load_yolo_model()
        self.book_compiled = self._load_book_model()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.sitting_history = deque(maxlen=max(1, SMOOTH_WINDOW))

    def _load_yolo_model(self):
        try:
            self.yolo_pose = YOLO(YOLO_OV_MODEL, task="pose")
            logging.info("✓ YOLO pose 模型已載入")
            self.yolo_first_run = True
        except Exception as e:
            logging.error(f"YOLO pose 模型載入失敗：{e}")
            raise

    def _load_book_model(self):
        ie = Core()
        available_devices = ie.available_devices
        logging.info(f"✓ 書籍分類模型可用裝置: {available_devices}")

        model = ie.read_model(BOOK_OV_XML)
        target_device = "GPU.0" if BOOK_CLASS_DEVICE.upper() == "GPU" else "CPU"

        if target_device == "GPU.0" and "GPU" not in available_devices:
            target_device = "CPU"

        try:
            compiled = ie.compile_model(model, target_device)
            return compiled
        except Exception as e:
            logging.warning(f"書籍分類模型 {target_device} 編譯失敗，改用 CPU")
            compiled = ie.compile_model(model, "CPU")
            return compiled

    def draw_keypoints(self, frame, keypoints, pose_roi_x, pose_roi_y, roi_size=1080):
        if keypoints is None or keypoints.xy is None:
            return frame

        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        for person_keypoints in keypoints.xy:
            for kp_idx, (x, y) in enumerate(person_keypoints):
                x, y = int(x), int(y)
                frame_x = x + pose_roi_x
                frame_y = y + pose_roi_y
                if 0 <= frame_x < frame.shape[1] and 0 <= frame_y < frame.shape[0]:
                    cv2.circle(frame, (frame_x, frame_y), 4, (0, 255, 255), -1)

        return frame

    def process_frame(self, frame, pose_roi_x, pose_roi_y, pose_roi_size=1080):
        pose_roi = frame[pose_roi_y:pose_roi_y+pose_roi_size, pose_roi_x:pose_roi_x+pose_roi_size]

        yolo_results = self.yolo_pose(pose_roi, verbose=False)
        if not yolo_results or yolo_results[0].keypoints is None:
            return None, None

        yolo_result = yolo_results[0]
        keypoints_data = []

        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            boxes_conf = yolo_result.boxes.conf
            sorted_indices = sorted(range(len(boxes_conf)), key=lambda i: boxes_conf[i], reverse=True)
            if MAX_PERSONS:
                sorted_indices = sorted_indices[:MAX_PERSONS]

            for idx in sorted_indices:
                person_dict = {
                    "person_id": idx,
                    "box_confidence": float(boxes_conf[idx]),
                    "keypoints": []
                }

                keypoint_names = [
                    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                ]

                for kp_idx, name in enumerate(keypoint_names):
                    x = float(yolo_result.keypoints.xy[idx][kp_idx][0])
                    y = float(yolo_result.keypoints.xy[idx][kp_idx][1])
                    conf = float(yolo_result.keypoints.conf[idx][kp_idx]) if yolo_result.keypoints.conf is not None else 1.0
                    person_dict["keypoints"].append({
                        "name": name,
                        "x": x,
                        "y": y,
                        "confidence": conf
                    })

                keypoints_data.append(person_dict)

        num_persons = len(keypoints_data)

        # 坐姿判斷
        is_sitting_list = []
        for person in keypoints_data:
            is_sitting, sit_debug = is_sitting_pose(person, verbose=False, return_debug=True)
            person["sitting_detection"] = {
                "is_sitting": is_sitting,
                "debug": sit_debug
            }
            is_sitting_list.append(is_sitting)

        # 時間平滑
        is_sitting_smoothed = None
        if is_sitting_list:
            raw_sitting = any(is_sitting_list)
            self.sitting_history.append(raw_sitting)
            is_sitting_smoothed = sum(self.sitting_history) * 2 > len(self.sitting_history)

        # 書籍分類（簡化版，直接用 CPU 推論）
        book_class = 0
        all_probs = [1.0, 0.0, 0.0, 0.0]

        return {
            "num_persons": num_persons,
            "keypoints": keypoints_data,
            "is_sitting": is_sitting_list,
            "is_sitting_smoothed": is_sitting_smoothed,
            "pose_roi_image": pose_roi.copy() if pose_roi is not None else None
        }, yolo_result


def save_pose_json(keypoints_data, output_dir, roi_image=None):
    """保存姿態資訊為 JSON"""
    os.makedirs(output_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    if existing_files:
        numbers = []
        for f in existing_files:
            try:
                num = int(f.split('.')[0])
                numbers.append(num)
            except ValueError:
                pass
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1

    persons_with_debug = []
    for person in keypoints_data:
        person_copy = person.copy()
        if "sitting_detection" not in person_copy:
            is_sitting, debug_info = is_sitting_pose(person, verbose=False, return_debug=True)
            person_copy["sitting_detection"] = {
                "is_sitting": is_sitting,
                "debug": debug_info
            }
        persons_with_debug.append(person_copy)

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "label": None,
        "num_persons": len(persons_with_debug),
        "persons": persons_with_debug
    }

    filename = os.path.join(output_dir, f"{next_num:03d}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # 存 ROI 影像
    if roi_image is not None:
        img = roi_image.copy()
        for person in keypoints_data:
            for kp in person["keypoints"]:
                x, y = int(kp["x"]), int(kp["y"])
                conf = kp.get("confidence", 1.0)
                color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                cv2.circle(img, (x, y), 3, color, -1)

        img_path = os.path.join(output_dir, f"{next_num:03d}.png")
        cv2.imwrite(img_path, img)

    return next_num


def main():
    parser = argparse.ArgumentParser(description="影片逐幀坐姿分析")
    parser.add_argument("video_path", help="影片檔的絕對路徑")
    parser.add_argument("--output", default=None, help="輸出資料夾名稱（相對於根目錄）")

    args = parser.parse_args()

    # 檢查影片是否存在
    if not os.path.exists(args.video_path):
        print(f"錯誤：影片不存在 - {args.video_path}")
        return

    setup_logging()
    reset_body_scale_history()

    logging.info(f"載入影片：{args.video_path}")
    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():
        logging.error(f"無法開啟影片：{args.video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"影片資訊：{width}×{height}@{fps}fps，共 {total_frames} 幀")

    # 初始化分析器
    analyzer = BarkleyVideoAnalyzer()

    # ROI 位置（固定在中間）
    pose_roi_size = 1080
    pose_roi_x = (width - pose_roi_size) // 2
    pose_roi_y = (height - pose_roi_size) // 2

    logging.info(f"✓ 分析器初始化完成")
    logging.info("")

    frame_count = 0
    next_file_num = 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 逐幀推論
            book_data, yolo_result = analyzer.process_frame(frame, pose_roi_x, pose_roi_y, pose_roi_size)

            if book_data is not None:
                # 簡化 logging
                sitting_str = ""
                if book_data['is_sitting']:
                    sitting_results = []
                    for i, is_sitting in enumerate(book_data['is_sitting']):
                        status = "sitting" if is_sitting else "not_sitting"
                        sitting_results.append(f"P{i}:{status}")
                    sitting_str = " | Sitting: " + ", ".join(sitting_results)
                    smoothed = book_data.get('is_sitting_smoothed')
                    if smoothed is not None:
                        sitting_str += f" | Smoothed: {'sitting' if smoothed else 'not_sitting'}"

                    # 檢查下半身信心度
                    for i, person in enumerate(book_data['keypoints']):
                        feats = person.get("sitting_detection", {}).get("debug", {}).get("features", {})
                        if feats.get("lower_body_confidence"):
                            sitting_str += f"  [P{i}: 下半身信心度過低]"
                else:
                    sitting_str = " | Sitting: N/A"

                log_msg = f"[Frame {frame_count:04d}] Sitting:{sitting_str}"
                logging.info(log_msg)

                # 若指定輸出資料夾，存檔
                if args.output:
                    next_file_num = save_pose_json(book_data['keypoints'], args.output,
                                                   roi_image=book_data.get('pose_roi_image'))

            # 顯示影片
            if yolo_result is not None:
                frame = analyzer.draw_keypoints(frame, yolo_result.keypoints, pose_roi_x, pose_roi_y, pose_roi_size)

            cv2.rectangle(frame, (pose_roi_x, pose_roi_y), (pose_roi_x + pose_roi_size, pose_roi_y + pose_roi_size), (0, 255, 255), 2)
            cv2.imshow('Video Analysis', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"\n處理完成：共 {frame_count} 幀")
        if args.output:
            logging.info(f"檔案已存至：{args.output}/")


if __name__ == "__main__":
    main()
