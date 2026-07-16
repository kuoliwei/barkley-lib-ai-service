import os
# 設定 OpenVINO 優先使用 GPU
os.environ['OPENVINO_DEVICE_PRIORITIES'] = 'GPU,CPU'

import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
import json
import logging
from collections import deque
from datetime import datetime
from pose_detector import is_sitting_pose
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
# 幀採樣設定（可自由修改）
# ============================================================
INFERENCE_INTERVAL = 1.0      # 推論間隔（秒），0 表示每幀都推論
                              # 例如 1.0 = 每 1 秒推論一次
                              #      2.0 = 每 2 秒推論一次
                              #      0.5 = 每 0.5 秒推論一次
# ============================================================

# ============================================================
# 自動存檔設定（收集連續樣本用）
# ============================================================
AUTO_SAVE_LABEL = "sitting"   # 每次推論自動存檔並套用此標註
                              # "sitting"  = 這段錄製過程全程是坐著
                              # "standing" = 這段錄製過程全程是站著
                              # None       = 關閉自動存檔（只用 C/V 鍵手動存）
# ============================================================

# ============================================================
# 人數上限（去除鬼影重複偵測）
# ============================================================
MAX_PERSONS = 1               # YOLO 偶爾會把同一人重複偵測成兩個框，
                              # 其中一個關鍵點劣化會污染判斷。
                              # 只保留 box 置信度最高的 N 個人；None 表示不限制
# ============================================================

# ============================================================
# 坐姿判斷時間平滑
# ============================================================
SMOOTH_WINDOW = 3             # 取最近 N 次推論的多數決作為顯示結果，
                              # 消除單幀擦邊誤判（如 084 差 0.01 分的情況）。
                              # 1 表示不平滑
# ============================================================

# 設定 logging
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"barkley_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_file


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


class BarkleyTracker:
    def __init__(self):
        _check_models()
        self._load_yolo_model()
        self.book_compiled = self._load_book_model()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        # 坐姿判斷的時間平滑：保留最近 N 次推論的原始結果做多數決
        self.sitting_history = deque(maxlen=max(1, SMOOTH_WINDOW))

    def _load_yolo_model(self):
        try:
            self.yolo_pose = YOLO(YOLO_OV_MODEL, task="pose")
            logging.info("YOLO pose 模型：成功載入 OpenVINO 格式（優先使用 GPU，由環境變數控制）")
            self.yolo_first_run = True  # 用來在首次推論時檢查裝置
        except Exception as e:
            logging.error(f"YOLO pose 模型載入失敗：{e}")
            raise

    def _check_yolo_device(self):
        """在首次推論時檢查 YOLO 是否真的用了設定的裝置"""
        if not self.yolo_first_run:
            return
        self.yolo_first_run = False

        try:
            ie = Core()
            available = ie.available_devices
            logging.info(f"YOLO - OpenVINO 可用裝置: {available}")
            logging.info(f"YOLO - 設定裝置: {YOLO_POSE_DEVICE}")

            # 根據設定驗證裝置
            target_device = "GPU.0" if YOLO_POSE_DEVICE.upper() == "GPU" else "CPU"

            if target_device == "GPU.0" and "GPU" not in available:
                logging.warning("YOLO - GPU 不可用，改用 CPU")
                target_device = "CPU"

            try:
                xml_path = YOLO_OV_MODEL + "/yolo11x-pose.xml"
                test_model = ie.read_model(xml_path)
                ie.compile_model(test_model, target_device)
                if target_device == "GPU.0":
                    logging.info("YOLO - ✓ GPU.0 編譯成功，推論使用 Intel GPU (Iris Xe)")
                else:
                    logging.info("YOLO - ✓ CPU 編譯成功，推論使用 CPU")
            except Exception as err:
                logging.warning(f"YOLO - {target_device} 編譯失敗: {err}，改用 CPU")
                # 嘗試編譯至 CPU
                test_model = ie.read_model(xml_path)
                ie.compile_model(test_model, "CPU")
                logging.info("YOLO - ✓ CPU 編譯成功，推論使用 CPU")
        except Exception as e:
            logging.warning(f"YOLO - 無法檢查裝置信息: {e}")

    def _load_book_model(self):
        ie = Core()
        available_devices = ie.available_devices
        logging.info(f"書籍分類 - OpenVINO 可用裝置: {available_devices}")

        model = ie.read_model(BOOK_OV_XML)
        logging.info(f"書籍分類 - 設定裝置: {BOOK_CLASS_DEVICE}")

        # 根據設定使用 GPU 或 CPU
        target_device = "GPU.0" if BOOK_CLASS_DEVICE.upper() == "GPU" else "CPU"

        if target_device == "GPU.0" and "GPU" not in available_devices:
            logging.warning(f"書籍分類模型：GPU 不可用，改用 CPU")
            target_device = "CPU"

        try:
            compiled = ie.compile_model(model, target_device)
            if target_device == "GPU.0":
                logging.info("書籍分類模型：✓ GPU.0 編譯成功，使用 Intel GPU (Iris Xe)")
            else:
                logging.info("書籍分類模型：✓ CPU 編譯成功，使用 CPU")
            return compiled
        except Exception as e:
            logging.warning(f"書籍分類模型：{target_device} 編譯失敗 ({e})，改用 CPU")
            compiled = ie.compile_model(model, "CPU")
            logging.info("書籍分類模型：✓ CPU 編譯成功，使用 CPU")
            return compiled

    def draw_keypoints(self, frame, keypoints, pose_roi_x, pose_roi_y, roi_size=224):
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
                cv2.circle(frame, (frame_x, frame_y), 4, (0, 255, 255), -1)
                cv2.putText(frame, keypoint_names[kp_idx], (frame_x + 5, frame_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        return frame

    def extract_pose_roi(self, frame, roi_x, roi_y, roi_size=224):
        h, w = frame.shape[:2]
        x1 = max(0, roi_x)
        y1 = max(0, roi_y)
        x2 = min(w, roi_x + roi_size)
        y2 = min(h, roi_y + roi_size)
        return frame[y1:y2, x1:x2]

    def extract_book_roi(self, frame, roi_x=100, roi_y=100, roi_size=224):
        h, w = frame.shape[:2]
        x1 = max(0, roi_x)
        y1 = max(0, roi_y)
        x2 = min(w, roi_x + roi_size)
        y2 = min(h, roi_y + roi_size)

        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
            padded_roi = cv2.copyMakeBorder(
                roi, 0, roi_size - roi.shape[0], 0, roi_size - roi.shape[1],
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            roi = padded_roi

        return roi

    def classify_book(self, roi):
        from PIL import Image
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess(roi_pil).unsqueeze(0).numpy()

        infer_request = self.book_compiled.create_infer_request()
        infer_request.infer({0: input_tensor})
        output = infer_request.get_output_tensor(0).data

        # softmax
        exp_out = np.exp(output - output.max())
        probabilities = exp_out / exp_out.sum()
        predicted_class = int(probabilities.argmax())
        confidence = float(probabilities[0, predicted_class])
        all_probs = probabilities[0]

        return predicted_class, confidence, all_probs

    def process_frame(self, frame, pose_roi_x=100, pose_roi_y=100, book_roi_x=100, book_roi_y=100, pose_roi_size=1080, book_roi_size=224):
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        pose_roi = self.extract_pose_roi(frame, pose_roi_x, pose_roi_y, pose_roi_size)
        # 保留未畫骨架的原始 ROI，供 C/V 鍵存檔驗證用（pose_roi 是 frame 的 view，畫圖後會被污染）
        pose_roi_snapshot = pose_roi.copy()

        # 首次推論時檢查是否真的用了 GPU
        self._check_yolo_device()

        # 不傳 device 參數，由環境變數控制
        results = self.yolo_pose(pose_roi, verbose=False)

        num_persons = 0
        keypoints_data = []
        for result in results:
            num_persons = len(result.boxes) if result.boxes is not None else 0
            if result.keypoints is not None:
                frame = self.draw_keypoints(frame, result.keypoints, pose_roi_x, pose_roi_y, pose_roi_size)
                # 關鍵點置信度：俯視角站立時下肢會被身體遮擋，置信度是重要的判斷/除錯依據
                kp_confs = result.keypoints.conf
                box_confs = result.boxes.conf if result.boxes is not None else None
                for person_idx, person_keypoints in enumerate(result.keypoints.xy):
                    person_confs = kp_confs[person_idx] if kp_confs is not None else None
                    person_data = {
                        "person_id": person_idx,
                        "box_confidence": float(box_confs[person_idx]) if box_confs is not None and person_idx < len(box_confs) else None,
                        "keypoints": []
                    }
                    for kp_idx, (x, y) in enumerate(person_keypoints):
                        kp_entry = {
                            "name": keypoint_names[kp_idx],
                            "x": float(x),
                            "y": float(y)
                        }
                        if person_confs is not None:
                            kp_entry["confidence"] = float(person_confs[kp_idx])
                        person_data["keypoints"].append(kp_entry)
                    keypoints_data.append(person_data)

        # 去除鬼影重複偵測：只保留 box 置信度最高的 MAX_PERSONS 個人
        if MAX_PERSONS and len(keypoints_data) > MAX_PERSONS:
            keypoints_data.sort(key=lambda p: p.get("box_confidence") or 0, reverse=True)
            keypoints_data = keypoints_data[:MAX_PERSONS]
            num_persons = len(keypoints_data)

        book_roi = self.extract_book_roi(frame, book_roi_x, book_roi_y, book_roi_size)
        book_class, confidence, all_probs = self.classify_book(book_roi)

        book_names = ['no_book', 'book_1', 'book_2', 'book_3']

        # 先做坐姿判斷，結果要顯示在 pose ROI 上方的文字
        # 結果連同 debug 一併存進 person dict，save_pose_json 直接重用，
        # 避免重複呼叫 is_sitting_pose 造成肩寬歷史被同一幀寫入兩次
        is_sitting_list = []
        for person in keypoints_data:
            is_sitting, sit_debug = is_sitting_pose(person, verbose=False, return_debug=True)
            person["sitting_detection"] = {
                "is_sitting": is_sitting,
                "debug": sit_debug
            }
            is_sitting_list.append(is_sitting)

        # 時間平滑：取最近 SMOOTH_WINDOW 次推論的多數決，消除單幀擦邊誤判
        is_sitting_smoothed = None
        if is_sitting_list:
            raw_sitting = any(is_sitting_list)
            self.sitting_history.append(raw_sitting)
            is_sitting_smoothed = sum(self.sitting_history) * 2 > len(self.sitting_history)
            sitting_text = "sitting" if is_sitting_smoothed else "not sitting"
            if is_sitting_smoothed != raw_sitting:
                sitting_text += f" (raw: {'sitting' if raw_sitting else 'not sitting'})"
        else:
            sitting_text = "no person"

        cv2.rectangle(frame, (pose_roi_x, pose_roi_y), (pose_roi_x + pose_roi_size, pose_roi_y + pose_roi_size), (0, 255, 255), 2)
        cv2.putText(frame, f"Pose ROI ({num_persons} persons) | {sitting_text}", (pose_roi_x + 10, pose_roi_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.rectangle(frame, (book_roi_x, book_roi_y), (book_roi_x + book_roi_size, book_roi_y + book_roi_size), (0, 255, 0), 2)
        text = f"{book_names[book_class]}"
        cv2.putText(frame, text, (book_roi_x, book_roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, {
            "book_class": book_class,
            "book_name": book_names[book_class],
            "confidence": confidence,
            "num_persons": num_persons,
            "all_probs": all_probs,
            "keypoints": keypoints_data,
            "is_sitting": is_sitting_list,
            "is_sitting_smoothed": is_sitting_smoothed,
            "pose_roi_image": pose_roi_snapshot
        }


def list_all_cameras():
    print("Available cameras:")
    found = False
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            found = True
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"  Index {index}: {width}x{height} @ {fps}fps")
            cap.release()

    if not found:
        print("  No cameras found")


def open_camera(index, width=1920, height=1080):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Requested: {width}×{height}, Actual: {actual_width}×{actual_height}")

        return cap
    return None


def save_pose_json(keypoints_data, output_dir=None, label=None, roi_image=None):
    """保存姿態資訊為 JSON（含標註、置信度、特徵值），並存 ROI 影像供人工檢視

    output_dir: 輸出資料夾路徑，若為 None 則不存檔
    label: 人工標註的真實姿態 "sitting" / "standing"，用於離線驗證與閾值校正
    roi_image: YOLO pose ROI 的原始影像（未畫骨架），會畫上關鍵點後另存 PNG
    """
    if output_dir is None:
        return

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
        # 重用 process_frame 已算好的結果；沒有才重算（例如手動呼叫時）
        if "sitting_detection" not in person_copy:
            is_sitting, debug_info = is_sitting_pose(person, verbose=False, return_debug=True)
            person_copy["sitting_detection"] = {
                "is_sitting": is_sitting,
                "debug": debug_info
            }
        persons_with_debug.append(person_copy)

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "num_persons": len(persons_with_debug),
        "persons": persons_with_debug
    }

    filename = os.path.join(output_dir, f"{next_num:03d}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved pose data to: {filename}")

    # 存 ROI 影像並畫上關鍵點與置信度，方便日後人工比對「為什麼判斷失敗」
    if roi_image is not None:
        img = roi_image.copy()
        for person in keypoints_data:
            for kp in person["keypoints"]:
                if kp["x"] == 0 and kp["y"] == 0:
                    continue
                pt = (int(kp["x"]), int(kp["y"]))
                cv2.circle(img, pt, 4, (0, 255, 255), -1)
                conf = kp.get("confidence")
                kp_label = kp["name"] if conf is None else f"{kp['name']} {conf:.2f}"
                cv2.putText(img, kp_label, (pt[0] + 5, pt[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        if label:
            cv2.putText(img, f"label: {label}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        img_path = os.path.join(output_dir, f"{next_num:03d}.png")
        cv2.imwrite(img_path, img)
        print(f"Saved ROI image to: {img_path}")

    # 立即回報標註與演算法判斷是否一致
    for i, person in enumerate(persons_with_debug):
        pred = "sitting" if person["sitting_detection"]["is_sitting"] else "not sitting"
        feats = person["sitting_detection"]["debug"].get("features", {})
        feat_str = f"thigh_ratio={feats.get('thigh_ratio')}, ext_ratio={feats.get('extension_ratio')}, score={feats.get('score')}"
        if label:
            mark = "OK" if pred == label else "MISMATCH!"
            print(f"  Person {i}: label={label}, predicted={pred} [{mark}] ({feat_str})")
        else:
            print(f"  Person {i}: predicted={pred} ({feat_str})")


def run_tracker_in_yolo(cap, output_dir=None, book_roi_x=100, book_roi_y=428, pose_roi_x=None, pose_roi_y=None, book_roi_size=224, pose_roi_size=1080):
    import time
    tracker = BarkleyTracker()

    if not cap.isOpened():
        print("Error: 無法打開攝像頭")
        return

    active_roi = "book"

    # 明確建立視窗並強制置頂，避免視窗開在其他視窗後面而看不到
    cv2.namedWindow('Book Recognition', cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setWindowProperty('Book Recognition', cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass  # 舊版 OpenCV 不支援 TOPMOST，忽略

    try:
        first_frame = True
        last_inference_time = time.time()
        book_data = None  # 保存上一次的推論結果
        frame_with_results = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if first_frame:
                h, w = frame.shape[:2]
                print(f"Actual camera resolution: {w}×{h}")
                if INFERENCE_INTERVAL > 0:
                    logging.info(f"幀採樣間隔: {INFERENCE_INTERVAL} 秒")
                first_frame = False

                # 計算 YOLO pose ROI 的中心點初始位置（畫面正中間）
                if pose_roi_x is None:
                    pose_roi_x = (w - pose_roi_size) // 2
                if pose_roi_y is None:
                    pose_roi_y = (h - pose_roi_size) // 2
                logging.info(f"YOLO pose ROI 初始位置: ({pose_roi_x}, {pose_roi_y})")

            h, w = frame.shape[:2]
            if w != 1920 or h != 1080:
                frame = cv2.resize(frame, (1920, 1080))

            # 根據時間間隔決定是否推論
            current_time = time.time()
            should_infer = False

            if INFERENCE_INTERVAL <= 0:
                # 每幀推論
                should_infer = True
            elif current_time - last_inference_time >= INFERENCE_INTERVAL:
                # 時間間隔推論
                should_infer = True
                last_inference_time = current_time

            if should_infer:
                frame_with_results, book_data = tracker.process_frame(
                    frame, pose_roi_x, pose_roi_y, book_roi_x, book_roi_y, pose_roi_size, book_roi_size
                )
            else:
                # 不推論，使用上一次的結果
                if frame_with_results is None:
                    # 第一幀時必須推論
                    frame_with_results, book_data = tracker.process_frame(
                        frame, pose_roi_x, pose_roi_y, book_roi_x, book_roi_y, pose_roi_size, book_roi_size
                    )
                    last_inference_time = current_time
                # 不推論時繼續顯示上一次的結果（frame_with_results 已有繪製的骨架和框框）

            status_text = f"Active ROI: {active_roi.upper()} (Press TAB to switch)"
            cv2.putText(frame_with_results, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 在左上角顯示採樣狀態
            if INFERENCE_INTERVAL > 0:
                sample_text = f"Inference interval: {INFERENCE_INTERVAL}s {'[推論中]' if should_infer else '[使用快取]'}"
                cv2.putText(frame_with_results, sample_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if should_infer else (0, 165, 255), 1)

            # 每次推論時列印結果
            if should_infer and book_data is not None:
                # ============ 舊版 logging（已註解）============
                # book_names_print = ['no_book', 'book_1', 'book_2', 'book_3']
                # probs_str = ", ".join([f"{name}: {prob:.4f}" for name, prob in zip(book_names_print, book_data['all_probs'])])
                #
                # sitting_str = ""
                # if book_data['is_sitting']:
                #     sitting_results = []
                #     for i, is_sitting in enumerate(book_data['is_sitting']):
                #         status = "sitting" if is_sitting else "not_sitting"
                #         sitting_results.append(f"P{i}:{status}")
                #     sitting_str = " | Sitting: " + ", ".join(sitting_results)
                #     smoothed = book_data.get('is_sitting_smoothed')
                #     if smoothed is not None:
                #         sitting_str += f" | Smoothed: {'sitting' if smoothed else 'not_sitting'}"
                # else:
                #     sitting_str = " | Sitting: N/A"
                #
                # log_msg = f"Pose: {book_data['num_persons']} persons | Book: {probs_str}{sitting_str}"
                # logging.info(log_msg)
                # ================================================

                # ============ 新版簡化 logging ============
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

                    # 檢查是否有下半身信心度過低的情況
                    for i, person in enumerate(book_data['keypoints']):
                        feats = person.get("sitting_detection", {}).get("debug", {}).get("features", {})
                        if feats.get("lower_body_confidence"):
                            sitting_str += f"  [P{i}: 下半身信心度過低]"
                else:
                    sitting_str = " | Sitting: N/A"

                log_msg = f"Sitting:{sitting_str}"
                logging.info(log_msg)
                # =========================================

                # 每次推論自動存檔（累積連續樣本供離線分析）
                # if AUTO_SAVE_LABEL and book_data['keypoints'] and output_dir:
                #     save_pose_json(book_data['keypoints'], output_dir=output_dir, label=AUTO_SAVE_LABEL,
                #                    roi_image=book_data.get('pose_roi_image'))

            cv2.imshow('Book Recognition', frame_with_results)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # elif key == ord('c') or key == ord('C'):
            #     # C 鍵：標註「我現在是坐著」存檔（供離線驗證）
            #     if book_data['keypoints'] and output_dir:
            #         save_pose_json(book_data['keypoints'], output_dir=output_dir, label="sitting",
            #                        roi_image=book_data.get('pose_roi_image'))
            #     else:
            #         print("No pose data to save")
            # elif key == ord('v') or key == ord('V'):
            #     # V 鍵：標註「我現在是站著」存檔（供離線驗證）
            #     if book_data['keypoints'] and output_dir:
            #         save_pose_json(book_data['keypoints'], output_dir=output_dir, label="standing",
            #                        roi_image=book_data.get('pose_roi_image'))
            #     else:
            #         print("No pose data to save")
            elif key == 9:  # TAB
                active_roi = "pose" if active_roi == "book" else "book"
                print(f"Switched to {active_roi.upper()} ROI")
            elif key == ord('w'):
                if active_roi == "book":
                    book_roi_y = max(0, book_roi_y - 5)
                else:
                    pose_roi_y = max(0, pose_roi_y - 5)
            elif key == ord('s'):
                if active_roi == "book":
                    book_roi_y = min(frame.shape[0] - book_roi_size, book_roi_y + 5)
                else:
                    pose_roi_y = min(frame.shape[0] - pose_roi_size, pose_roi_y + 5)
            elif key == ord('a'):
                if active_roi == "book":
                    book_roi_x = max(0, book_roi_x - 5)
                else:
                    pose_roi_x = max(0, pose_roi_x - 5)
            elif key == ord('d'):
                if active_roi == "book":
                    book_roi_x = min(frame.shape[1] - book_roi_size, book_roi_x + 5)
                else:
                    pose_roi_x = min(frame.shape[1] - pose_roi_size, pose_roi_x + 5)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Barkley Tracker (WebCam)")
    parser.add_argument("--output", default=None, help="輸出資料夾名稱（相對於根目錄，未指定則不存檔）")
    parser.add_argument("--camera", type=int, default=1, help="相機索引（預設：1）")
    args = parser.parse_args()

    log_file = setup_logging()
    logging.info("=" * 60)
    logging.info("Barkley Tracker 啟動")
    logging.info(f"Log 檔案：{log_file}")
    if args.output:
        logging.info(f"輸出資料夾：{args.output}")
    else:
        logging.info("模式：不存檔（只顯示 console log）")
    logging.info("=" * 60)

    # list_all_cameras()
    cap = open_camera(args.camera)
    if cap is not None:
        run_tracker_in_yolo(cap, output_dir=args.output)
    else:
        logging.error("Unable to open camera at index 1")

    logging.info("=" * 60)
    logging.info("Barkley Tracker 已關閉")
    logging.info("=" * 60)
