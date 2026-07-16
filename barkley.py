import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import resnet18
from torchvision import transforms
import json
import os
from datetime import datetime
from pose_detector import is_sitting_pose

class BarkleyTracker:
    def __init__(self):
        self.yolo_pose = YOLO('yolo11x-pose.pt')
        self.book_model = self._load_book_model()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_book_model(self):
        """加載書籍分類模型"""
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 4)
        state_dict = torch.load('barkley_book_v3.pth', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def draw_keypoints(self, frame, keypoints, pose_roi_x, pose_roi_y, roi_size=224):
        """在 ROI 區域內畫出關鍵點"""
        if keypoints is None or keypoints.xy is None:
            return frame

        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # YOLO 返回的座標是相對於 ROI 的，需要加上偏移量轉換為全畫面座標
        for person_keypoints in keypoints.xy:
            for kp_idx, (x, y) in enumerate(person_keypoints):
                x, y = int(x), int(y)
                # 轉換為全畫面座標
                frame_x = x + pose_roi_x
                frame_y = y + pose_roi_y
                cv2.circle(frame, (frame_x, frame_y), 4, (0, 255, 255), -1)
                cv2.putText(frame, keypoint_names[kp_idx], (frame_x + 5, frame_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        return frame

    def extract_pose_roi(self, frame, roi_x, roi_y, roi_size=224):
        """從畫面中切割人體 ROI"""
        h, w = frame.shape[:2]
        x1 = max(0, roi_x)
        y1 = max(0, roi_y)
        x2 = min(w, roi_x + roi_size)
        y2 = min(h, roi_y + roi_size)
        return frame[y1:y2, x1:x2]

    def extract_book_roi(self, frame, roi_x=100, roi_y=100, roi_size=224):
        """從畫面中切割書籍區域"""
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
        """分類書籍"""
        from PIL import Image
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess(roi_pil).unsqueeze(0)

        with torch.no_grad():
            output = self.book_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            all_probs = probabilities[0].cpu().numpy()

        return predicted_class, confidence, all_probs

    def process_frame(self, frame, pose_roi_x=100, pose_roi_y=100, book_roi_x=100, book_roi_y=100, pose_roi_size=448, book_roi_size=224):
        """處理單幀圖像"""
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # YOLO 人體姿態檢測（使用人體 ROI）
        pose_roi = self.extract_pose_roi(frame, pose_roi_x, pose_roi_y, pose_roi_size)
        results = self.yolo_pose(pose_roi, verbose=False)

        num_persons = 0
        keypoints_data = []
        for result in results:
            num_persons = len(result.boxes) if result.boxes is not None else 0
            if result.keypoints is not None:
                frame = self.draw_keypoints(frame, result.keypoints, pose_roi_x, pose_roi_y, pose_roi_size)
                # 收集關鍵點資訊
                for person_idx, person_keypoints in enumerate(result.keypoints.xy):
                    person_data = {
                        "person_id": person_idx,
                        "keypoints": []
                    }
                    for kp_idx, (x, y) in enumerate(person_keypoints):
                        person_data["keypoints"].append({
                            "name": keypoint_names[kp_idx],
                            "x": float(x),
                            "y": float(y)
                        })
                    keypoints_data.append(person_data)

        # 書籍識別
        book_roi = self.extract_book_roi(frame, book_roi_x, book_roi_y, book_roi_size)
        book_class, confidence, all_probs = self.classify_book(book_roi)

        book_names = ['no_book', 'book_1', 'book_2', 'book_3']

        # 畫人體 ROI 框（黃色）
        cv2.rectangle(frame, (pose_roi_x, pose_roi_y), (pose_roi_x + pose_roi_size, pose_roi_y + pose_roi_size), (0, 255, 255), 2)
        cv2.putText(frame, f"Pose ROI ({num_persons} persons)", (pose_roi_x, pose_roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 畫書籍 ROI 框（綠色）
        cv2.rectangle(frame, (book_roi_x, book_roi_y), (book_roi_x + book_roi_size, book_roi_y + book_roi_size), (0, 255, 0), 2)
        text = f"{book_names[book_class]}"
        cv2.putText(frame, text, (book_roi_x, book_roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 檢測坐姿
        is_sitting_list = []
        for person in keypoints_data:
            is_sitting = is_sitting_pose(person, verbose=False)
            is_sitting_list.append(is_sitting)

        return frame, {
            "book_class": book_class,
            "book_name": book_names[book_class],
            "confidence": confidence,
            "num_persons": num_persons,
            "all_probs": all_probs,
            "keypoints": keypoints_data,
            "is_sitting": is_sitting_list
        }


def list_all_cameras():
    """列出所有可用的攝像頭"""
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
    """打開指定索引的攝像頭並設置分辨率"""
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


def save_pose_json(keypoints_data, output_dir="posejson"):
    """保存姿態資訊為 JSON 檔案，使用流水號，並包含坐姿檢測調試資訊"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)

    # 尋找下一個可用的流水號
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    if existing_files:
        # 提取編號
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

    # 為每個人添加坐姿檢測調試資訊
    persons_with_debug = []
    for person in keypoints_data:
        person_copy = person.copy()
        is_sitting, debug_info = is_sitting_pose(person, verbose=False, return_debug=True)
        person_copy["sitting_detection"] = {
            "is_sitting": is_sitting,
            "debug": debug_info
        }
        persons_with_debug.append(person_copy)

    # 建立 JSON 資料
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "num_persons": len(persons_with_debug),
        "persons": persons_with_debug
    }

    # 保存檔案
    filename = os.path.join(output_dir, f"{next_num:03d}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved pose data to: {filename}")


def run_tracker_in_yolo(cap, book_roi_x=100, book_roi_y=428, pose_roi_x=736, pose_roi_y=316, book_roi_size=224, pose_roi_size=448):
    """主循環：持續抓取攝像頭並處理"""
    import time
    tracker = BarkleyTracker()

    if not cap.isOpened():
        print("Error: 無法打開攝像頭")
        return

    active_roi = "book"  # 預設操作書籍 ROI

    try:
        first_frame = True
        last_print_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if first_frame:
                h, w = frame.shape[:2]
                print(f"Actual camera resolution: {w}×{h}")
                first_frame = False

            h, w = frame.shape[:2]
            if w != 1920 or h != 1080:
                frame = cv2.resize(frame, (1920, 1080))

            frame_with_results, book_data = tracker.process_frame(
                frame, pose_roi_x, pose_roi_y, book_roi_x, book_roi_y, pose_roi_size, book_roi_size
            )

            # 在左上角顯示當前操作的 ROI
            status_text = f"Active ROI: {active_roi.upper()} (Press TAB to switch)"
            cv2.putText(frame_with_results, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 每隔 1 秒打印一次結果
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                book_names_print = ['no_book', 'book_1', 'book_2', 'book_3']
                probs_str = ", ".join([f"{name}: {prob:.4f}" for name, prob in zip(book_names_print, book_data['all_probs'])])

                # 坐姿檢測結果
                sitting_str = ""
                if book_data['is_sitting']:
                    sitting_results = []
                    for i, is_sitting in enumerate(book_data['is_sitting']):
                        status = "sitting" if is_sitting else "not_sitting"
                        sitting_results.append(f"P{i}:{status}")
                    sitting_str = " | Sitting: " + ", ".join(sitting_results)
                else:
                    sitting_str = " | Sitting: N/A"

                print(f"Pose: {book_data['num_persons']} persons | Book: {probs_str}{sitting_str}")
                last_print_time = current_time

            cv2.imshow('Book Recognition', frame_with_results)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') or key == ord('C'):  # C 鍵 - 捕捉姿態
                if book_data['keypoints']:
                    save_pose_json(book_data['keypoints'])
                else:
                    print("No pose data to save")
            elif key == 9:  # TAB 鍵
                active_roi = "pose" if active_roi == "book" else "book"
                print(f"Switched to {active_roi.upper()} ROI")
            elif key == ord('w'):  # 上
                if active_roi == "book":
                    book_roi_y = max(0, book_roi_y - 5)
                else:
                    pose_roi_y = max(0, pose_roi_y - 5)
            elif key == ord('s'):  # 下
                if active_roi == "book":
                    book_roi_y = min(frame.shape[0] - book_roi_size, book_roi_y + 5)
                else:
                    pose_roi_y = min(frame.shape[0] - pose_roi_size, pose_roi_y + 5)
            elif key == ord('a'):  # 左
                if active_roi == "book":
                    book_roi_x = max(0, book_roi_x - 5)
                else:
                    pose_roi_x = max(0, pose_roi_x - 5)
            elif key == ord('d'):  # 右
                if active_roi == "book":
                    book_roi_x = min(frame.shape[1] - book_roi_size, book_roi_x + 5)
                else:
                    pose_roi_x = min(frame.shape[1] - pose_roi_size, pose_roi_x + 5)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # list_all_cameras()
    cap = open_camera(1)  # OBS Virtual Camera at index 1
    if cap is not None:
        run_tracker_in_yolo(cap)
    else:
        print("Unable to open camera at index 1")
