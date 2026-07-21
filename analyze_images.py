"""
圖片批量分析：讀取圖檔資料，做 YOLO pose 推論 + 坐姿判定
產生與 barkley_openvino.py 相同格式的 JSON 檔（覆蓋舊檔案）

使用方式：
  python analyze_images.py [image_dir1] [image_dir2] ...

範例：
  python analyze_images.py sit_pose_photo stand_pose_photo
  python analyze_images.py ./captures
"""

import os
import sys
import cv2
import json
import glob
import logging
from datetime import datetime
from ultralytics import YOLO
from pose_detector import is_sitting_pose, reset_body_scale_history
from openvino.runtime import Core

# ============================================================
# 設定
# ============================================================
YOLO_OV_MODEL = "yolo11x-pose_openvino_model"
YOLO_POSE_DEVICE = "GPU"
# ============================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )

def _check_models():
    if not os.path.exists(YOLO_OV_MODEL):
        print(f"找不到 {YOLO_OV_MODEL}，請先執行轉換腳本：")
        print("  python export_openvino.py")
        raise FileNotFoundError("OpenVINO 模型不存在")

def load_yolo_model():
    """載入 YOLO pose 模型"""
    _check_models()
    try:
        yolo_pose = YOLO(YOLO_OV_MODEL, task="pose")
        logging.info("✓ YOLO pose 模型已載入")

        # 檢查可用裝置
        try:
            ie = Core()
            available = ie.available_devices
            if YOLO_POSE_DEVICE.upper() == "GPU" and "GPU" in available:
                logging.info(f"✓ 推論裝置: GPU (可用: {available})")
            else:
                logging.info(f"✓ 推論裝置: CPU (可用: {available})")
        except Exception as e:
            logging.warning(f"無法檢查裝置: {e}")

        return yolo_pose
    except Exception as e:
        logging.error(f"YOLO pose 模型載入失敗：{e}")
        raise

def convert_yolo_result_to_persons(yolo_result, box_confidence=None):
    """
    將 YOLO 推論結果轉換為 JSON 格式

    預期格式（同 barkley_openvino.py）：
    {
        "person_id": int,
        "box_confidence": float,
        "keypoints": [
            {"name": "...", "x": float, "y": float, "confidence": float},
            ...
        ]
    }
    """
    if yolo_result.keypoints is None or yolo_result.keypoints.xy is None:
        return None

    keypoint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    persons = []
    boxes = yolo_result.boxes

    for person_idx, (xy, conf) in enumerate(zip(yolo_result.keypoints.xy, yolo_result.keypoints.conf)):
        keypoints_list = []
        for kp_idx, (x, y) in enumerate(xy):
            kp_confidence = float(conf[kp_idx]) if conf is not None else 1.0
            keypoints_list.append({
                "name": keypoint_names[kp_idx],
                "x": float(x),
                "y": float(y),
                "confidence": kp_confidence
            })

        # 從 YOLO box 提取人物信心度
        box_conf = float(boxes.conf[person_idx]) if boxes.conf is not None else 0.9

        person_dict = {
            "person_id": person_idx,
            "box_confidence": box_conf,
            "keypoints": keypoints_list
        }
        persons.append(person_dict)

    return persons if persons else None


def analyze_and_save_image(image_path, yolo_model, output_dir):
    """
    分析單張圖片並保存 JSON

    回傳 (success, message)
    """
    try:
        # 讀取圖片
        frame = cv2.imread(image_path)
        if frame is None:
            return False, f"無法讀取圖片"

        # YOLO 推論
        yolo_results = yolo_model(frame, verbose=False)
        if not yolo_results or len(yolo_results) == 0:
            return False, "YOLO 未偵測到人物"

        yolo_result = yolo_results[0]

        # 轉換為 JSON 格式
        persons = convert_yolo_result_to_persons(yolo_result)
        if persons is None:
            return False, "無法轉換關鍵點"

        # 對每個人做姿態判定 + 附加 sitting_detection
        for person in persons:
            status, debug_info = is_sitting_pose(person, verbose=False, return_debug=True)
            person["sitting_detection"] = {
                "status": status,
                "debug": debug_info
            }

        # 構建 JSON 結構（同 barkley_openvino.py）
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "label": None,  # 新分析的圖片不帶標籤
            "num_persons": len(persons),
            "persons": persons
        }

        # 決定輸出檔名：取原圖名稱，副檔名改成 .json
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.json")

        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)

        # 寫入 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # 統計姿態判定結果
        status_counts = {}
        for p in persons:
            status = p["sitting_detection"]["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        status_str = ", ".join(f"{status}:{count}" for status, count in sorted(status_counts.items()))
        return True, f"{len(persons)} 人 ({status_str})"

    except Exception as e:
        return False, f"處理失敗: {str(e)}"

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="批量分析圖片並產生 JSON（覆蓋舊檔案）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python analyze_images.py sit_pose_photo stand_pose_photo
  python analyze_images.py ./captures
        """
    )
    parser.add_argument("image_dirs", nargs="+", help="一個或多個圖片目錄")

    args = parser.parse_args()

    setup_logging()

    # 載入模型
    logging.info("載入 YOLO pose 模型...")
    yolo_model = load_yolo_model()

    # 重置基準（開始新一輪分析）
    logging.info("重置肩寬基準...")
    reset_body_scale_history()

    # 批量處理所有目錄
    total_images = 0
    total_success = 0
    total_failed = 0

    for image_dir in args.image_dirs:
        if not os.path.isdir(image_dir):
            logging.warning(f"⚠ 跳過：目錄不存在 - {image_dir}")
            continue

        # 找所有圖片（避免重複）
        image_files = set()
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.update(glob.glob(os.path.join(image_dir, ext)))
            image_files.update(glob.glob(os.path.join(image_dir, ext.upper())))

        image_files = sorted(list(image_files))

        if not image_files:
            logging.warning(f"⚠ 跳過：找不到圖片 - {image_dir}")
            continue
        logging.info(f"\n分析 {image_dir} ({len(image_files)} 張圖片)")
        logging.info("-" * 70)

        success = 0
        failed = 0

        for image_path in image_files:
            is_ok, msg = analyze_and_save_image(image_path, yolo_model, image_dir)
            total_images += 1

            status = "✓" if is_ok else "✗"
            basename = os.path.basename(image_path)

            if is_ok:
                success += 1
                total_success += 1
                logging.info(f"{status} {basename:<20} → {msg}")
            else:
                failed += 1
                total_failed += 1
                logging.info(f"{status} {basename:<20} → 失敗: {msg}")

        logging.info(f"\n小計：{success} 成功，{failed} 失敗")

    # 最終統計
    logging.info("\n" + "=" * 70)
    logging.info(f"總計: {total_images} 張圖片 ({total_success} 成功，{total_failed} 失敗)")
    if total_success > 0:
        logging.info("✓ 所有 JSON 已產生/更新")

if __name__ == "__main__":
    main()
