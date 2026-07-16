"""
Barkley 推論效能基準測試

測試 YOLO 和書籍分類模型在 CPU 和 GPU 上的推論速度
"""

import os
import time
import numpy as np
import cv2
import logging
from datetime import datetime

# 設定 OpenVINO
os.environ['OPENVINO_DEVICE_PRIORITIES'] = 'GPU,CPU'

from openvino.runtime import Core
from ultralytics import YOLO
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image


# ============================================================
# 測試設定
# ============================================================
YOLO_OV_MODEL = "yolo11x-pose_openvino_model"
BOOK_OV_XML = "barkley_book_v3.xml"
NUM_ITERATIONS = 10  # 每個測試跑的次數
POSE_ROI_SIZE = 448
BOOK_ROI_SIZE = 224
# ============================================================


def setup_logging():
    """設定日誌"""
    log_file = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_file


def create_dummy_frames():
    """建立虛擬測試圖像"""
    pose_frame = np.random.randint(0, 255, (POSE_ROI_SIZE, POSE_ROI_SIZE, 3), dtype=np.uint8)
    book_frame = np.random.randint(0, 255, (BOOK_ROI_SIZE, BOOK_ROI_SIZE, 3), dtype=np.uint8)
    return pose_frame, book_frame


def benchmark_yolo(device="GPU"):
    """測試 YOLO 推論"""
    logging.info(f"\n{'='*60}")
    logging.info(f"YOLO 姿態檢測 - {device} 測試")
    logging.info(f"{'='*60}")

    ie = Core()
    available_devices = ie.available_devices
    logging.info(f"可用裝置: {available_devices}")

    target_device = "GPU.0" if device.upper() == "GPU" else "CPU"

    # 檢查裝置是否可用
    if target_device == "GPU.0" and "GPU" not in available_devices:
        logging.warning(f"GPU 不可用，改用 CPU")
        target_device = "CPU"
        device = "CPU"

    try:
        # 載入和編譯模型
        xml_path = YOLO_OV_MODEL + "/yolo11x-pose.xml"
        model = ie.read_model(xml_path)
        compiled = ie.compile_model(model, target_device)
        logging.info(f"✓ 模型編譯成功 (裝置: {target_device})")
    except Exception as e:
        logging.error(f"✗ 模型編譯失敗: {e}")
        return None

    # 建立虛擬圖像 (YOLO 需要 [1, 3, 640, 640] 的格式)
    pose_frame = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8).astype(np.float32)

    # 預熱
    logging.info("預熱推論 (1 次)...")
    infer_request = compiled.create_infer_request()
    infer_request.infer({0: pose_frame})

    # 執行基準測試
    logging.info(f"執行 {NUM_ITERATIONS} 次推論（跳過第一次穩定化）...")
    times = []
    for i in range(NUM_ITERATIONS + 1):  # 多執行一次，之後跳過
        start = time.time()
        infer_request.infer({0: pose_frame})
        elapsed = time.time() - start
        if i == 0:
            logging.info(f"  穩定化: {elapsed*1000:.2f} ms (跳過)")
        else:
            times.append(elapsed)
            logging.info(f"  第 {i}/{NUM_ITERATIONS} 次: {elapsed*1000:.2f} ms")

    # 統計結果（只用穩定後的數據）
    times = np.array(times)
    avg_time = times.mean() * 1000  # 轉成 ms
    min_time = times.min() * 1000
    max_time = times.max() * 1000
    std_time = times.std() * 1000
    fps = 1000.0 / avg_time

    logging.info(f"\n結果統計:")
    logging.info(f"  平均時間: {avg_time:.2f} ms")
    logging.info(f"  最小時間: {min_time:.2f} ms")
    logging.info(f"  最大時間: {max_time:.2f} ms")
    logging.info(f"  標準差: {std_time:.2f} ms")
    logging.info(f"  FPS: {fps:.2f}")

    return {
        "device": device,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "fps": fps
    }


def benchmark_book_classification(device="GPU"):
    """測試書籍分類推論"""
    logging.info(f"\n{'='*60}")
    logging.info(f"書籍分類 - {device} 測試")
    logging.info(f"{'='*60}")

    ie = Core()
    available_devices = ie.available_devices
    logging.info(f"可用裝置: {available_devices}")

    target_device = "GPU.0" if device.upper() == "GPU" else "CPU"

    # 檢查裝置是否可用
    if target_device == "GPU.0" and "GPU" not in available_devices:
        logging.warning(f"GPU 不可用，改用 CPU")
        target_device = "CPU"
        device = "CPU"

    try:
        # 載入和編譯模型
        model = ie.read_model(BOOK_OV_XML)
        compiled = ie.compile_model(model, target_device)
        logging.info(f"✓ 模型編譯成功 (裝置: {target_device})")
    except Exception as e:
        logging.error(f"✗ 模型編譯失敗: {e}")
        return None

    # 建立虛擬圖像
    _, book_frame = create_dummy_frames()

    # 預處理
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    book_pil = Image.fromarray(cv2.cvtColor(book_frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(book_pil).unsqueeze(0).numpy()

    # 預熱
    logging.info("預熱推論 (1 次)...")
    infer_request = compiled.create_infer_request()
    infer_request.infer({0: input_tensor})

    # 執行基準測試
    logging.info(f"執行 {NUM_ITERATIONS} 次推論（跳過第一次穩定化）...")
    times = []
    for i in range(NUM_ITERATIONS + 1):  # 多執行一次，之後跳過
        start = time.time()
        infer_request.infer({0: input_tensor})
        elapsed = time.time() - start
        if i == 0:
            logging.info(f"  穩定化: {elapsed*1000:.2f} ms (跳過)")
        else:
            times.append(elapsed)
            logging.info(f"  第 {i}/{NUM_ITERATIONS} 次: {elapsed*1000:.2f} ms")

    # 統計結果（只用穩定後的數據）
    times = np.array(times)
    avg_time = times.mean() * 1000  # 轉成 ms
    min_time = times.min() * 1000
    max_time = times.max() * 1000
    std_time = times.std() * 1000
    fps = 1000.0 / avg_time

    logging.info(f"\n結果統計:")
    logging.info(f"  平均時間: {avg_time:.2f} ms")
    logging.info(f"  最小時間: {min_time:.2f} ms")
    logging.info(f"  最大時間: {max_time:.2f} ms")
    logging.info(f"  標準差: {std_time:.2f} ms")
    logging.info(f"  FPS: {fps:.2f}")

    return {
        "device": device,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "fps": fps
    }


def print_comparison(yolo_cpu, yolo_gpu, book_cpu, book_gpu):
    """列印對比結果"""
    logging.info(f"\n\n{'='*60}")
    logging.info("性能對比總結")
    logging.info(f"{'='*60}")

    if yolo_cpu and yolo_gpu:
        logging.info(f"\nYOLO 姿態檢測:")
        logging.info(f"  CPU: {yolo_cpu['avg_time']:.2f} ms ({yolo_cpu['fps']:.2f} FPS)")
        logging.info(f"  GPU: {yolo_gpu['avg_time']:.2f} ms ({yolo_gpu['fps']:.2f} FPS)")
        speedup = yolo_cpu['avg_time'] / yolo_gpu['avg_time']
        logging.info(f"  加速比: {speedup:.2f}x (GPU 快)")

    if book_cpu and book_gpu:
        logging.info(f"\n書籍分類:")
        logging.info(f"  CPU: {book_cpu['avg_time']:.2f} ms ({book_cpu['fps']:.2f} FPS)")
        logging.info(f"  GPU: {book_gpu['avg_time']:.2f} ms ({book_gpu['fps']:.2f} FPS)")
        speedup = book_cpu['avg_time'] / book_gpu['avg_time']
        logging.info(f"  加速比: {speedup:.2f}x (GPU 快)")

    logging.info(f"\n建議:")
    if yolo_gpu and yolo_gpu['avg_time'] < yolo_cpu['avg_time']:
        logging.info(f"  YOLO: 使用 GPU")
    else:
        logging.info(f"  YOLO: 使用 CPU")

    if book_gpu and book_gpu['avg_time'] < book_cpu['avg_time']:
        logging.info(f"  書籍分類: 使用 GPU")
    else:
        logging.info(f"  書籍分類: 使用 CPU")

    logging.info(f"{'='*60}\n")


if __name__ == "__main__":
    log_file = setup_logging()
    logging.info("開始 Barkley 推論性能基準測試")
    logging.info(f"測試次數: {NUM_ITERATIONS}")
    logging.info(f"日誌檔案: {log_file}")

    # 測試 YOLO
    yolo_cpu = benchmark_yolo("CPU")
    yolo_gpu = benchmark_yolo("GPU")

    # 測試書籍分類
    book_cpu = benchmark_book_classification("CPU")
    book_gpu = benchmark_book_classification("GPU")

    # 列印對比
    print_comparison(yolo_cpu, yolo_gpu, book_cpu, book_gpu)

    logging.info("基準測試完成")
    print(f"\n詳細結果已保存到: {log_file}")
