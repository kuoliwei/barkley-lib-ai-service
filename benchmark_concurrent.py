"""
並發推論性能測試

測試 YOLO 和書籍分類同時推論時的性能
比較：(CPU, CPU) vs (GPU, GPU)
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
from torchvision import transforms
from PIL import Image


# ============================================================
# 測試設定
# ============================================================
YOLO_OV_MODEL = "yolo11x-pose_openvino_model"
BOOK_OV_XML = "barkley_book_v3.xml"
NUM_ITERATIONS = 10  # 每個測試跑的次數
# ============================================================


def setup_logging():
    """設定日誌"""
    log_file = f"benchmark_concurrent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_file


def benchmark_concurrent(yolo_device="GPU", book_device="GPU"):
    """
    測試 YOLO 和書籍分類同時推論

    模擬真實場景：
    1. 同時載入兩個模型
    2. 連續推論 NUM_ITERATIONS 次
    3. 每次都要推論 YOLO + 書籍分類
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"並發推論測試 - YOLO({yolo_device}) + 書籍({book_device})")
    logging.info(f"{'='*60}")

    ie = Core()
    available_devices = ie.available_devices
    logging.info(f"可用裝置: {available_devices}")

    # 設定目標裝置
    yolo_target = "GPU.0" if yolo_device.upper() == "GPU" else "CPU"
    book_target = "GPU.0" if book_device.upper() == "GPU" else "CPU"

    if yolo_target == "GPU.0" and "GPU" not in available_devices:
        yolo_target = "CPU"
    if book_target == "GPU.0" and "GPU" not in available_devices:
        book_target = "CPU"

    try:
        # 載入 YOLO 模型
        logging.info(f"載入 YOLO 到 {yolo_target}...")
        yolo_model = ie.read_model(YOLO_OV_MODEL + "/yolo11x-pose.xml")
        yolo_compiled = ie.compile_model(yolo_model, yolo_target)
        logging.info(f"✓ YOLO 載入成功")

        # 載入書籍分類模型
        logging.info(f"載入書籍分類到 {book_target}...")
        book_model = ie.read_model(BOOK_OV_XML)
        book_compiled = ie.compile_model(book_model, book_target)
        logging.info(f"✓ 書籍分類載入成功")
    except Exception as e:
        logging.error(f"✗ 模型載入失敗: {e}")
        return None

    # 建立虛擬輸入
    yolo_input = np.random.randint(0, 255, (1, 3, 640, 640), dtype=np.uint8).astype(np.float32)
    book_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # 預處理書籍圖像
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    book_pil = Image.fromarray(cv2.cvtColor(book_frame, cv2.COLOR_BGR2RGB))
    book_input = preprocess(book_pil).unsqueeze(0).numpy()

    # 建立推論請求
    yolo_request = yolo_compiled.create_infer_request()
    book_request = book_compiled.create_infer_request()

    # 預熱
    logging.info("預熱推論...")
    yolo_request.infer({0: yolo_input})
    book_request.infer({0: book_input})

    # 測試：順序推論（先 YOLO 再書籍）
    logging.info(f"\n執行 {NUM_ITERATIONS} 次並發推論（順序執行）...")
    times_yolo = []
    times_book = []
    times_total = []

    for i in range(NUM_ITERATIONS):
        start_total = time.time()

        # YOLO 推論
        start_yolo = time.time()
        yolo_request.infer({0: yolo_input})
        elapsed_yolo = time.time() - start_yolo
        times_yolo.append(elapsed_yolo)

        # 書籍分類推論
        start_book = time.time()
        book_request.infer({0: book_input})
        elapsed_book = time.time() - start_book
        times_book.append(elapsed_book)

        elapsed_total = time.time() - start_total
        times_total.append(elapsed_total)

        logging.info(f"  第 {i+1}/{NUM_ITERATIONS} 次:")
        logging.info(f"    YOLO: {elapsed_yolo*1000:.2f} ms")
        logging.info(f"    書籍: {elapsed_book*1000:.2f} ms")
        logging.info(f"    總計: {elapsed_total*1000:.2f} ms")

    # 統計結果
    times_yolo = np.array(times_yolo)
    times_book = np.array(times_book)
    times_total = np.array(times_total)

    logging.info(f"\n結果統計:")
    logging.info(f"  YOLO:")
    logging.info(f"    平均時間: {times_yolo.mean()*1000:.2f} ms")
    logging.info(f"    範圍: {times_yolo.min()*1000:.2f} - {times_yolo.max()*1000:.2f} ms")

    logging.info(f"  書籍分類:")
    logging.info(f"    平均時間: {times_book.mean()*1000:.2f} ms")
    logging.info(f"    範圍: {times_book.min()*1000:.2f} - {times_book.max()*1000:.2f} ms")

    logging.info(f"  總計（每幀）:")
    logging.info(f"    平均時間: {times_total.mean()*1000:.2f} ms")
    logging.info(f"    範圍: {times_total.min()*1000:.2f} - {times_total.max()*1000:.2f} ms")
    logging.info(f"    FPS: {1.0 / times_total.mean():.2f}")

    return {
        "yolo_device": yolo_device,
        "book_device": book_device,
        "yolo_avg": times_yolo.mean() * 1000,
        "book_avg": times_book.mean() * 1000,
        "total_avg": times_total.mean() * 1000,
        "fps": 1.0 / times_total.mean()
    }


def print_comparison(cpu_cpu, gpu_gpu):
    """列印對比結果"""
    logging.info(f"\n\n{'='*60}")
    logging.info("並發推論對比總結")
    logging.info(f"{'='*60}")

    if cpu_cpu:
        logging.info(f"\nCPU + CPU:")
        logging.info(f"  YOLO:   {cpu_cpu['yolo_avg']:.2f} ms")
        logging.info(f"  書籍:   {cpu_cpu['book_avg']:.2f} ms")
        logging.info(f"  總計:   {cpu_cpu['total_avg']:.2f} ms/幀")
        logging.info(f"  FPS:    {cpu_cpu['fps']:.2f}")

    if gpu_gpu:
        logging.info(f"\nGPU + GPU:")
        logging.info(f"  YOLO:   {gpu_gpu['yolo_avg']:.2f} ms")
        logging.info(f"  書籍:   {gpu_gpu['book_avg']:.2f} ms")
        logging.info(f"  總計:   {gpu_gpu['total_avg']:.2f} ms/幀")
        logging.info(f"  FPS:    {gpu_gpu['fps']:.2f}")

    if gpu_cpu and gpu_gpu:
        if gpu_cpu['total_avg'] < gpu_gpu['total_avg']:
            faster = "混合模式"
            speedup = gpu_cpu['total_avg'] / gpu_gpu['total_avg']
            improvement = (1 - speedup) * 100
        else:
            faster = "全 GPU 模式"
            speedup = gpu_gpu['total_avg'] / gpu_cpu['total_avg']
            improvement = (speedup - 1) * 100

        logging.info(f"\n對比:")
        logging.info(f"  時間差: {abs(gpu_cpu['total_avg'] - gpu_gpu['total_avg']):.2f} ms")
        logging.info(f"  {faster} 快 {improvement:.1f}%")

        logging.info(f"\n建議:")
        if abs(gpu_cpu['total_avg'] - gpu_gpu['total_avg']) < 20:
            logging.info(f"  ⚖️  性能接近，可任選其一")
            logging.info(f"      - 混合模式：避免 GPU 資源競爭，更穩定")
            logging.info(f"      - 全 GPU 模式：整體最快")
        elif gpu_cpu['total_avg'] < gpu_gpu['total_avg']:
            logging.info(f"  ✅ 使用混合模式 (YOLO:GPU, 書籍:CPU)")
        else:
            logging.info(f"  ✅ 使用全 GPU 模式 (YOLO:GPU, 書籍:GPU)")

    logging.info(f"{'='*60}\n")


if __name__ == "__main__":
    log_file = setup_logging()
    logging.info("開始並發推論性能測試")
    logging.info(f"測試次數: {NUM_ITERATIONS}")
    logging.info(f"日誌檔案: {log_file}")

    # 測試混合模式：YOLO(GPU) + 書籍(CPU)
    gpu_cpu = benchmark_concurrent("GPU", "CPU")

    # 測試全 GPU：YOLO(GPU) + 書籍(GPU)
    gpu_gpu = benchmark_concurrent("GPU", "GPU")

    # 列印對比
    print_comparison(gpu_cpu, gpu_gpu)

    logging.info("並發推論測試完成")
    print(f"\n詳細結果已保存到: {log_file}")
