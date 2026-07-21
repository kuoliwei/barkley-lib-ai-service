import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from queue import Queue

import cv2
import websockets

from barkley_openvino import BarkleyTracker, setup_logging, open_camera
from pose_detector import reset_body_scale_history


# ============================================================
# ROI 設定（唯一來源）
# 位置以「中心點」表示；程式內部會換算成左上角
# ============================================================
POSE_ROI_CENTER_X = 960   # None = 啟動時置中於畫面中央
POSE_ROI_CENTER_Y = 660   # None = 啟動時置中於畫面中央
POSE_ROI_SIZE = 768        # 人體姿態 ROI 邊長
BOOK_ROI_CENTER_X = 782    # 書籍 ROI 中心 X（等同舊左上角 100 + 224/2）
BOOK_ROI_CENTER_Y = 165    # 書籍 ROI 中心 Y（等同舊左上角 428 + 224/2）
BOOK_ROI_SIZE = 224        # 書籍分類 ROI 邊長
# ============================================================


class BarkleyWebSocketClient:
    def __init__(self, uri, camera_index,
                 pose_roi_cx=POSE_ROI_CENTER_X, pose_roi_cy=POSE_ROI_CENTER_Y,
                 book_roi_cx=BOOK_ROI_CENTER_X, book_roi_cy=BOOK_ROI_CENTER_Y):
        self.uri = uri
        self.camera_index = camera_index
        # ROI 中心點（pose 為 None 表示啟動時置中於畫面中央）
        self.pose_roi_cx = pose_roi_cx
        self.pose_roi_cy = pose_roi_cy
        self.book_roi_cx = book_roi_cx
        self.book_roi_cy = book_roi_cy
        self.tracker = None
        self.cap = None
        self.ws = None
        self.running = False
        self.frame_thread = None
        self.result_queue = Queue()

    def setup(self):
        log_file = setup_logging()
        logging.info("=" * 60)
        logging.info("Barkley WebSocket Client 啟動")
        logging.info(f"Log 檔案：{log_file}")
        logging.info(f"連接伺服器：{self.uri}")
        logging.info("=" * 60)

        self.tracker = BarkleyTracker()
        self.cap = open_camera(self.camera_index)

        if self.cap is None:
            logging.error(f"無法打開攝像頭 {self.camera_index}")
            raise RuntimeError("Camera initialization failed")

        reset_body_scale_history()
        logging.info("模型和攝像頭初始化完成")

    def extract_person_status(self, book_data):
        """從推論結果提取 person 狀態"""
        if not book_data or not book_data.get("status_smoothed"):
            return "null"
        status = book_data["status_smoothed"]
        if status == "sitting":
            return "sit-down"
        return "null"

    def extract_book_class(self, book_data):
        """從推論結果提取 book 類別"""
        if not book_data:
            return "null"
        book_class = book_data.get("book_class")
        if book_class == 0:
            return "null"
        return str(book_class - 1) if book_class > 0 else "null"

    def frame_capture_loop(self, inference_interval):
        """在獨立 thread 中持續讀取和推論幀，推論完直接存入佇列，同時顯示 OpenCV 監控視窗

        ROI 設定一律取自模組頂端常數（POSE_ROI_*, BOOK_ROI_*）。
        pose/book 的座標會隨自動置中與 WASD 鍵調整，故複製成區域變數。
        """
        try:
            if not self.cap.isOpened():
                logging.error("攝像頭連線已中斷")
                return

            pose_roi_size = POSE_ROI_SIZE
            book_roi_size = BOOK_ROI_SIZE
            # 左上角座標，於首幀由中心點換算後填入（WASD 之後直接調整左上角）
            pose_roi_x = pose_roi_y = None
            book_roi_x = book_roi_y = None

            active_roi = "book"
            first_frame = True
            last_inference_time = time.time()
            book_data = None
            frame_with_results = None

            # 建立視窗（不強制置頂，避免蓋住其他視窗）
            cv2.namedWindow('Book Recognition', cv2.WINDOW_AUTOSIZE)

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("無法讀取幀")
                    break

                if first_frame:
                    h, w = frame.shape[:2]
                    logging.info(f"實際攝像頭解析度: {w}×{h}")
                    if inference_interval > 0:
                        logging.info(f"幀採樣間隔: {inference_interval} 秒")
                    first_frame = False

                    # 中心點 → 左上角。pose 中心為 None 時置中於畫面中央
                    if self.pose_roi_cx is None:
                        pose_roi_x = (w - pose_roi_size) // 2
                    else:
                        pose_roi_x = self.pose_roi_cx - pose_roi_size // 2
                    if self.pose_roi_cy is None:
                        pose_roi_y = (h - pose_roi_size) // 2
                    else:
                        pose_roi_y = self.pose_roi_cy - pose_roi_size // 2

                    book_roi_x = self.book_roi_cx - book_roi_size // 2
                    book_roi_y = self.book_roi_cy - book_roi_size // 2

                    logging.info(f"pose ROI 左上角: ({pose_roi_x}, {pose_roi_y}) | book ROI 左上角: ({book_roi_x}, {book_roi_y})")

                h, w = frame.shape[:2]
                if w != 1920 or h != 1080:
                    frame = cv2.resize(frame, (1920, 1080))

                current_time = time.time()
                should_infer = False

                if inference_interval <= 0:
                    should_infer = True
                elif current_time - last_inference_time >= inference_interval:
                    should_infer = True
                    last_inference_time = current_time

                if should_infer:
                    try:
                        frame_with_results, book_data = self.tracker.process_frame(
                            frame, pose_roi_x, pose_roi_y, book_roi_x, book_roi_y,
                            pose_roi_size, book_roi_size
                        )
                        self.result_queue.put(book_data)
                    except Exception as e:
                        logging.error(f"推論失敗: {e}")
                else:
                    # 不推論時使用上一次的結果
                    if frame_with_results is None:
                        frame_with_results, book_data = self.tracker.process_frame(
                            frame, pose_roi_x, pose_roi_y, book_roi_x, book_roi_y,
                            pose_roi_size, book_roi_size
                        )
                        last_inference_time = current_time

                # 顯示 UI 文字
                status_text = f"Active ROI: {active_roi.upper()} (Press TAB to switch)"
                cv2.putText(frame_with_results, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if inference_interval > 0:
                    sample_text = f"Inference interval: {inference_interval}s {'[推論中]' if should_infer else '[使用快取]'}"
                    cv2.putText(frame_with_results, sample_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if should_infer else (0, 165, 255), 1)

                # 在每個 ROI 上標註目前中心座標（WASD 移動後即時更新，方便回填設定檔）
                pose_cx = pose_roi_x + pose_roi_size // 2
                pose_cy = pose_roi_y + pose_roi_size // 2
                book_cx = book_roi_x + book_roi_size // 2
                book_cy = book_roi_y + book_roi_size // 2
                cv2.putText(frame_with_results, f"pose center: ({pose_cx}, {pose_cy})",
                           (pose_roi_x + 10, pose_roi_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame_with_results, f"book center: ({book_cx}, {book_cy})",
                           (book_roi_x, book_roi_y + book_roi_size + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow('Book Recognition', frame_with_results)

                # 鍵盤控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == 9:  # TAB
                    active_roi = "pose" if active_roi == "book" else "book"
                    logging.info(f"切換到 {active_roi.upper()} ROI")
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

                # 每次推論時列印結果
                if should_infer and book_data is not None:
                    status_str = ""
                    if book_data['status_list']:
                        status_results = []
                        for i, status in enumerate(book_data['status_list']):
                            status_results.append(f"P{i}:{status}")
                        status_str = " | Status: " + ", ".join(status_results)
                        smoothed = book_data.get('status_smoothed')
                        if smoothed is not None:
                            status_str += f" | Smoothed: {smoothed}"
                    else:
                        status_str = " | Status: N/A"

                    log_msg = f"Pose:{status_str}"
                    logging.info(log_msg)

        except Exception as e:
            logging.error(f"frame_capture_loop 錯誤: {e}")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

    async def send_results(self):
        """連接到 Unity WebSocket Server 並發送推論結果"""
        async with websockets.connect(self.uri) as websocket:
            self.ws = websocket
            logging.info(f"已連接到伺服器: {self.uri}")

            try:
                while self.running:
                    # 從佇列取出推論結果（非阻塞）
                    try:
                        book_data = self.result_queue.get_nowait()

                        person_status = self.extract_person_status(book_data)
                        book_class = self.extract_book_class(book_data)

                        message = {
                            "book": book_class,
                            "person": person_status
                        }

                        try:
                            await websocket.send(json.dumps(message))
                            logging.info(f"發送: {json.dumps(message)}")
                        except websockets.exceptions.ConnectionClosed:
                            logging.error("伺服器連線已中斷")
                            break
                    except:
                        # 佇列為空，等待下一個結果
                        await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logging.info("發送迴圈已取消")
            except Exception as e:
                logging.error(f"send_results 錯誤: {e}")
            finally:
                self.ws = None

    async def run(self, inference_interval):
        """主執行迴圈"""
        self.running = True
        self.latest_result = None

        self.frame_thread = threading.Thread(
            target=self.frame_capture_loop,
            args=(inference_interval,),
            daemon=True
        )
        self.frame_thread.start()
        logging.info("幀捕獲 thread 已啟動")

        try:
            while self.running:
                try:
                    await self.send_results()
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("連線中斷，嘗試重新連接...")
                    await asyncio.sleep(2)
                except Exception as e:
                    logging.error(f"連接錯誤: {e}")
                    await asyncio.sleep(2)

        except KeyboardInterrupt:
            logging.info("收到中斷信號")
        finally:
            self.running = False
            if self.frame_thread:
                self.frame_thread.join(timeout=5)

            logging.info("=" * 60)
            logging.info("Barkley WebSocket Client 已關閉")
            logging.info("=" * 60)

    def shutdown(self):
        """優雅關閉"""
        self.running = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Barkley WebSocket Client - 連接 Unity Server")
    parser.add_argument("--server", default="ws://localhost:8888/Chat", help="WebSocket 伺服器地址（預設: ws://localhost:8888/Chat）")
    parser.add_argument("--camera", type=int, default=0, help="攝像頭索引（預設: 0 = Elgato Facecam）")
    parser.add_argument("--inference-interval", type=float, default=1.0, help="推論間隔（秒）（預設: 1.0）")
    parser.add_argument("--pose-roi-x", type=int, default=POSE_ROI_CENTER_X, help="人體姿態 ROI 中心 X（預設: 畫面置中）")
    parser.add_argument("--pose-roi-y", type=int, default=POSE_ROI_CENTER_Y, help="人體姿態 ROI 中心 Y（預設: 畫面置中）")
    parser.add_argument("--book-roi-x", type=int, default=BOOK_ROI_CENTER_X, help=f"書籍 ROI 中心 X（預設: {BOOK_ROI_CENTER_X}）")
    parser.add_argument("--book-roi-y", type=int, default=BOOK_ROI_CENTER_Y, help=f"書籍 ROI 中心 Y（預設: {BOOK_ROI_CENTER_Y}）")
    args = parser.parse_args()

    client = BarkleyWebSocketClient(
        uri=args.server,
        camera_index=args.camera,
        pose_roi_cx=args.pose_roi_x,
        pose_roi_cy=args.pose_roi_y,
        book_roi_cx=args.book_roi_x,
        book_roi_cy=args.book_roi_y,
    )

    try:
        client.setup()
        asyncio.run(client.run(inference_interval=args.inference_interval))
    except Exception as e:
        logging.error(f"程式錯誤: {e}")
    finally:
        client.shutdown()
