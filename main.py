from multiprocessing import Process, Queue
from barkley import run_tracker_in_yolo, ans_to_websocket
if __name__ == "__main__":
    uri = "ws://localhost:8888/Chat"#"ws://192.168.50.234:8888/Chat"#"ws://localhost:7777/Chat"
    ans_queues = Queue()
    tracker_yolo = Process(
        target=run_tracker_in_yolo,
        args=(ans_queues,),
        daemon=True
    )
    
    tracker_yolo.start()
    print("tracker_yolo 已啟動!")
    tracker_websocket = Process(
        target=ans_to_websocket,
        args=(ans_queues, uri,),
        daemon=True
    )

    tracker_websocket.start()
    print("tracker_websocket 已啟動!")
    tracker_yolo.join()
    tracker_yolo.terminate()
    tracker_websocket.join()
    tracker_websocket.terminate()