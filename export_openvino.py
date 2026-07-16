"""
一次性轉換腳本：將 YOLO 和 ResNet18 匯出為 OpenVINO 格式
執行一次即可，之後 barkley_openvino.py 直接載入轉換後的模型
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from ultralytics import YOLO
import os


def export_yolo():
    print("=" * 50)
    print("[1/2] 匯出 YOLO pose 模型...")
    output_dir = "yolo11x-pose_openvino_model"
    if os.path.exists(output_dir):
        print(f"  已存在：{output_dir}，略過")
        return
    model = YOLO("yolo11x-pose.pt")
    model.export(format="openvino")
    print(f"  完成：{output_dir}/")


def export_book_model():
    print("=" * 50)
    print("[2/2] 匯出 ResNet18 書籍分類模型...")
    onnx_path = "barkley_book_v3.onnx"
    xml_path = "barkley_book_v3.xml"

    if os.path.exists(xml_path):
        print(f"  已存在：{xml_path}，略過")
        return

    # 載入 PyTorch 模型
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 4)
    state_dict = torch.load("barkley_book_v3.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 匯出 ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"  ONNX 已輸出：{onnx_path}")

    # ONNX 轉 OpenVINO IR
    try:
        import openvino as ov
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, xml_path)
        print(f"  OpenVINO IR 已輸出：{xml_path}")
    except Exception as e:
        print(f"  轉換失敗：{e}")
        return

    # 清理暫存 ONNX
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"  已清理暫存檔：{onnx_path}")


if __name__ == "__main__":
    print("OpenVINO 模型轉換工具")
    print("=" * 50)
    export_yolo()
    export_book_model()
    print("=" * 50)
    print("全部完成！可以執行 barkley_openvino.py")
