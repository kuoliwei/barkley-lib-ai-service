# barkley_book_v3.pth 模型規格

## 輸入（Input）
- 格式：RGB 圖像
- 解析度：224×224 像素
- 資料型態：torch.Tensor，shape: (B, 3, 224, 224)
- 預處理：ImageNet 標準化
  - mean: [0.485, 0.456, 0.406]
  - std: [0.229, 0.224, 0.225]

## 輸出（Output）
- 格式：softmax 概率分布
- 類別數：4
- 資料型態：torch.Tensor，shape: (B, 4)
- 值域：[0, 1]，總和為 1
- 類別映射：
  - 0: no_book
  - 1: book_1
  - 2: book_2
  - 3: book_3

## 模型架構
- 基礎：ResNet18
- 特徵提取層：預訓練 ImageNet 權重
- 分類層：Linear(512 → 4)
- 參數量：約 11.7M
- 檔案大小：約 45 MB

## 特性
- 輕量級：推理快速
- 推理時間：~28ms（GPU）/ ~100ms（CPU）
- 準確度：針對特定書籍優化
- 魯棒性：對角度、光線變化有容忍度

## 限制
- 固定輸入尺寸：224×224
- 固定類別數：4（無法動態增加）
- 特定於訓練資料：在訓練條件外表現不保證

## 使用示例

```python
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn

# 加載模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 4)
model.load_state_dict(torch.load('barkley_book_v3.pth'))
model.eval()

# 預處理
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 推理
with torch.no_grad():
    output = model(input_tensor)  # shape: (1, 4)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()
```

## 訓練配置（推測）
- 基礎模型：ImageNet 預訓練 ResNet18
- 微調策略：遷移學習，凍結特徵層
- 訓練資料：各類別約 50-200 張照片
- 訓練輪數：4-10 epoch
- 最佳化器：Adam
- 損失函數：CrossEntropyLoss
