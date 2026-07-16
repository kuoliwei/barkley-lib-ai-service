# 書籍辨識模型訓練配置

## 模型規格
- 架構：ResNet18
- 輸入：224×224 RGB
- 輸出類別：4
- 類別映射：
  - 0: no_book
  - 1: book_1
  - 2: book_2
  - 3: book_3

## 訓練資料結構
```
training_data/
├── no_book/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ... (50-100 張)
├── book_1/
│   ├── 001.jpg
│   └── ... (50-100 張)
├── book_2/
│   ├── 001.jpg
│   └── ... (50-100 張)
└── book_3/
    ├── 001.jpg
    └── ... (50-100 張)
```

## 訓練代碼

```python
from fastai.vision.all import *

# 加載資料
dls = ImageDataLoaders.from_folder(
    'training_data/',
    train='.',
    valid_pct=0.2,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms()
)

# 建立模型
learn = vision_learner(dls, resnet18, metrics=error_rate)

# 訓練
learn.fine_tune(4)

# 匯出
learn.export('book_model_new.pkl')
```

## 模型轉換

```python
import torch
from torchvision.models import resnet18
import torch.nn as nn

learn = load_learner('book_model_new.pkl')
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 4)
model.load_state_dict(learn.model.state_dict())
torch.save(model.state_dict(), 'barkley_book_new.pth')
```

## 系統整合

1. 將 `barkley_book_new.pth` 覆蓋 `barkley_book_v3.pth`
2. 更新 `barkley.py` 中的類別標籤：
   ```python
   book_names = ['no_book', 'book_1', 'book_2', 'book_3']
   ```
3. 測試推理流程

## 參數設定
- 輸入分辨率：224×224
- 預處理：ImageNet 標準化
- 迴圈數：4
- 驗證集比例：20%
- 最少資料量：每類 50 張
- 推薦資料量：每類 100+ 張
