import torch

# 加載模型權重
state = torch.load("barkley_book_v3.pth", map_location="cpu")

print("=" * 60)
print("模型架構驗證")
print("=" * 60)

print("\n所有層結構:")
keys = sorted(state.keys())
for k in keys:
    print(f"  {k}")

print("\n" + "=" * 60)
print("架構判斷:")
print("=" * 60)

# 找出最高的層編號
max_layer1 = -1
max_layer2 = -1
max_layer3 = -1
max_layer4 = -1

for k in keys:
    if k.startswith("layer1."):
        idx = int(k.split(".")[1])
        max_layer1 = max(max_layer1, idx)
    elif k.startswith("layer2."):
        idx = int(k.split(".")[1])
        max_layer2 = max(max_layer2, idx)
    elif k.startswith("layer3."):
        idx = int(k.split(".")[1])
        max_layer3 = max(max_layer3, idx)
    elif k.startswith("layer4."):
        idx = int(k.split(".")[1])
        max_layer4 = max(max_layer4, idx)

print(f"layer1 最大索引: {max_layer1}")
print(f"layer2 最大索引: {max_layer2}")
print(f"layer3 最大索引: {max_layer3}")
print(f"layer4 最大索引: {max_layer4}")

print("\n" + "=" * 60)

# 判斷架構
if max_layer1 == 1 and max_layer2 == 1 and max_layer3 == 1 and max_layer4 == 1:
    print("✓ 確認: ResNet18")
elif max_layer1 == 2 and max_layer3 == 5 and max_layer4 == 2:
    print("✓ 確認: ResNet34")
elif max_layer1 == 2 and max_layer3 == 5 and max_layer4 == 2:
    print("✓ 確認: ResNet50 或 ResNet101")
else:
    print("? 未知架構")

print("=" * 60)
