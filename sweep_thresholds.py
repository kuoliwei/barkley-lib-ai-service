"""
閾值嚴格度掃描：測試「坐姿判定變嚴格」對兩類召回率的影響

對每組閾值組合，重置狀態後依檔名順序重放全部標註樣本，統計：
- 坐姿召回率：真坐姿被判 sitting 的比例（變嚴格會下降）
- 站姿召回率：真站姿被判 standing 的比例（變嚴格會上升）
"""

import glob
import json
import os

import pose_detector
from pose_detector import is_sitting_pose, reset_body_scale_history


def replay(thigh_b, ext_b, torso_b):
    """套用一組閾值重放全部樣本，回傳統計"""
    pose_detector.THIGH_RATIO_BOUNDARY = thigh_b
    pose_detector.EXTENSION_RATIO_BOUNDARY = ext_b
    pose_detector.TORSO_RATIO_BOUNDARY = torso_b
    reset_body_scale_history()

    stats = {"sitting": [0, 0], "standing": [0, 0]}  # label -> [correct, total]
    for fp in sorted(glob.glob(os.path.join("posejson", "*.json"))):
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        label = data.get("label")
        if label not in stats:
            continue
        for person in data.get("persons", []):
            pred = "sitting" if is_sitting_pose(person) else "standing"
            stats[label][1] += 1
            if pred == label:
                stats[label][0] += 1
    return stats


def main():
    # 原始設定 + 逐步變嚴格的組合
    combos = [
        # (thigh, ext, torso, 說明)
        (0.55, 0.45, 0.5,  "目前設定（寬鬆）"),
        (0.65, 0.55, 0.7,  "稍嚴"),
        (0.75, 0.60, 0.9,  "中等嚴格"),
        (0.85, 0.70, 1.1,  "嚴格"),
        (0.95, 0.80, 1.3,  "非常嚴格"),
        (0.75, 0.60, 99,   "中等嚴格＋關閉torso備用"),
        (0.85, 0.70, 99,   "嚴格＋關閉torso備用"),
    ]

    print(f"{'設定':<24} {'thigh':>6} {'ext':>6} {'torso':>6} | {'坐姿召回':>12} {'站姿召回':>12} {'總正確率':>10}")
    print("-" * 95)
    for thigh_b, ext_b, torso_b, name in combos:
        s = replay(thigh_b, ext_b, torso_b)
        sit_c, sit_t = s["sitting"]
        st_c, st_t = s["standing"]
        total_c, total_t = sit_c + st_c, sit_t + st_t
        sit_r = f"{sit_c}/{sit_t} ({100*sit_c/sit_t:.0f}%)" if sit_t else "N/A"
        st_r = f"{st_c}/{st_t} ({100*st_c/st_t:.0f}%)" if st_t else "N/A"
        print(f"{name:<24} {thigh_b:>6} {ext_b:>6} {torso_b:>6} | {sit_r:>12} {st_r:>12} {100*total_c/total_t:>9.1f}%")


if __name__ == "__main__":
    main()
