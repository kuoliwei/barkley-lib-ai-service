"""
分析 posejson/ 中的標註樣本，驗證坐姿演算法並建議閾值

使用流程：
1. 執行 barkley_openvino.py
2. 坐著時按 C（標註 sitting）、站著時按 V（標註 standing）
   建議收集多種姿勢：正坐、駝背、翹腳、側身坐、站立、彎腰站…每種至少 3~5 筆
3. 執行 python analyze_captures.py
   會列出每筆樣本的特徵值、判斷結果、與標註是否一致，
   並根據兩類樣本的特徵分布建議新的決策邊界
"""

import glob
import json
import os

from pose_detector import (
    is_sitting_pose,
    reset_body_scale_history,
    THIGH_RATIO_BOUNDARY,
    EXTENSION_RATIO_BOUNDARY,
)

SMOOTH_WINDOW = 3  # 與 barkley_openvino.py 的設定一致


def load_samples(output_dir="posejson"):
    # 依檔名順序重放，模擬線上執行時肩寬中位數逐步累積的行為
    reset_body_scale_history()
    rows = []
    for fp in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"跳過 {fp}: {e}")
            continue

        label = data.get("label")
        for person in data.get("persons", []):
            pred, dbg = is_sitting_pose(person, return_debug=True)
            feats = dbg.get("features", {})
            rows.append({
                "file": os.path.basename(fp),
                "label": label,
                "pred": "sitting" if pred else "standing",
                "thigh_ratio": feats.get("thigh_ratio"),
                "extension_ratio": feats.get("extension_ratio"),
                "torso_ratio": feats.get("torso_ratio"),
                "shin_ratio": feats.get("shin_ratio"),
                "score": feats.get("score"),
                "fallback": feats.get("fallback"),
                "reason": dbg.get("reason"),
                "missing": dbg.get("missing_keypoints", []),
            })
    return rows


def fmt(v, width=8):
    if v is None:
        return " " * (width - 4) + "N/A "
    return f"{v:>{width}.3f}"


def main():
    rows = load_samples()
    if not rows:
        print("posejson/ 中沒有樣本。請先在 barkley_openvino.py 中按 C/V 收集標註資料。")
        return

    print("=" * 110)
    print(f"{'檔案':<12} {'標註':<10} {'判斷':<10} {'thigh':>8} {'ext':>8} {'torso':>8} {'shin':>8} {'score':>8}  結果")
    print("-" * 110)

    correct = 0
    labeled = 0
    for r in rows:
        if r["label"]:
            labeled += 1
            ok = r["pred"] == r["label"]
            if ok:
                correct += 1
            mark = "OK" if ok else "MISMATCH!"
        else:
            mark = "(未標註)"
        fb = " [備用:軀幹]" if r.get("fallback") else ""
        print(f"{r['file']:<12} {str(r['label']):<10} {r['pred']:<10} "
              f"{fmt(r['thigh_ratio'])} {fmt(r['extension_ratio'])} {fmt(r['torso_ratio'])} "
              f"{fmt(r['shin_ratio'])} {fmt(r['score'])}  {mark}{fb}")
        if r["missing"]:
            print(f"{'':>34} 不可用關鍵點: {r['missing']}")

    print("=" * 100)

    if labeled:
        print(f"\n標註樣本準確率（單幀）: {correct}/{labeled} ({100.0 * correct / labeled:.1f}%)")

    # ---- 模擬時間平滑（最近 SMOOTH_WINDOW 次多數決）後的準確率 ----
    from collections import deque
    history = deque(maxlen=SMOOTH_WINDOW)
    smooth_correct = 0
    smooth_labeled = 0
    for r in rows:
        history.append(1 if r["pred"] == "sitting" else 0)
        smoothed = "sitting" if sum(history) * 2 > len(history) else "standing"
        if r["label"]:
            smooth_labeled += 1
            if smoothed == r["label"]:
                smooth_correct += 1
    if smooth_labeled:
        print(f"標註樣本準確率（平滑 N={SMOOTH_WINDOW}）: {smooth_correct}/{smooth_labeled} ({100.0 * smooth_correct / smooth_labeled:.1f}%)")

    # ---- 依標註分組統計特徵分布 ----
    for feat in ("thigh_ratio", "extension_ratio", "score"):
        sit_vals = [r[feat] for r in rows if r["label"] == "sitting" and r[feat] is not None]
        stand_vals = [r[feat] for r in rows if r["label"] == "standing" and r[feat] is not None]

        if sit_vals or stand_vals:
            print(f"\n[{feat}]")
        if sit_vals:
            print(f"  sitting : min={min(sit_vals):.3f}  mean={sum(sit_vals)/len(sit_vals):.3f}  max={max(sit_vals):.3f}  (n={len(sit_vals)})")
        if stand_vals:
            print(f"  standing: min={min(stand_vals):.3f}  mean={sum(stand_vals)/len(stand_vals):.3f}  max={max(stand_vals):.3f}  (n={len(stand_vals)})")

        # 兩類都有樣本時，建議決策邊界（兩類平均值的中點）
        if sit_vals and stand_vals and feat in ("thigh_ratio", "extension_ratio"):
            suggested = (sum(sit_vals) / len(sit_vals) + sum(stand_vals) / len(stand_vals)) / 2
            current = THIGH_RATIO_BOUNDARY if feat == "thigh_ratio" else EXTENSION_RATIO_BOUNDARY
            gap_lo, gap_hi = max(stand_vals), min(sit_vals)
            print(f"  目前邊界: {current}   建議邊界: {suggested:.3f}")
            if gap_lo < gap_hi:
                print(f"  兩類完全分離，安全區間: {gap_lo:.3f} ~ {gap_hi:.3f}")
            else:
                print(f"  警告: 兩類特徵有重疊（standing 最大 {gap_lo:.3f} > sitting 最小 {gap_hi:.3f}），單靠此特徵無法完全區分")


if __name__ == "__main__":
    main()
