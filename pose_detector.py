"""
坐姿檢測演算法 - 俯視角（overhead / bird's-eye view）版本

俯視角幾何原理（與側視角完全不同，這是先前版本判斷相反的原因）：

- 站立：身體垂直於地面、沿相機光軸方向延伸 → 透視收縮（foreshortening），
  膝蓋與腳踝的投影都擠在髖部附近，整個骨架在影像上縮成一小團
  （大約只有肩寬大小）。膝-髖影像距離「小」。
- 坐著：大腿呈水平 → 俯視能看到大腿的完整長度，膝蓋投影離髖部很遠
  （實測約 1~2 倍肩寬）；小腿垂直 → 腳踝投影靠近膝蓋。膝-髖影像距離「大」。

因此主要特徵是「下肢投影長度相對身體寬度的比例」：
- 大腿投影比 (thigh_ratio) 大 → 坐姿
- 大腿投影比小（被透視收縮）→ 站姿

所有距離都用 2D 歐氏距離計算，與人在畫面中面向的方向無關
（先前版本只用 y 座標差，人面向左右時會失效）。

參考文獻：
- FlyPose (arXiv 2601.05747): 空拍俯視角姿態估計，指出俯視自我遮擋
  會讓部分關鍵點不可靠，需搭配置信度過濾
- Top-View Omnidirectional HPE (CVPR 2023W): 俯視影像的透視收縮特性
"""

import math
import statistics
from collections import deque

# ============ 可調參數（用 analyze_captures.py 依標註資料校正） ============
KP_CONF_THRESHOLD = 0.3          # 關鍵點置信度門檻（低於此視為不可見）
THIGH_RATIO_BOUNDARY = 0.55      # 大腿投影比決策邊界（大於此偏向坐姿）
EXTENSION_RATIO_BOUNDARY = 0.45  # 髖-踝伸展比決策邊界
THIGH_WEIGHT = 2.0               # 大腿投影比權重（主要特徵）
EXTENSION_WEIGHT = 1.0           # 髖-踝伸展比權重（輔助特徵）
TORSO_RATIO_BOUNDARY = 0.5       # 軀幹投影比邊界（下肢不可見時的備用判斷）
                                 # 深彎腰坐姿會遮住膝蓋，但軀幹傾斜 → 肩-髖投影大
                                 # 站立時軀幹垂直 → 肩-髖投影趨近 0
SCALE_HISTORY_SIZE = 60          # 肩寬中位數的滑動視窗大小（幀數）
PROXIMITY_RATIO_BOUNDARY = 1.25  # 近距檢查：當幀肩寬 / 坐姿基準中位數 超過此值
                                 # → 肩膀比坐姿時更靠近俯視相機 = 站立
                                 # 實測：坐姿 0.98~1.07，站立（含站著彎腰）1.3~2.0
SCALE_GATE_LOW = 0.75            # 寫入閘門下限：比值低於此的幀（鬼影/垃圾偵測）
                                 # 不寫入歷史，避免拉低基準造成連鎖誤判
RECALIBRATE_AFTER_REJECTS = 30   # 連續 N 幀被閘門擋下 → 基準可能已失效
                                 # （換人/污染），自動重置歷史重新校正
SCALE_WARMUP_MIN = 10            # 暖機期：歷史累積滿 N 筆前不啟動閘門，
                                 # 照單全收。避免第一顆種子（可能是異常值）
                                 # 鎖死基準帶造成連鎖拒收

# ---- 身體尺度校正 ----
# 當幀肩寬會因身體傾斜/旋轉在幾秒內擺盪 ±30%，直接當分母是噪音放大器。
# 改用滑動視窗中位數：對單幀姿勢變化穩定，且換人入座後視窗會逐漸
# 適應新體型（維持體型無關性，不綁死特定的人）。
_shoulder_width_history = deque(maxlen=SCALE_HISTORY_SIZE)
_scale_reject_streak = 0  # 連續被寫入閘門擋下的幀數（自癒重置用）


def reset_body_scale_history():
    """清空肩寬歷史（換場景/換人/離線分析開始時呼叫）"""
    global _scale_reject_streak
    _shoulder_width_history.clear()
    _scale_reject_streak = 0


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _center(points):
    """回傳多個有效點的中心；points 為 (x, y) 列表"""
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points))


def is_sitting_pose(person_keypoints, verbose=False, return_debug=False):
    """
    判斷一個人是否坐著 - 俯視角版本

    參數：
        person_keypoints: dict，包含 keypoints 列表
            每個 keypoint 為 {"name", "x", "y"}，可選 "confidence"
        verbose: bool，是否打印詳細過程
        return_debug: bool，是否返回調試信息

    返回：
        如果 return_debug=True：返回 (bool, dict)
        否則：返回 bool
    """
    debug_info = {
        "features": {},
        "missing_keypoints": [],
        "result": None,
        "reason": None
    }

    def _fail(reason):
        debug_info["result"] = False
        debug_info["reason"] = reason
        if verbose:
            print(f"Result: NOT SITTING ({reason})")
        if return_debug:
            return False, debug_info
        return False

    # ---- 提取關鍵點（過濾無效點：(0,0) 或低置信度） ----
    kps = {}
    all_kps_with_conf = {}  # 保留所有關鍵點含信心度供後續檢查
    try:
        for kp in person_keypoints['keypoints']:
            x, y = kp['x'], kp['y']
            conf = kp.get('confidence')
            all_kps_with_conf[kp['name']] = conf if conf is not None else 1.0
            if x == 0 and y == 0:
                debug_info["missing_keypoints"].append(kp['name'])
                continue
            if conf is not None and conf < KP_CONF_THRESHOLD:
                debug_info["missing_keypoints"].append(f"{kp['name']}(conf={conf:.2f})")
                continue
            kps[kp['name']] = (x, y)
    except (KeyError, TypeError):
        return _fail("Invalid keypoint format")

    # ---- 檢查下半身信心度：若過半低於 0.5，則判定為 NOT SITTING ----
    lower_body_parts = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    low_conf_count = sum(1 for part in lower_body_parts if all_kps_with_conf.get(part, 0) < 0.5)
    if low_conf_count > len(lower_body_parts) / 2:
        debug_info["features"] = {
            "body_scale": None,
            "shoulder_width": None,
            "lower_body_confidence": f"{low_conf_count}/{len(lower_body_parts)} 低於 0.5",
        }
        debug_info["result"] = False
        debug_info["reason"] = "Not Sitting (下半身信心度過低)"
        if verbose:
            print(f"下半身信心度過低: {low_conf_count}/{len(lower_body_parts)} < 0.5 → NOT SITTING")
        if return_debug:
            return False, debug_info
        return False

    # ---- 身體尺度：優先用肩寬（俯視時肩膀最不會被遮擋），其次髖寬 ----
    shoulder_width = None
    if 'left_shoulder' in kps and 'right_shoulder' in kps:
        shoulder_width = _dist(kps['left_shoulder'], kps['right_shoulder'])

    hip_width = None
    if 'left_hip' in kps and 'right_hip' in kps:
        hip_width = _dist(kps['left_hip'], kps['right_hip'])

    # ---- 近距檢查：站立時肩膀靠近俯視相機，肩寬投影暴增 ----
    global _scale_reject_streak

    # 暖機期內（樣本不足）基準還不可信，不計算比值、不啟動閘門
    proximity_ratio = None
    if shoulder_width and len(_shoulder_width_history) >= SCALE_WARMUP_MIN:
        baseline = statistics.median(_shoulder_width_history)
        if baseline > 1:
            proximity_ratio = shoulder_width / baseline

    # 自癒重置：連續太多幀落在閘門外（不論偏高或偏低），
    # 代表基準本身可能已失效（換人 / 被污染），清空歷史重新校正
    in_gate = (proximity_ratio is None or
               SCALE_GATE_LOW <= proximity_ratio <= PROXIMITY_RATIO_BOUNDARY)
    if in_gate:
        _scale_reject_streak = 0
    else:
        _scale_reject_streak += 1
        if _scale_reject_streak >= RECALIBRATE_AFTER_REJECTS:
            _shoulder_width_history.clear()
            _scale_reject_streak = 0
            proximity_ratio = None  # 舊基準作廢，本幀成為新基準的種子
            in_gate = True

    if proximity_ratio is not None and proximity_ratio > PROXIMITY_RATIO_BOUNDARY:
        debug_info["features"] = {
            "shoulder_width": round(shoulder_width, 2),
            "scale_baseline": round(statistics.median(_shoulder_width_history), 2),
            "proximity_ratio": round(proximity_ratio, 3),
            "thigh_ratio": None,
            "extension_ratio": None,
            "torso_ratio": None,
            "score": None,
            "fallback": "proximity",
        }
        debug_info["result"] = False
        debug_info["reason"] = f"Standing (肩寬為坐姿基準的 {proximity_ratio:.2f} 倍，靠近相機=站立)"
        if verbose:
            print(f"近距檢查: 肩寬 {shoulder_width:.1f} / 基準 {debug_info['features']['scale_baseline']} = {proximity_ratio:.2f} (> {PROXIMITY_RATIO_BOUNDARY}) → STANDING")
        if return_debug:
            return False, debug_info
        return False

    # 用滑動視窗中位數作為穩定的身體尺度（避免單幀肩寬抖動污染比值）
    # 雙向寫入閘門：只有比值在 [SCALE_GATE_LOW, PROXIMITY_RATIO_BOUNDARY] 內
    # （或尚無基準）的幀才寫入歷史。偏低的幀（鬼影/垃圾偵測）擋在門外，
    # 基準不會被拉歪；被擋的幀仍交給後面的幾何評分判斷
    if shoulder_width and shoulder_width > 1 and in_gate:
        _shoulder_width_history.append(shoulder_width)

    if _shoulder_width_history:
        body_scale = statistics.median(_shoulder_width_history)
    elif shoulder_width:
        body_scale = shoulder_width
    else:
        body_scale = hip_width

    if not body_scale or body_scale < 1:
        return _fail("無法取得身體尺度（肩膀與髖部關鍵點皆不可用）")

    # ---- 軀幹投影比：肩-髖中心距離 / 身體尺度 ----
    # 站立（軀幹垂直）→ 肩髖投影重疊，比值小
    # 坐著前傾/彎腰（軀幹傾斜）→ 比值大
    shoulder_center = _center([kps[k] for k in ('left_shoulder', 'right_shoulder') if k in kps])
    hip_center = _center([kps[k] for k in ('left_hip', 'right_hip') if k in kps])
    torso_ratio = None
    if shoulder_center and hip_center:
        torso_ratio = _dist(shoulder_center, hip_center) / body_scale

    # ---- 特徵 1：大腿投影比（主要特徵） ----
    thigh_lengths = {}
    for side in ('left', 'right'):
        hip = kps.get(f'{side}_hip')
        knee = kps.get(f'{side}_knee')
        if hip and knee:
            thigh_lengths[side] = _dist(hip, knee)

    if not thigh_lengths:
        # 大腿不可見有兩種可能：
        # (a) 俯視站立 → 身體垂直自我遮擋，此時軀幹投影也小
        # (b) 坐著深彎腰 → 頭與背遮住膝蓋，但軀幹傾斜 → 軀幹投影大
        # 用軀幹投影比區分兩者
        if torso_ratio is not None:
            is_sitting = torso_ratio > TORSO_RATIO_BOUNDARY
            debug_info["features"] = {
                "body_scale": round(body_scale, 2),
                "shoulder_width": round(shoulder_width, 2) if shoulder_width else None,
                "hip_width": round(hip_width, 2) if hip_width else None,
                "torso_ratio": round(torso_ratio, 3),
                "thigh_ratio": None,
                "extension_ratio": None,
                "score": None,
                "fallback": "torso_ratio",
            }
            debug_info["result"] = is_sitting
            debug_info["reason"] = ("Sitting (下肢被遮擋，但軀幹傾斜=彎腰坐姿)" if is_sitting
                                    else "Standing (下肢不可見且軀幹投影小=站立)")
            if verbose:
                print(f"下肢不可見，改用軀幹投影比判斷: {torso_ratio:.3f} (邊界 {TORSO_RATIO_BOUNDARY})")
                print(f"Result: {'SITTING' if is_sitting else 'STANDING'}")
            if return_debug:
                return is_sitting, debug_info
            return is_sitting
        return _fail("下肢與軀幹關鍵點皆不足，無法判斷")

    thigh_ratio = (sum(thigh_lengths.values()) / len(thigh_lengths)) / body_scale

    # ---- 特徵 2：髖-踝伸展比（輔助特徵） ----
    ankle_center = _center([kps[k] for k in ('left_ankle', 'right_ankle') if k in kps])

    extension_ratio = None
    if hip_center and ankle_center:
        extension_ratio = _dist(hip_center, ankle_center) / body_scale

    # ---- 特徵 3：小腿投影比（僅記錄供分析，不參與判斷） ----
    shin_lengths = {}
    for side in ('left', 'right'):
        knee = kps.get(f'{side}_knee')
        ankle = kps.get(f'{side}_ankle')
        if knee and ankle:
            shin_lengths[side] = _dist(knee, ankle)
    shin_ratio = None
    if shin_lengths:
        shin_ratio = (sum(shin_lengths.values()) / len(shin_lengths)) / body_scale

    # ---- 加權評分 ----
    score = THIGH_WEIGHT * (thigh_ratio - THIGH_RATIO_BOUNDARY)
    if extension_ratio is not None:
        score += EXTENSION_WEIGHT * (extension_ratio - EXTENSION_RATIO_BOUNDARY)

    is_sitting = score > 0

    debug_info["features"] = {
        "body_scale": round(body_scale, 2),
        "shoulder_width": round(shoulder_width, 2) if shoulder_width else None,
        "hip_width": round(hip_width, 2) if hip_width else None,
        "proximity_ratio": round(proximity_ratio, 3) if proximity_ratio is not None else None,
        "torso_ratio": round(torso_ratio, 3) if torso_ratio is not None else None,
        "thigh_px": {k: round(v, 2) for k, v in thigh_lengths.items()},
        "thigh_ratio": round(thigh_ratio, 3),
        "extension_ratio": round(extension_ratio, 3) if extension_ratio is not None else None,
        "shin_ratio": round(shin_ratio, 3) if shin_ratio is not None else None,
        "score": round(score, 3),
        "boundaries": {
            "thigh_ratio_boundary": THIGH_RATIO_BOUNDARY,
            "extension_ratio_boundary": EXTENSION_RATIO_BOUNDARY,
        }
    }

    if verbose:
        print("=" * 60)
        print("俯視角坐姿檢測")
        print("=" * 60)
        print(f"身體尺度: {body_scale:.1f} px (肩寬={shoulder_width and round(shoulder_width,1)}, 髖寬={hip_width and round(hip_width,1)})")
        print(f"大腿投影: {', '.join(f'{k}={v:.1f}px' for k, v in thigh_lengths.items())}")
        print(f"大腿投影比: {thigh_ratio:.3f}  (邊界 {THIGH_RATIO_BOUNDARY}, 大於→坐姿)")
        if extension_ratio is not None:
            print(f"髖-踝伸展比: {extension_ratio:.3f}  (邊界 {EXTENSION_RATIO_BOUNDARY}, 大於→坐姿)")
        if shin_ratio is not None:
            print(f"小腿投影比: {shin_ratio:.3f}  (僅記錄)")
        if debug_info["missing_keypoints"]:
            print(f"不可用關鍵點: {debug_info['missing_keypoints']}")
        print(f"加權總分: {score:.3f}  (>0 → 坐姿)")
        print(f"Result: {'SITTING' if is_sitting else 'STANDING'}")
        print("=" * 60)

    debug_info["result"] = is_sitting
    debug_info["reason"] = "Sitting" if is_sitting else "Standing"

    if return_debug:
        return is_sitting, debug_info
    return is_sitting


def analyze_pose_json(json_file_path):
    """
    分析 JSON 檔案中的姿態

    參數：
        json_file_path: str，JSON 檔案路徑

    返回：
        list: 包含每個人的坐姿判斷結果
    """
    import json

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return []

    results = []
    for person in data.get('persons', []):
        person_id = person.get('person_id', 'unknown')
        is_sitting = is_sitting_pose(person, verbose=True)
        results.append({
            'person_id': person_id,
            'is_sitting': is_sitting,
            'label': data.get('label'),
            'timestamp': data.get('timestamp', 'unknown')
        })
        print()

    return results


if __name__ == "__main__":
    import glob
    import os

    print("=" * 60)
    print("Sitting Pose Detection Algorithm Test")
    print("=" * 60)

    json_files = sorted(glob.glob(os.path.join("posejson", "*.json")))
    if not json_files:
        print("posejson/ 中沒有 JSON 檔案")

    all_results = []
    for jf in json_files:
        print(f"\n--- {jf} ---")
        all_results.extend(analyze_pose_json(jf))

    print("=" * 60)
    print("Detection Results Summary")
    print("=" * 60)
    for result in all_results:
        status = "SITTING" if result['is_sitting'] else "NOT SITTING"
        label = result.get('label')
        label_str = f" (標註: {label})" if label else ""
        print(f"Person {result['person_id']}: {status}{label_str}")
