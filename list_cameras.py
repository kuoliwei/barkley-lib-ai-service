import cv2
import subprocess


def list_cameras_directshow_friendly_name():
    """從 WMI 取得所有攝像頭名稱（顯示名稱和 Windows 相機 app 一致）"""
    try:
        ps_command = r"""
# 查詢所有 USB 影像設備 - 只取 Name 欄位（和 Windows 相機 app 一致）
$devices = Get-WmiObject -Query "select * from Win32_PnPEntity where (Name like '%camera%' or Name like '%webcam%' or Name like '%video%' or Name like '%Elgato%')" -ErrorAction SilentlyContinue

$cameras = @()
foreach ($device in $devices) {
    $name = $device.Name

    # 過濾掉明顯不是相機的設備
    if ($name -notlike '*audio*' -and $name -notlike '*microphone*' -and $name -notlike '*speaker*' -and
        $name -notlike '*network*' -and $name -notlike '*adapter*' -and $name -notlike '*wireless*' -and
        $name -notlike '*graphics*' -and $name -notlike '*display*' -and $name -notlike '*gpu*' -and
        $name -notlike '*intel*' -and $name -notlike '*nvidia*' -and $name -notlike '*amd*') {

        $cameras += $name
    }
}

# 輸出結果，每行一個相機名稱
$cameras | ForEach-Object { Write-Host $_ }
"""

        result = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True,
            timeout=10
        )

        cameras = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line:
                cameras.append(line)

        return cameras

    except Exception as e:
        print(f"攝像頭查詢失敗: {e}")
        return []


def list_all_cameras():
    """列出所有可用的攝像頭及其詳細信息"""
    print("=" * 70)
    print("可用攝像頭")
    print("=" * 70)

    # 取得攝像頭 FriendlyName（和 Windows 相機 app 一致）
    camera_names = list_cameras_directshow_friendly_name()

    found = False
    opencv_index = 0

    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            found = True

            # 獲取解析度和幀率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # 嘗試讀取一幀以驗證攝像頭是否真正可用
            ret, frame = cap.read()
            status = "✓ 可用" if ret else "✗ 無法讀取幀"

            # 獲取攝像頭名稱
            name = camera_names[opencv_index] if opencv_index < len(camera_names) else "未知"
            opencv_index += 1

            print(f"\nIndex: {index}")
            print(f"  名稱: {name}")
            print(f"  解析度: {width}×{height}")
            print(f"  FPS: {fps}")
            print(f"  狀態: {status}")

            cap.release()

    print("\n" + "=" * 70)
    if not found:
        print("未找到可用攝像頭")
    else:
        print("列表完成")
    print("=" * 70)


if __name__ == "__main__":
    list_all_cameras()
