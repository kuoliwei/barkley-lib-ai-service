import subprocess


def get_directshow_friendly_names():
    """直接查詢 DirectShow Property Bag 中的 FriendlyName"""
    try:
        ps_command = r"""
# 直接從 DirectShow 的 PropertyBag 取得 FriendlyName
$objShell = New-Object -ComObject WScript.Shell

# 使用 PowerShell COM 介面查詢 DirectShow 設備
Add-Type -AssemblyName System.Runtime.InteropServices

$cameras = @()

# 這是查詢 DirectShow 設備最直接的方法
$devEnum = New-Object -ComObject WbemScripting.SWbemLocator
$service = $devEnum.ConnectServer($null, "root\cimv2")

# 查詢 USB 視訊設備（包括攝像頭）
$query = "select * from Win32_PnPEntity where (Name like '%camera%' or Name like '%webcam%' or Name like '%video%' or PNPClass='Camera' or PNPClass='Image')"
$devices = $service.ExecQuery($query)

foreach ($device in $devices) {
    $name = $device.Name

    # 嚴格過濾
    if ($name -notlike '*audio*' -and $name -notlike '*microphone*' -and $name -notlike '*speaker*' -and
        $name -notlike '*network*' -and $name -notlike '*adapter*' -and $name -notlike '*wireless*' -and
        $name -notlike '*graphics*' -and $name -notlike '*display*' -and $name -notlike '*gpu*' -and
        $name -notlike '*intel*' -and $name -notlike '*nvidia*' -and $name -notlike '*amd*') {

        Write-Host "Name: $name"
        Write-Host "Description: $($device.Description)"
        Write-Host "PNPDeviceID: $($device.PNPDeviceID)"
        Write-Host "Class: $($device.PNPClass)"
        Write-Host "---"
    }
}
"""

        result = subprocess.run(
            ['powershell', '-Command', ps_command],
            capture_output=True,
            text=True,
            timeout=10
        )

        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)

    except Exception as e:
        print(f"查詢失敗: {e}")


if __name__ == "__main__":
    get_directshow_friendly_names()
