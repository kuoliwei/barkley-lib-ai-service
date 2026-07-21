"""
Practical examples of how to use DirectShow camera name retrieval in your project
"""

from list_cameras import list_cameras_directshow_friendly_name
import subprocess


# ============================================================================
# EXAMPLE 1: Get camera name for OpenCV index
# ============================================================================

def get_camera_name_for_index(opencv_index):
    """
    Get the camera name for a specific OpenCV camera index.

    WARNING: Registry order may not match OpenCV index order perfectly.
    This is a best-effort mapping.

    Args:
        opencv_index: OpenCV camera index (0, 1, 2, ...)

    Returns:
        str: Camera name or "Unknown" if not found
    """
    cameras = list_cameras_directshow_friendly_name()
    if opencv_index < len(cameras):
        return cameras[opencv_index]
    return "Unknown"


# ============================================================================
# EXAMPLE 2: Find camera index by name
# ============================================================================

def find_camera_index_by_name(target_name):
    """
    Find OpenCV camera index by name substring.

    Args:
        target_name: Camera name or substring to search for
                     (e.g., "Logitech", "Elgato", "BRIO")

    Returns:
        int: OpenCV index if found, -1 if not found
    """
    cameras = list_cameras_directshow_friendly_name()

    for index, camera_name in enumerate(cameras):
        if target_name.lower() in camera_name.lower():
            return index

    return -1  # Not found


# ============================================================================
# EXAMPLE 3: List all cameras with their properties
# ============================================================================

def list_all_cameras_with_properties():
    """
    List all cameras found in registry with their properties from the registry.
    """
    cameras = list_cameras_directshow_friendly_name()

    print("=" * 70)
    print("Available Cameras")
    print("=" * 70)

    if not cameras:
        print("No cameras found")
        return

    for index, name in enumerate(cameras):
        print(f"\nCamera {index}: {name}")
        # Could add more properties here if needed


# ============================================================================
# EXAMPLE 4: Real-world usage with OpenCV
# ============================================================================

def open_camera_by_name(camera_name_substring):
    """
    Open a camera by name substring using OpenCV.

    Args:
        camera_name_substring: Part of the camera name to match

    Returns:
        cv2.VideoCapture object or None if not found
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed")
        return None

    index = find_camera_index_by_name(camera_name_substring)

    if index == -1:
        print(f"Camera matching '{camera_name_substring}' not found")
        return None

    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Opened camera {index}: {get_camera_name_for_index(index)}")
        return cap
    else:
        print(f"Failed to open camera {index}")
        return None


# ============================================================================
# EXAMPLE 5: Check if specific camera is available
# ============================================================================

def is_camera_available(camera_name_substring):
    """
    Check if a camera matching the name is available.

    Args:
        camera_name_substring: Part of the camera name to match

    Returns:
        bool: True if camera found, False otherwise
    """
    return find_camera_index_by_name(camera_name_substring) != -1


# ============================================================================
# EXAMPLE 6: Get all camera names
# ============================================================================

def get_all_camera_names():
    """
    Get a list of all available camera names.

    Returns:
        list: Camera names in registry order
    """
    return list_cameras_directshow_friendly_name()


# ============================================================================
# EXAMPLE 7: Find Logitech camera specifically
# ============================================================================

def get_logitech_camera_index():
    """
    Find Logitech camera index if available.

    Returns:
        int: Camera index or -1 if not found
    """
    return find_camera_index_by_name("Logitech")


# ============================================================================
# EXAMPLE 8: Find Elgato camera specifically
# ============================================================================

def get_elgato_camera_index():
    """
    Find Elgato camera index if available.

    Returns:
        int: Camera index or -1 if not found
    """
    return find_camera_index_by_name("Elgato")


# ============================================================================
# EXAMPLE 9: Verify camera is working
# ============================================================================

def verify_camera_working(camera_name_substring):
    """
    Verify a camera is working by attempting to capture a frame.

    Args:
        camera_name_substring: Part of the camera name to match

    Returns:
        bool: True if camera works, False otherwise
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed")
        return False

    index = find_camera_index_by_name(camera_name_substring)
    if index == -1:
        return False

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False

    ret, frame = cap.read()
    cap.release()

    return ret


# ============================================================================
# EXAMPLE 10: Print camera information table
# ============================================================================

def print_camera_info_table():
    """
    Print a nicely formatted table of camera information.
    """
    cameras = list_cameras_directshow_friendly_name()

    print("\n" + "=" * 70)
    print("Camera Registry Information")
    print("=" * 70)
    print(f"{'Index':<10} {'Camera Name':<50}")
    print("-" * 70)

    for index, name in enumerate(cameras):
        print(f"{index:<10} {name:<50}")

    print("=" * 70)
    print(f"Total: {len(cameras)} camera(s)")
    print("=" * 70 + "\n")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Camera Name Retrieval - Usage Examples\n")

    # Example 1: Get camera name for index 0
    print("1. Get camera name for index 0:")
    print(f"   {get_camera_name_for_index(0)}\n")

    # Example 2: Find camera by name
    print("2. Find camera index for 'Logitech':")
    print(f"   Index: {find_camera_index_by_name('Logitech')}\n")

    # Example 3: List all cameras
    print("3. List all cameras:")
    for name in get_all_camera_names():
        print(f"   - {name}")
    print()

    # Example 4: Check if Logitech camera available
    print("4. Is Logitech camera available?")
    print(f"   {is_camera_available('Logitech')}\n")

    # Example 5: Print table
    print("5. Camera information table:")
    print_camera_info_table()

    # Example 6: Verify camera working
    print("6. Verify first camera is working:")
    first_camera = get_all_camera_names()[0] if get_all_camera_names() else None
    if first_camera:
        working = verify_camera_working(first_camera)
        print(f"   Camera '{first_camera}' working: {working}\n")
