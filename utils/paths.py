import os
import sys

def resource_path(relative_path: str) -> str:

    # Project root folder when running from source code
    project_root = os.path.dirname(os.path.dirname(__file__))

    # When running as a PyInstaller-created .exe, resources live in _MEIPASS
    base_path = getattr(sys, '_MEIPASS', project_root)

    return os.path.join(base_path, relative_path)
