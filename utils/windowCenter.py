def center_window(window, windowWidth, windowHeight):
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    x = int((width // 2) - (windowWidth // 2))
    y = int((height // 2) - (windowHeight // 2))
    return window.geometry(f"{windowWidth}x{windowHeight}+{x}+{y}")
