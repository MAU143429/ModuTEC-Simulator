# +++ al inicio
import customtkinter as ctk   # <- Faltaba este import

from utils.styles import *
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

class VerticalRightToolbar(NavigationToolbar2Tk):
    toolitems = [t for t in NavigationToolbar2Tk.toolitems
                 if t and t[0] in ("Home", "Pan", "Zoom", "Save")]

    def __init__(self, canvas, window):
        super().__init__(canvas, window, pack_toolbar=False)

        try:
            self.configure(bg=MAIN_BACKGROUND_COLOR, bd=0, relief="flat", highlightthickness=0)
        except Exception:
            pass

        # ↓↓↓ Esto evita el “status bar” y cualquier hueco asociado
        try:
            if hasattr(self, "_message_label") and self._message_label:
                self._message_label.destroy()
        except Exception:
            pass
        self.set_message = lambda *a, **k: None  # anula cualquier intento de escribir estado

    def _Button(self, text, image_file, toggle, command):
        b = super()._Button(text, image_file, toggle, command)
        try:
            b.configure(bg=MAIN_BACKGROUND_COLOR,
                        activebackground=MAIN_BACKGROUND_COLOR,
                        bd=0, relief="flat", highlightthickness=0)
        except Exception:
            pass
        b.pack_configure(side="top", pady=6, padx=0)
        return b

    def _Spacer(self):
        s = ctk.CTkFrame(self, fg_color=MAIN_BACKGROUND_COLOR, height=4, width=32, corner_radius=0)
        s.pack(side="top", pady=4)
        return s
