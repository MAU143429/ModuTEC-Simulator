import customtkinter as ctk
import numpy as np
import tkinter.filedialog as fd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SignalPlot(ctk.CTkFrame):
    def __init__(self, master, title="Signal"):
        super().__init__(master)
        self.pack_propagate(False)
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot(np.zeros(1000))

    def plot(self, y):
        self.ax.clear()
        self.ax.plot(y, color='green')
        self.ax.set_title(self.title)
        self.canvas.draw()

class ModuTECSim(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ModuTEC Sim")
        self.geometry("1000x700")

        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # === Left Panel (Controls) ===
        control_frame = ctk.CTkFrame(self)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        control_frame.grid_rowconfigure((0,1,2,3,4,5,6), weight=1)

        ctk.CTkLabel(control_frame, text="ModuTEC Sim", font=("Arial", 20, "bold")).pack(pady=(10, 20))

        self.upload_btn = ctk.CTkButton(control_frame, text="Upload Audio File", command=self.load_audio)
        self.upload_btn.pack(pady=5)

        self.mod_combo = ctk.CTkComboBox(control_frame, values=["AM", "FM", "ASK", "FSK"])
        self.mod_combo.set("Select Modulation Type")
        self.mod_combo.pack(pady=5)

        self.sample_rate_input = ctk.CTkEntry(control_frame, placeholder_text="Sample Rate (Hz)")
        self.sample_rate_input.insert(0, "44100")
        self.sample_rate_input.pack(pady=5)

        self.carrier_freq_input = ctk.CTkEntry(control_frame, placeholder_text="Carrier Frequency (Hz)")
        self.carrier_freq_input.insert(0, "100000")
        self.carrier_freq_input.pack(pady=5)

        self.apply_btn = ctk.CTkButton(control_frame, text="Apply Changes", command=self.apply_changes)
        self.apply_btn.pack(pady=10)

        # === Right Panel (Dashboard + Plots) ===
        dashboard_frame = ctk.CTkFrame(self)
        dashboard_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        dashboard_frame.grid_rowconfigure((1,2,3), weight=1)
        dashboard_frame.grid_columnconfigure(0, weight=1)

        self.info_label = ctk.CTkLabel(dashboard_frame, text="Simulator Dashboard\nNo audio loaded", anchor="w", justify="left")
        self.info_label.grid(row=0, column=0, sticky="ew", pady=(5, 10))

        self.original_plot = SignalPlot(dashboard_frame, title="Señal Original")
        self.original_plot.grid(row=1, column=0, sticky="nsew", pady=5)

        self.modulated_plot = SignalPlot(dashboard_frame, title="Señal Modulada")
        self.modulated_plot.grid(row=2, column=0, sticky="nsew", pady=5)

        self.demodulated_plot = SignalPlot(dashboard_frame, title="Señal Demodulada")
        self.demodulated_plot.grid(row=3, column=0, sticky="nsew", pady=5)

    def load_audio(self):
        file_path = fd.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if file_path:
            self.audio_file = file_path
            self.info_label.configure(text=f"Simulator Dashboard\nAudio File: {file_path.split('/')[-1]}")
            t = np.linspace(0, 1, 44100)
            signal = np.sin(2 * np.pi * 440 * t)
            self.original_plot.plot(signal)

    def apply_changes(self):
        mod_type = self.mod_combo.get()
        sample_rate = self.sample_rate_input.get()
        carrier_freq = self.carrier_freq_input.get()
        print(f"Applied Modulation: {mod_type}, Sample Rate: {sample_rate}, Carrier: {carrier_freq}")

        t = np.linspace(0, 1, 44100)
        modulated = np.cos(2 * np.pi * 440 * t)
        demodulated = np.sin(2 * np.pi * 220 * t)

        self.modulated_plot.plot(modulated)
        self.demodulated_plot.plot(demodulated)

if __name__ == '__main__':
    app = ModuTECSim()
    app.mainloop()
